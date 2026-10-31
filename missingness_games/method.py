"""Causal observation bank and detached entropy-anchored allocation."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('full', 'baseline', 'fixed_bank', 'no_memory', 'random_only',
            'group_dro', 'fixed_hard', 'permuted', 'independent_masks', 'unguarded')


class Model(nn.Module):
    """Shared TCN receiving revealed values, observation masks and elapsed time."""
    def __init__(self, config):
        super().__init__()
        self.c = dict(config)
        self.c.setdefault('variant', 'full')
        features = config.get('input_shape', [128, 6])[-1]
        width = config.get('width', 64)
        classes = config.get('classes', 3)
        self.encoder = nn.Sequential(
            nn.Conv1d(3 * features, width, 3, padding=1), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=4, dilation=4), nn.ReLU())
        self.head = nn.Linear(width, classes)
        self.register_buffer('mean', torch.zeros(features))
        self.register_buffer('scale', torch.ones(features))
        self.register_buffer('class_weights', torch.ones(classes))
        self.register_buffer('memory', torch.zeros(4))
        self.register_buffer('memory_initialized', torch.tensor(False))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        reference = torch.tensor(config.get('reference_weights', [.25] * 4), dtype=torch.float32)
        if reference.shape != (4,) or (reference <= 0).any():
            raise ValueError('reference_weights must contain four positive entries')
        self.register_buffer('reference', reference / reference.sum())
        self.register_buffer('dro_weights', reference / reference.sum())
        self.register_buffer('history_threshold', torch.tensor(0.))
        self.register_buffer('history_shift_threshold', torch.tensor(0.))
        self.last_diagnostics = {}

    def prepare(self, data):
        """Fit normalization, class weights and history thresholds on training only."""
        raw = np.asarray(data['x'], dtype=np.float32)
        observed = np.asarray(data.get('mask', np.isfinite(raw)), dtype=bool) & np.isfinite(raw)
        means, scales = [], []
        for channel in range(raw.shape[-1]):
            values = raw[:, :, channel][observed[:, :, channel]]
            if not len(values):
                raise ValueError(f'No observed training values in channel {channel}')
            means.append(float(values.mean())); scales.append(max(float(values.std()), 1e-6))
        self.mean.copy_(self.mean.new_tensor(means)); self.scale.copy_(self.scale.new_tensor(scales))
        counts = np.bincount(data['y'].astype(int), minlength=len(self.class_weights))
        if (counts == 0).any():
            raise ValueError('All declared classes need training support')
        self.class_weights.copy_(self.class_weights.new_tensor(len(data['y']) / (len(counts) * counts)))
        standardized = np.nan_to_num((raw - np.asarray(means)) / np.asarray(scales))
        history = np.zeros(len(raw)); traces = []
        for time in range(raw.shape[1]):
            traces.append(history.copy())
            mask = observed[:, time]
            amplitude = (abs(standardized[:, time]) * mask).sum(1) / np.maximum(mask.sum(1), 1)
            history = np.where(mask.any(1), .9 * history + .1 * amplitude, history)
        values = np.concatenate(traces)
        self.history_threshold.fill_(float(np.quantile(values, .75)))
        self.history_shift_threshold.fill_(float(np.quantile(values, .60)))

    def normalized(self, x):
        return (torch.nan_to_num(x) - self.mean) / self.scale

    def policy(self, x, index, observed=None):
        """Choose each whole action before reading current values; reject infeasible proposals."""
        observed = torch.isfinite(x) if observed is None else observed.bool() & torch.isfinite(x)
        x = torch.where(observed, self.normalized(x), 0.)
        n, time, channels = x.shape
        fraction = float(self.c.get('deletion_fraction', .4))
        maximum = int(self.c.get('max_gap', max(1, time // 4)))
        if not 0 <= fraction <= 1 or maximum < 1:
            raise ValueError('Require deletion_fraction in [0,1] and max_gap >= 1')
        groups = self.c.get('channel_groups', [list(range(channels // 2)), list(range(channels // 2, channels))])
        groups = [group for group in groups if group]
        if sorted(channel for group in groups for channel in group) != list(range(channels)):
            raise ValueError('channel_groups must partition channels exactly once')
        gap = torch.zeros(n, channels, device=x.device, dtype=torch.long)
        budget = torch.full((n,), int(time * channels * fraction), device=x.device, dtype=torch.long)
        history = x.new_zeros(n)
        masks = []
        threshold = self.history_shift_threshold if self.updates + 1 >= self.c.get('history_shift_update', 51) else self.history_threshold
        for t in range(time):
            if index == 0:
                proposal = torch.rand(n, channels, device=x.device) < self.c.get('independent_probability', .4)
            elif index == 1:
                duration = int(self.c.get('channel_duration', max(1, time // 8)))
                trigger = torch.rand(n, channels, device=x.device) < self.c.get('channel_start_probability', 16 / max(3 * time, 1))
                proposal = ((gap > 0) & (gap < duration)) | ((gap == 0) & trigger)
            elif index in (2, 3, 4, 5):
                proposal = torch.zeros_like(gap, dtype=torch.bool)
                duration = int(self.c.get('group_duration', max(1, 3 * time // 16))) if index == 2 else maximum
                if index == 5:
                    duration = max(1, maximum // 2 + (t % max(1, maximum // 2)))
                probability = self.c.get('group_start_probability', 32 / max(9 * time, 1)) if index == 2 else self.c.get('history_start_probability', .1)
                for group in groups:
                    clock = gap[:, group].max(1).values
                    trigger = torch.rand(n, device=x.device) < probability
                    if index in (3, 4):
                        trigger &= history > threshold if index == 3 else history <= threshold
                    action = ((clock > 0) & (clock < duration)) | ((clock == 0) & trigger)
                    proposal[:, group] = action[:, None]
            else:
                raise ValueError('Unknown missingness policy')
            feasible = (proposal.sum(1) <= budget) & (~(proposal & (gap + 1 > maximum))).all(1)
            deletion = proposal & feasible[:, None]
            budget -= deletion.sum(1)
            gap = (gap + 1) * deletion
            effective = (~deletion) & observed[:, t]
            amplitude = (x[:, t].abs() * effective).sum(1) / effective.sum(1).clamp_min(1)
            history = torch.where(effective.any(1), .9 * history + .1 * amplitude, history)
            masks.append(effective.float())
        return torch.stack(masks, 1)

    def classify(self, x, mask):
        x = self.normalized(x)
        mask = mask.float()
        gap, elapsed = torch.zeros_like(mask[:, 0]), []
        for step in range(mask.shape[1]):
            gap = (gap + 1) * (1 - mask[:, step])
            elapsed.append(gap)
        features = torch.cat([torch.where(mask.bool(), x, 0.), mask, torch.stack(elapsed, 1) / mask.shape[1]], -1)
        code = self.encoder(features.transpose(1, 2)).transpose(1, 2)
        valid = mask.any(-1).float()
        pooled = (code * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)

    def forward(self, batch):
        raw = batch['x'].float()
        observed = batch.get('mask', torch.isfinite(raw)).bool() & torch.isfinite(raw)
        clean = self.classify(raw, observed)
        result = dict(probability=clean.softmax(-1), clean_logits=clean)
        if self.training and self.c['variant'] != 'baseline':
            indices = [0] * 4 if self.c['variant'] == 'independent_masks' else range(4)
            masks = [self.policy(raw, index, observed) for index in indices]
            result['policy_logits'] = torch.stack([self.classify(raw, mask) for mask in masks], 1)
            result['realized_missing_rates'] = torch.stack([(observed.float() - mask).mean() for mask in masks])
        return result

    def cross_entropy(self, logits, target):
        return F.cross_entropy(logits, target.long(), weight=self.class_weights, reduction='none')

    def selection_metrics(self, batch):
        """Return summed clean CE and maximum per-policy class-balanced validation CE."""
        raw = batch['x'].float()
        observed = batch.get('mask', torch.isfinite(raw)).bool() & torch.isfinite(raw)
        with torch.no_grad():
            clean = self.cross_entropy(self.classify(raw, observed), batch['y'])
            losses = torch.stack([self.cross_entropy(self.classify(raw, self.policy(raw, k, observed)), batch['y']) for k in range(4)], 1)
        return float(clean.sum()), losses.sum(0).cpu().numpy()

    def objective(self, batch, prediction):
        clean = self.cross_entropy(prediction['clean_logits'], batch['y']).mean()
        if self.c['variant'] == 'baseline':
            return clean
        losses = torch.stack([self.cross_entropy(prediction['policy_logits'][:, k], batch['y']).mean() for k in range(4)])
        beta = 0. if self.c['variant'] == 'no_memory' else float(self.c.get('memory', .9))
        temperature = float(self.c.get('temperature', .2))
        if not 0 <= beta < 1 or temperature <= 0:
            raise ValueError('Require memory in [0,1) and temperature > 0')
        if self.training:
            if not self.memory_initialized:
                self.memory.copy_(losses.detach()); self.memory_initialized.fill_(True)
            else:
                self.memory.mul_(beta).add_(losses.detach(), alpha=1 - beta)
            self.updates.add_(1)
        weights = (self.reference.log() + self.memory / temperature).softmax(0).detach()
        if self.c['variant'] == 'fixed_bank':
            weights = self.reference.detach()
        elif self.c['variant'] in ('random_only', 'fixed_hard'):
            index = 0 if self.c['variant'] == 'random_only' else int(self.c['fixed_hard_index'])
            weights = F.one_hot(torch.tensor(index, device=losses.device), 4).to(losses)
        elif self.c['variant'] == 'group_dro':
            if self.training:
                self.dro_weights.copy_((self.dro_weights.log() + self.c.get('dro_step_size', .01) * losses.detach()).softmax(0))
            weights = self.dro_weights.detach().clone()
        elif self.c['variant'] == 'permuted':
            weights = weights[torch.randperm(4, device=weights.device)]
        self.last_diagnostics = dict(policy_losses=losses.detach().cpu().tolist(),
            memory=self.memory.cpu().tolist(), weights=weights.cpu().tolist(),
            introduced_missing_rates=prediction['realized_missing_rates'].detach().cpu().tolist())
        return clean + float(self.c.get('robust_weight', 1.)) * (weights * losses).sum()


def demo(seed, count, config):
    """Generate labeled sequences solely for execution checks."""
    rng = np.random.default_rng(seed)
    shape = config.get('input_shape', [24, 6])
    x = rng.normal(size=(count, *shape)).astype(np.float32)
    classes = config.get('classes', 3)
    y = np.arange(count) % classes
    x[:, :, :classes] += np.eye(classes)[y][:, None, :]
    return dict(x=x, y=y.astype(np.int64), group=np.arange(count))
