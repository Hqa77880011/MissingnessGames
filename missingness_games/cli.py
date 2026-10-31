"""Train and evaluate the finite-bank observation game on group-disjoint NPZ data."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .method import Model, VARIANTS, demo
from .metrics import measure


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def tensor_batch(data, indices, device):
    return {key: torch.as_tensor(value[indices], device=device) for key, value in data.items()
            if value.ndim and len(value) == len(data['x']) and value.dtype.kind in 'biuf'}


def prepare(source, output):
    data = read(source)
    n = len(data['x'])
    if data['x'].ndim != 3 or data['y'].shape != (n,) or data['y'].dtype.kind not in 'iu' or (data['y'] < 0).any():
        raise ValueError('Require x[N,T,D] and integer sequence labels y[N]')
    groups = data.get('group', np.arange(n))
    if 'split' in data:
        codes = sorted(np.unique(data['split']).tolist())
        if codes not in ([0, 1, 2], [0, 1, 2, 3]):
            raise ValueError('Use split codes 0/1/2 or 0/1/2/3')
        masks = [data['split'] == code for code in codes]
    else:
        unique = np.unique(groups)
        first, second, third = [int(f * len(unique)) for f in (.6, .75, .85)]
        masks = [np.isin(groups, part) for part in (unique[:first], unique[first:second], unique[second:third], unique[third:])]
    names = ('train', 'validation', 'test') if len(masks) == 3 else ('train', 'validation', 'calibration', 'test')
    for i, (name, selected) in enumerate(zip(names, masks)):
        if not selected.any():
            raise ValueError('Every declared split must be nonempty')
        if any(np.intersect1d(groups[selected], groups[other]).size for other in masks[:i]):
            raise ValueError('A subject/run group crosses split boundaries')
        values = {key: value[selected] if key not in ('class_duration','activity_codes') and value.ndim and len(value) == n else value for key, value in data.items() if key != 'split'}
        values['group'] = groups[selected]
        np.savez_compressed(output / f'{name}.npz', **values)


@torch.no_grad()
def validate(model, data, batch_size, device, seed):
    model.eval()
    clean, risks = 0., np.zeros(4)
    devices = [torch.device(device).index or 0] if device.startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0, len(data['x']), batch_size):
            indices = np.arange(start, min(start + batch_size, len(data['x'])))
            clean_sum, policy_sum = model.selection_metrics(tensor_batch(data, indices, device))
            clean += clean_sum; risks += policy_sum
    return clean / len(data['x']), risks / len(data['x'])


def train(args, config, output):
    if min(args.epochs, args.batch_size, args.patience, args.lr) <= 0:
        raise ValueError('Positive training budgets and learning rate required')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = read(Path(args.input) / 'train.npz')
    validation = read(Path(args.input) / 'validation.npz')
    if np.intersect1d(data['group'], validation['group']).size:
        raise ValueError('Training and validation groups overlap')
    config.setdefault('input_shape', list(data['x'].shape[1:]))
    config.setdefault('classes', int(data['y'].max()) + 1)
    config.setdefault('width', 64)
    model = Model(config).to(args.device)
    model.prepare(data)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=args.device, weights_only=True)['state'])
    reference_ce = None
    if args.variant not in ('baseline', 'unguarded'):
        if not args.reference:
            raise ValueError('Guarded training requires --reference from a clean-trained baseline')
        reference = torch.load(args.reference, map_location=args.device, weights_only=True)
        if set(reference['training_groups']) != set(np.unique(data['group']).tolist()) or set(reference['validation_groups']) != set(np.unique(validation['group']).tolist()):
            raise ValueError('Reference must use the same training and development groups')
        devices = [torch.device(args.device).index or 0] if args.device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            clean_model = Model(reference['config']).to(args.device)
            clean_model.load_state_dict(reference['state'])
            reference_ce = validate(clean_model, validation, args.batch_size, args.device, args.seed)[0]
    if args.variant == 'fixed_hard' and 'fixed_hard_index' not in config:
        raise ValueError('fixed_hard requires a policy index selected by a separate training-only pilot')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.get('weight_decay', 1e-4))
    history, best, stale, saved = [], float('inf'), 0, False
    best_stopping = float('inf')
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); total = 0.; diagnostics = []
        for indices in np.array_split(np.random.permutation(len(data['x'])), max(1, int(np.ceil(len(data['x']) / args.batch_size)))):
            batch = tensor_batch(data, indices, args.device)
            prediction = model(batch); loss = model.objective(batch, prediction)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(indices)
            if model.last_diagnostics:
                diagnostics.append(model.last_diagnostics)
        clean, risks = validate(model, validation, args.batch_size, args.device, args.seed)
        tolerance = float(config.get('clean_guard_tolerance', .015))
        if tolerance < 0:
            raise ValueError('clean_guard_tolerance must be nonnegative')
        eligible = reference_ce is None or clean <= reference_ce + tolerance
        score = clean if args.variant == 'baseline' else float(risks.max())
        history.append(dict(epoch=epoch + 1, training_loss=total / len(data['x']), clean_validation_ce=clean,
            reference_clean_ce=reference_ce, validation_policy_ce=risks.tolist(), validation_risk=score,
            eligible=eligible, allocation=diagnostics))
        if eligible and score < best:
            best, saved = score, True
            torch.save(dict(state=model.state_dict(), config=config,
                training_groups=np.unique(data['group']).tolist(), validation_groups=np.unique(validation['group']).tolist(),
                reference_clean_ce=reference_ce, selected_epoch=epoch + 1), output / 'model.pt')
        if score < best_stopping:
            best_stopping, stale = score, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    dump(output / 'history.json', history)
    if not saved:
        raise RuntimeError('No checkpoint satisfies clean CE <= reference CE + tolerance; see history.json')
    dump(output / 'training.json', dict(seconds=time.perf_counter()-started, best_validation_loss=best,
        seed=args.seed, epochs=len(history), batch_size=args.batch_size, learning_rate=args.lr, variant=args.variant))


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = Model(checkpoint['config']).to(device).eval()
    model.load_state_dict(checkpoint['state'])
    return model, checkpoint


@torch.no_grad()
def evaluate(args, output):
    model, checkpoint = load_model(args.model, args.device)
    data = read(args.input)
    excluded = checkpoint['training_groups'] + checkpoint['validation_groups'] + checkpoint.get('calibration_groups', [])
    if np.intersect1d(data['group'], excluded).size:
        raise ValueError('Evaluation groups overlap fitting/selection/calibration groups')
    probabilities = []
    for start in range(0, len(data['x']), args.batch_size):
        indices = np.arange(start, min(start + args.batch_size, len(data['x'])))
        p = model(tensor_batch(data, indices, args.device))['probability']
        if 'classification_temperature' in checkpoint:
            p = (p.clamp_min(1e-10).log()/checkpoint['classification_temperature']).softmax(-1)
        probabilities.append(p.cpu().numpy())
    prediction = dict(probability=np.concatenate(probabilities))
    np.savez_compressed(output/'predictions.npz', **prediction)
    dump(output/'metrics.json', measure(prediction, data))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('demo', 'prepare', 'train', 'evaluate', 'calibrate', 'pilot', 'plot'):
        p = sub.add_parser(name)
        p.add_argument('--output', required=True); p.add_argument('--input'); p.add_argument('--model')
        p.add_argument('--config'); p.add_argument('--set', action='append', default=[])
        p.add_argument('--variant', choices=VARIANTS, default='full')
        p.add_argument('--seed', type=int, default=13); p.add_argument('--count', type=int, default=96)
        p.add_argument('--epochs', type=int, default=150); p.add_argument('--patience', type=int, default=20)
        p.add_argument('--batch-size', type=int, default=64); p.add_argument('--lr', type=float, default=.001)
        p.add_argument('--device', default='cpu'); p.add_argument('--init'); p.add_argument('--reference'); p.add_argument('--pilot')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else {}
    for setting in args.set:
        key, value = setting.split('=', 1); config[key] = json.loads(value)
    if args.pilot:
        config['fixed_hard_index'] = json.loads(Path(args.pilot).read_text(encoding='utf-8'))['fixed_hard_index']
    config['variant'] = args.variant
    config['seed'] = args.seed
    output = Path(args.output)
    (output.parent if args.command in ('demo', 'calibrate', 'pilot') else output).mkdir(parents=True, exist_ok=True)
    if args.command == 'demo':
        np.savez_compressed(output, **demo(args.seed, args.count, config))
    elif args.command == 'prepare':
        prepare(args.input, output)
    elif args.command == 'train':
        train(args, config, output)
    elif args.command == 'evaluate':
        evaluate(args, output)
    elif args.command == 'pilot':
        model, checkpoint = load_model(args.model, args.device)
        data = read(args.input)
        if set(np.unique(data['group']).tolist()) != set(checkpoint['training_groups']):
            raise ValueError('Policy pilot must use the clean control training groups only')
        _, risks = validate(model, data, args.batch_size, args.device, args.seed)
        dump(output, dict(fixed_hard_index=int(risks.argmax()), training_policy_ce=risks.tolist()))
    elif args.command == 'calibrate':
        from scipy.optimize import minimize_scalar
        from scipy.special import softmax
        model, checkpoint = load_model(args.model, args.device)
        data = read(args.input)
        if np.intersect1d(data['group'], checkpoint['training_groups']+checkpoint['validation_groups']).size:
            raise ValueError('Calibration groups overlap fitting/selection groups')
        with torch.no_grad():
            logits = model(tensor_batch(data, np.arange(len(data['x'])), args.device))['clean_logits'].cpu().numpy()
        result = minimize_scalar(lambda value: -np.log(softmax(logits/np.exp(value),axis=1)[np.arange(len(logits)), data['y']].clip(1e-10)).mean(), bounds=(-3,3), method='bounded')
        checkpoint['classification_temperature'] = float(np.exp(result.x))
        checkpoint['calibration_groups'] = np.unique(data['group']).tolist(); torch.save(checkpoint, output)
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        paths = list(Path(args.input).rglob('metrics.json'))
        records = [(path, json.loads(path.read_text(encoding='utf-8'))) for path in paths]
        records = [(path, value) for path, value in records if isinstance(value, dict)]
        if not records:
            raise ValueError('No scalar evaluated metric records')
        for key in sorted(set().union(*(value.keys() for _,value in records))):
            selected = [(path,value[key]) for path,value in records if isinstance(value.get(key),(int,float))]
            if not selected: continue
            figure, axis = plt.subplots(figsize=(9,4))
            axis.bar([str(path.parent.relative_to(args.input)) for path,_ in selected],[value for _,value in selected])
            axis.tick_params(axis='x',labelrotation=30);axis.set_ylabel(key);figure.tight_layout()
            figure.savefig(output/f'{key}.png',dpi=160);plt.close(figure)


if __name__ == '__main__': main()
