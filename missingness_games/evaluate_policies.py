"""Evaluate each observation condition separately with the same frozen predictor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from .panels import macro_accuracy

from .cli import dump, read, tensor_batch, load_model
from .metrics import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deletion-fraction", type=float)
    parser.add_argument("--max-gap", type=int)
    parser.add_argument("--heldout",help="Frozen pre-test held-out bank JSON")
    parser.add_argument("--shuffle-masks", action="store_true", help="Count-preserving temporal intervention within originally observed positions")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.deletion_fraction is not None and not 0 <= args.deletion_fraction <= 1:
        parser.error("deletion-fraction must lie in [0,1]")
    if args.max_gap is not None and args.max_gap < 1:
        parser.error("max-gap must be positive")
    saved = torch.load(args.model, map_location=args.device, weights_only=True)
    config = dict(saved["config"])
    if args.deletion_fraction is not None:
        config["deletion_fraction"] = args.deletion_fraction
    if args.max_gap is not None:
        config["max_gap"] = args.max_gap
    model,_=load_model(args.model,args.device)
    model.c.update(config)
    model.c["evaluation_history_quantile"]=.6
    data = read(args.input)
    if "group" not in data:
        raise ValueError("Evaluation requires independent subject/run group IDs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    excluded=saved.get('training_groups',[])+saved.get('validation_groups',[])+saved.get('calibration_groups',[])
    if np.intersect1d(data['group'],excluded).size:raise ValueError('Evaluation groups overlap fitting/checkpoint/calibration groups')
    bank=json.loads(Path(args.heldout).read_text(encoding="utf-8-sig")) if args.heldout else json.loads(Path(__file__).with_name("heldout.json").read_text(encoding="utf-8-sig"))
    represented=["independent","channel_burst","group_burst","history_burst"]
    heldout=[entry["name"] for entry in bank]
    conditions=["clean",*represented,*heldout]
    if len(set(conditions))!=len(conditions):raise ValueError("Unique condition names required")
    rows, condition_group_scores, condition_probabilities = [], [], {}
    for index, condition in enumerate(conditions):
        model.c['evaluation_policy']=bank[index-5] if index>=5 else None
        # The same seed, input order and batch size give all variants the same masks.
        torch.manual_seed(args.seed)
        probabilities, masks = [], []
        with torch.no_grad():
            for start in range(0, len(data["x"]), args.batch_size):
                ids = np.arange(start, min(start + args.batch_size, len(data["x"])))
                batch = tensor_batch(data, ids, args.device)
                raw = batch["x"].float()
                observed = batch.get("mask", torch.isfinite(raw)).bool()
                values = torch.nan_to_num(raw)
                mask = observed.float() if index == 0 else model.policy(values, min(index-1,4), observed)
                if args.shuffle_masks and index > 0:
                    for row in range(len(mask)):
                        for channel in range(mask.shape[2]):
                            positions = observed[row, :, channel].nonzero().flatten()
                            order = torch.randperm(len(positions), device=mask.device)
                            mask[row, positions, channel] = mask[row, positions[order], channel].clone()
                prediction = model({"x": values, "mask": mask})["probability"]
                if "classification_temperature" in saved:
                    prediction = (prediction.clamp_min(1e-10).log() / saved["classification_temperature"]).softmax(-1)
                probabilities.append(prediction.cpu().numpy())
                masks.append(mask.cpu().numpy())
        probability, mask = np.concatenate(probabilities), np.concatenate(masks)
        condition_probabilities[condition]=probability
        metrics = measure({"probability": probability}, data)
        group_scores = [
            macro_accuracy(data["y"][data["group"] == group],
                                    probability[data["group"] == group].argmax(-1))
            for group in np.unique(data["group"])
        ]
        condition_group_scores.append(group_scores)
        native = np.asarray(data.get("mask", np.isfinite(data["x"]))).astype(bool) & np.isfinite(data["x"])
        introduced = native & ~mask.astype(bool)
        gap = np.zeros((len(mask), mask.shape[2]), dtype=int)
        maximum_gap = 0
        for step in range(mask.shape[1]):
            gap = (gap + 1) * (~mask[:, step].astype(bool))
            maximum_gap = max(maximum_gap, int(gap.max()))
        metrics.update(condition=condition, subject_macro_accuracy=float(np.mean(group_scores)),
                       observed_fraction=float(mask.mean()), native_missing_fraction=float(1-native.mean()),
                       introduced_missing_fraction=float(introduced.mean()), maximum_effective_gap=maximum_gap)
        rows.append(metrics)
        np.savez_compressed(output / (condition + ".npz"), probability=probability,
                            mask=mask, y=data["y"], group=data["group"])
    from .panels import panels
    panel=panels(condition_probabilities,data,float(model.native_boundary),model.rare_classes.nonzero().flatten().cpu().tolist(),represented,heldout)
    panel['heldout_bank']=bank
    dump(output/'panels.json',panel)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[row for row in panel['panels'] if row['heldout_worst'] is not None]
    figure,axis=plt.subplots(figsize=(9,4))
    axis.bar([row['completeness']+':'+row['rarity'] for row in selected],[row['heldout_worst'] for row in selected])
    axis.set_ylabel('Subject macro worst-policy accuracy');axis.tick_params(axis='x',labelrotation=25)
    figure.tight_layout();figure.savefig(output/'panels.png',dpi=160);plt.close(figure)
    dump(output / "metrics.json", rows)
    dump(output / "summary.json", {
        "clean_subject_macro_accuracy": rows[0]["subject_macro_accuracy"],
        "represented_worst_subject_macro_accuracy": float(np.min(np.asarray(condition_group_scores[1:5]), axis=0).mean()),
        "heldout_worst_subject_macro_accuracy": float(np.min(np.asarray(condition_group_scores[5:]), axis=0).mean()),
        "conditions": [r["condition"] for r in rows], "heldout_bank":bank, "represented_history_quantile":.6,
        "seed": args.seed, "batch_size": args.batch_size, "count_preserving_temporal_shuffle": args.shuffle_masks,
        "deletion_fraction": config.get("deletion_fraction", .4),
        "max_gap": config.get("max_gap", data["x"].shape[1] // 4),
    })


if __name__ == "__main__":
    main()

