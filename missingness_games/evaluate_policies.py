"""Evaluate each observation condition separately with the same frozen predictor."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from .cli import dump, read, tensor_batch
from .method import Model
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
    model = Model(config).to(args.device).eval()
    model.load_state_dict(saved["state"])
    data = read(args.input)
    if "group" not in data:
        raise ValueError("Evaluation requires independent subject/run group IDs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, condition_group_scores = [], []
    for index, condition in enumerate(["clean", "random", "persistent", "history", "periodic"]):
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
                mask = observed.float() if index == 0 else model.policy(values, index - 1, observed)
                prediction = model({"x": values, "mask": mask})["probability"]
                if "classification_temperature" in saved:
                    prediction = (prediction.clamp_min(1e-10).log() / saved["classification_temperature"]).softmax(-1)
                probabilities.append(prediction.cpu().numpy())
                masks.append(mask.cpu().numpy())
        probability, mask = np.concatenate(probabilities), np.concatenate(masks)
        metrics = measure({"probability": probability}, data)
        group_scores = [
            balanced_accuracy_score(data["y"][data["group"] == group],
                                    probability[data["group"] == group].argmax(-1))
            for group in np.unique(data["group"])
        ]
        condition_group_scores.append(group_scores)
        metrics.update(condition=condition, subject_macro_accuracy=float(np.mean(group_scores)),
                       observed_fraction=float(mask.mean()))
        rows.append(metrics)
        np.savez_compressed(output / (condition + ".npz"), probability=probability,
                            mask=mask, y=data["y"], group=data["group"])
    dump(output / "metrics.json", rows)
    dump(output / "summary.json", {
        "clean_subject_macro_accuracy": rows[0]["subject_macro_accuracy"],
        "represented_worst_subject_macro_accuracy": float(np.min(np.asarray(condition_group_scores[1:]), axis=0).mean()),
        "conditions": [r["condition"] for r in rows],
        "seed": args.seed, "batch_size": args.batch_size,
        "deletion_fraction": config.get("deletion_fraction", .25),
        "max_gap": config.get("max_gap", 4),
    })


if __name__ == "__main__":
    main()

