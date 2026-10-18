# Missingness-Mechanism Games for Robust Multisensor Representation Learning

Implementation of **Missingness-Mechanism Games for Robust Multisensor Representation Learning**.

Causal observation policies share a per-sequence deletion budget and maximum channel-gap constraint. Random, persistent, history-dependent, and stage-periodic policies select deletions using only past acquired observations. A mask-aware GRU learns against the KL-regularized adversarial distribution over policy losses, updated from an EMA memory. The baseline uses clean observations; fixed-bank and no-memory controls change only the game update.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,T,D]` contains multichannel time series and integer `y[N]` sequence classes. Pad sequences with a separate externally fixed convention and keep subjects/runs in one group. The bank has no access to the current unobserved value when it decides a mask.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
missingness-games demo --output data/example.npz --count 96 --config config.json
missingness-games prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
missingness-games train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
missingness-games evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
missingness-games plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
missingness-games calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `fixed_bank`, `no_memory`, `random_only`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
missingness-games train --input data/prepared --output runs/baseline --config config.json --variant baseline
missingness-games evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Evaluating observation policies

Inference applies one frozen predictor to the observations and optional `mask[N,T,D]` supplied in the input. The training policy bank and loss memory are not executed as a test-time ensemble. Native missing entries may use a zero mask or NaN values; hidden values do not enter the predictor. Validation checkpoint selection uses clean/native-observation cross entropy in this compact implementation; the paper's separate clean-risk eligibility guard is not implemented.

Evaluate the represented conditions separately, with identical masks for all trained variants:

```bash
python -m missingness_games.evaluate_policies --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/policies --seed 13 --batch-size 16
```

Use the same evaluation seed, batch size, deletion fraction and maximum gap for every variant, including the clean-trained baseline. `--deletion-fraction` and `--max-gap` set a declared stress condition. The command saves one prediction/mask archive per condition, clean and per-policy subject-mean balanced accuracy, and the minimum represented-policy endpoint. This is not an unseen-policy result. `group` must identify independent subjects/runs, not individual overlapping windows. Existing `evaluate` evaluates only the supplied observation condition.
