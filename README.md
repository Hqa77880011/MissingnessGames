# MissingnessGames

Implementation of **Missingness-Mechanism Games for Robust Multisensor Representation Learning**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate synthetic multichannel time series with classes determined from sequence means.

```bash
missingness-games demo --output data/generated.npz --count 96 --config config.json --seed 13
missingness-games prepare --input data/generated.npz --output data/prepared
```

`x[N,T,D]` contains multichannel time series and integer `y[N]` sequence classes. Pad sequences with a separate externally fixed convention and keep subjects/runs in one group. The bank has no access to the current unobserved value when it decides a mask.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
missingness-games train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

## Evaluation

```bash
missingness-games evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

Optional confidence calibration uses its independent split:

```bash
missingness-games calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
missingness-games evaluate --input data/prepared/test.npz --model runs/full/calibrated.pt --output runs/full/calibrated_evaluation
```

For fixed-mask policy comparisons (represented and held-out families are reported separately):

```bash
python -m missingness_games.evaluate_policies --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/policies --seed 13 --batch-size 16
```

`clean_risk_cap=0.5` is fixed in `config.json`; training fails explicitly if no checkpoint satisfies the validation guard.

## Ablations

Variants: `full`, `baseline`, `fixed_bank`, `no_memory`, `random_only`. Reuse the prepared data and training seed:

```bash
missingness-games train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
missingness-games evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
missingness-games plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
`runs/full/policies/` contains per-condition masks/predictions, `metrics.json` and `summary.json`.
