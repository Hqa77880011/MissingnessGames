# MissingnessGames

Implementation of **Missingness-Mechanism Games for Robust Multisensor Representation Learning**. This repository implements the observation-policy bank, detached allocation game, clean-risk selection rule and subject-level evaluation in the current manuscript. The implementation and generated functional fixtures are separate from the paper's reported experimental evidence.

## Installation

Python >=3.10, PyTorch, NumPy, SciPy, scikit-learn and Matplotlib:

```bash
python -m pip install -e .
```

## Data

Download the released [UCI HAR signals](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and/or [PAMAP2 Protocol recordings](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring). The exporter reads the extracted releases without downloading data or filling native gaps.

```bash
python -m missingness_games.data --dataset har --root data/UCI_HAR_Dataset --output data/har.npz --seed 13
missingness-games prepare --input data/har.npz --output data/har
python -m missingness_games.data --dataset pamap2 --root data/PAMAP2_Dataset --test-subject 101 --output data/pamap101.npz --seed 13
missingness-games prepare --input data/pamap101.npz --output data/pamap101
```

HAR uses total acceleration and body gyroscope only, in that order (six channels), supplied 128-sample windows, official test subjects, development subjects 7/14/22, and seeded caps of 50 windows per subject/activity. It excludes body-acceleration and precomputed feature files. PAMAP2 uses subjects 101–108, the stated 12 Protocol activities, 256-sample windows/128 stride on the native grid, and 18 channels: ±16g acceleration and gyroscope from hand/chest/ankle. Windows stay within one recording/activity and reject discontinuous clocks; NaNs remain missing. The test subject and next cyclic development subject are separate; training support is capped at 100 windows per subject/activity. The exporter writes channel groups in `*.config.json` and actual support in `*.support.json`; these counts must be checked against the selected release rather than assumed to match the paper ledger. A class with fewer than 20 test windows is identifiable from that support file.

For custom data, supply numeric NPZ `x[N,T,D]`, integer `y[N]`, `group[N]` (subject/run), optional binary `mask[N,T,D]`, and explicit `split[N]`. Codes 0/1/2 mean train/development/test; 0/1/2/3 mean train/development/optional calibration/test. A group cannot cross splits. Without codes, the demonstration convention assigns ordered groups 60/15/10/15. Real benchmark experiments should use their explicit subject protocol. Normalization, class weights and 75th/60th history thresholds are fitted from originally observed training values only and stored in the checkpoint.

## Training and inference

Train the clean control first; its frozen development cross-entropy defines the reference in the paper's guard. Select full checkpoints by the largest of four aggregate policy development risks, subject to clean balanced CE <= reference balanced CE + 0.015. Training fails with an infeasible configuration when no checkpoint passes. The guard never changes allocation weights.

```bash
missingness-games train --input data/har --output runs/clean --config data/har.config.json --variant baseline --epochs 150 --batch-size 64 --seed 13
missingness-games train --input data/har --output runs/full --config data/har.config.json --reference runs/clean/model.pt --variant full --epochs 150 --batch-size 64 --seed 13
missingness-games evaluate --input data/har/test.npz --model runs/full/model.pt --output runs/full/clean
```

The default network has three 64-channel temporal convolution blocks, masked pooling and one classifier. Inputs are revealed standardized values, masks and elapsed-time channels; inference runs one frozen predictor without policy execution or memory updates. The implementation uses kernel 3 with dilations 1/2/4 and elapsed times divided by window length, which the manuscript does not prescribe. Earlier GRU checkpoints are incompatible with this TCN revision.

Default allocation uses uniform positive reference weights, temperature 0.2, EMA beta 0.9, robust weight 1, AdamW learning rate 0.001 and weight decay 0.0001. Use `--set robust_weight=0.5`, `--set temperature=0.1`, etc. for declared development searches. `config.json` records the primary defaults; data-derived configuration files add the correct dimensions and channel groups. Defaults already match `config.json`; the CLI accepts one configuration file and `--set` overrides. The four policies are independent deletion, channel bursts, group bursts and history-triggered group bursts. Each complete proposed action passes the budget/gap check; infeasible proposals become all-observed actions before native masks are applied. History uses only previous effective reveals. The missingness ceiling is 0.4 and max gap T/4; realized deletion rates can be lower. HAR/PAMAP2 burst durations/probabilities follow the manuscript's configuration. The frozen history threshold changes at update 51. Memory starts at the first policy-loss vector, then follows the detached EMA; weights use `softmax(log(reference) + memory/temperature)`.

## Evaluation

```bash
python -m missingness_games.evaluate_policies --input data/har/test.npz --model runs/full/model.pt --output runs/full/policies --seed 13 --batch-size 64
python -m missingness_games.evaluate_policies --input data/har/test.npz --model runs/full/model.pt --output runs/full/shuffled --seed 13 --batch-size 64 --shuffle-masks
```

Evaluation rejects groups used for fitting, checkpoint selection or optional calibration. Clean accuracy, represented minimum and held-out minimum are calculated **within each subject**, using macro class accuracy (mean recall over supported classes), then averaged across subjects. Each condition saves probabilities, effective masks, labels, groups, realized native/introduced missing fractions and maximum effective gap. Batch size, order and seed must match across methods to share mask draws. The two explicit held-out rules reverse the history trigger and change burst duration; these are implementation choices for the manuscript's incompletely specified held-out distribution. `--max-gap` can declare a longer-outage stress and `--deletion-fraction` changes the allowance ceiling. Neither option fits on test labels. The shuffle intervention permutes introduced deletion among originally observed positions per channel, preserving every window/channel deletion count and native gaps.

`accuracy`, `balanced_accuracy`, `macro_f1`, multiclass Brier, NLL and 15-bin top-confidence ECE are also saved. Optional temperature calibration is separate from the primary paper protocol and requires an independent fourth partition:

```bash
missingness-games calibrate --input data/custom/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

## Baselines and ablations

Use the same prepared data, initialization seed, reference, raw-window batch size and update budget. `--variant` supports `full`, `baseline`, `fixed_bank` (uniform bank), `group_dro`, `fixed_hard`, `no_memory`, `random_only`, `permuted` (weight assignment), `independent_masks` (four IID views), and `unguarded`. Guarded and unguarded training use identical training randomness and stopping risk independently of eligibility; loading a clean reference preserves the learner RNG, and weight permutation uses a separate generator. The fixed-hard policy is selected on training data in a separate clean-model pilot:

```bash
missingness-games pilot --input data/har/train.npz --model runs/clean/model.pt --output runs/pilot.json --seed 13
missingness-games train --input data/har --output runs/fixed_hard --config data/har.config.json --reference runs/clean/model.pt --pilot runs/pilot.json --variant fixed_hard --seed 13
missingness-games train --input data/har --output runs/dro --config data/har.config.json --reference runs/clean/model.pt --variant group_dro --set dro_step_size=0.01 --seed 13
missingness-games train --input data/har --output runs/uniform --config data/har.config.json --reference runs/clean/model.pt --variant fixed_bank --seed 13
```

The external GRU-D, BRITS, SAITS, TimesNet and BIOT models are not bundled. Native-completeness/activity-rarity panels require the paper's frozen stratum specification and analysis of exported predictions; this implementation supplies support/masks without fabricating those experimental panels. This compact reference retains the four policy-view graphs until weighting, yielding the stated gradient while using more memory than the manuscript's sequential accumulation strategy. No training-work or hardware-performance result is prefilled.

## Outputs and verification

| Module / endpoint | Entry | Output / command |
|---|---|---|
| Released signals, native masks, subject splits | `missingness_games/data.py` | `python -m missingness_games.data ...` |
| Causal bank, TCN, elapsed channels, EMA/anchored response | `missingness_games/method.py` | `missingness-games train ...` |
| Relative clean-CE guard, baseline/DRO/ablations | `missingness_games/cli.py` | `model.pt`, `training.json`, `history.json` |
| Subject macro accuracy and represented/held-out minima | `missingness_games/evaluate_policies.py` | per-condition NPZ, `metrics.json`, `summary.json` |
| Count-preserving temporal intervention | `evaluate_policies.py` | `--shuffle-masks` |
| Scalar evaluation comparison figures | `missingness_games/cli.py` | `missingness-games plot --input runs --output runs/figures` |

`history.json` records policy losses, memory, detached weights and realized introduced rates for each training update. Checkpoints retain all learned parameters and training-only transforms. Full UCI HAR/PAMAP2 experiments and literature-baseline comparisons were not run in this revision.

```bash
python verify.py --output runs/functional-check
```

Actual CPU check environment: Python 3.13.5, PyTorch 2.7.1+cu118, NumPy 2.1.3, seed 13, one thread. The check uses generated inputs, ten variants for one epoch, raw-release schema fixtures, finite forward/loss/backward, encoder/head gradients and parameter changes, exact first-memory/EMA response, whole-action fallback, budget/gap/causality/hidden-value invariants, checkpoint equality, clean and policy evaluation, calibration, a training-only pilot, count-preserving shuffle and hand-calculated Brier/ECE/subject metrics. The functional fixture declares guard tolerance 2.0 solely to exercise every variant in one epoch; primary experiments retain 0.015.
