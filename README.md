# Flowers-17 classification with MobileNetV3-Small

COMP1818 Artificial Intelligence Applications coursework implementation.

This project compares an ImageNet-pretrained MobileNetV3-Small backbone using:

1. A frozen-feature linear classifier (linear probing).
2. Linear probing followed by full fine-tuning (LP-FT).
3. Linear probing followed by final-feature-block and classifier fine-tuning.

The backbone implementation and pretrained weights come from torchvision. LP-FT
is based on Kumar et al. (2022). See [REFERENCES.md](REFERENCES.md) for attribution.

## Recorded results

Official Flowers-17 split 1: 680 training, 340 validation and 340 test images.
Seed 42; checkpoints selected by validation cross-entropy.

| Method | Test accuracy | Macro-F1 | Correct / 340 |
|---|---:|---:|---:|
| Frozen baseline | 92.35% | 0.9232 | 314 |
| Full LP-FT | 92.65% | 0.9265 | 315 |
| Final-block extension | 92.94% | 0.9290 | 316 |

The extension's advantage over the baseline is two test images. One seed and one
split do not establish general superiority. Do not tune settings against the
observed test scores.

Saved configurations, histories, predictions, metrics and exported figures are
included under `results/`. Absolute paths in original experiment metadata describe
the machine used for those runs; they are retained as provenance, not setup paths.

## Repository contents

- `flower_data.py`: archive image loading and tensor preparation.
- `train_baseline.py`: frozen features and a linear classifier.
- `train_two_stage.py`: full or final-block fine-tuning.
- `train_extension.py`: final-block fine-tuning entry point.
- `evaluate_test.py`: held-out checkpoint evaluation.
- `plot_results.py`: saved-history and comparison plots.
- `check_project.py`: data checksums, image readability and pretrained model check.
- `requirements-lock.txt`: dependency versions recorded for the original environment.
- `data/download_manifest.json`, `models/source.json`: original asset URLs and SHA-256 hashes.

Python source files are copied unchanged from the coursework project.

## Setup

The original experiments used Python 3.12, PyTorch 2.2.2 and torchvision 0.17.2 on
an Intel Mac with CPU execution. The exact dependency record is included; package
availability and platform compatibility must be checked when recreating that environment.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

### Data and pretrained weights

Large assets are deliberately excluded. Restore them from the complete coursework
folder, or download the original assets using the source URLs in the manifests.
The scripts expect these paths relative to this README:

| Required file | Original source |
|---|---|
| `data/17flowers.tgz` | https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz |
| `data/datasplits.mat` | https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat |
| `models/mobilenet_v3_small-imagenet.pth` | https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth |

Keep the image archive compressed; the loader reads it directly. The original
Oxford README is included at `data/README.txt`.

After restoring the files, verify them:

```sh
python check_project.py
```

### Existing trained checkpoints

To evaluate the recorded runs without retraining, copy the following files from
the full coursework project into the same relative locations in this repository:

```text
results/baseline_20261112_030300_944563/baseline_model.pth
results/two_stage_20261112_030318_406056/two_stage_model.pth
results/final_block_20261112_030815_639295/final_block_model.pth
```

**Complete implementation, dataset and trained models:** add an accessible
OneDrive/Google Drive share link here before using this repository as the report's
implementation link. GitHub alone does not contain the large files listed above.

## Train new runs

```sh
python train_baseline.py
python train_two_stage.py --baseline results/YOUR_BASELINE_FOLDER
python train_extension.py --baseline results/YOUR_BASELINE_FOLDER
```

Use the same baseline folder for both fine-tuning methods. Each command prints
its new result directory. Defaults are 20 baseline epochs and 10 fine-tuning epochs,
seed 42, batch size 32, learning rates 0.001 and 0.00001, and no augmentation.
Fine-tuning supports `--patience N`; its default of zero disables early stopping.

## Evaluate

With the original checkpoints restored, the default command evaluates those runs:

```sh
python evaluate_test.py
```

For newly trained runs, provide all three folders together:

```sh
python evaluate_test.py --baseline results/YOUR_BASELINE_FOLDER --full results/YOUR_FULL_FOLDER --extension results/YOUR_EXTENSION_FOLDER
```

Paths are relative to the current terminal directory; absolute paths are accepted.
Evaluation writes a new result directory with metrics, predictions, confusion
matrices, plots and checkpoint hashes. Matrix rows are true classes and columns
are predictions, with numeric labels 0–16.

To rebuild saved figures without training or inference:

```sh
python plot_results.py
```

## Coursework access

The report should link to this repository and the complete artefact share. Ensure
the teaching team can access both. The report, its interpretation and final AI-use
declaration are maintained separately.
