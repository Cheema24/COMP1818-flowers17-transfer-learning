# Flower classification with MobileNetV3-Small

This is my coursework project for COMP1818 Artificial Intelligence Applications.
It uses a model called MobileNetV3-Small to sort flower photos into 17 categories.
The model starts with weights already trained on ImageNet.

The aim is to compare three ways of training it on the Flowers-17 dataset:

1. **Linear probing:** keep the model's image features fixed and train a new classifier.
2. **Full fine-tuning:** start with the trained classifier, then train the whole model.
3. **Final-block fine-tuning:** start with the same classifier, then train only the last feature block and the classifier.

The model and its starting weights come from torchvision. The two-stage training
method is based on Kumar et al. (2022). The sources are listed in
[REFERENCES.md](REFERENCES.md).

## Results

I used the first official dataset split: 680 images for training, 340 for
validation and 340 for testing. The random seed was 42. Each model was saved
when it had the lowest validation loss.

| Method | Test accuracy | Macro-F1 | Correct out of 340 |
|---|---:|---:|---:|
| Linear probing | 92.35% | 0.9232 | 314 |
| Full fine-tuning | 92.65% | 0.9265 | 315 |
| Final-block fine-tuning | 92.94% | 0.9290 | 316 |

Macro-F1 gives each flower category equal weight when combining precision and recall.

Final-block fine-tuning had the highest accuracy in this run, but it only got two
more images right than linear probing. More runs with different seeds and splits
would be needed to know whether it is consistently better. Any further tuning
should use the validation set, rather than the test results.

The settings, training history, predictions and graphs are in `results/`.
Some saved records include paths from the computer used for the experiments.
You do not need to use those paths on your own computer.

## Main files

| File | What it does |
|---|---|
| `flower_data.py` | Loads and prepares the flower images. |
| `train_baseline.py` | Trains the linear-probing baseline. |
| `train_two_stage.py` | Runs the second stage of fine-tuning. |
| `train_extension.py` | Runs final-block fine-tuning. |
| `evaluate_test.py` | Tests the three saved models. |
| `plot_results.py` | Makes graphs from the saved results. |
| `check_project.py` | Checks the data files and starting model weights. |
| `requirements-lock.txt` | Lists the package versions used for the project. |

The Python files are the same as those in the full coursework folder.

## Set up the project

The experiments ran on an Intel Mac using the CPU, Python 3.12,
PyTorch 2.2.2 and torchvision 0.17.2. The listed package versions record that
setup; you may need compatible versions for a different computer.

Open a terminal in the project folder and run:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

On Windows, use `.venv\Scripts\activate` to activate the environment instead.

### Add the data and starting weights

The large data and model files are not included on GitHub. Copy them from the
full coursework folder, or download them using the links below. Save each file
at the path shown.

| Save as | Download |
|---|---|
| `data/17flowers.tgz` | [Flower images](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz) |
| `data/datasplits.mat` | [Official dataset splits](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat) |
| `models/mobilenet_v3_small-imagenet.pth` | [ImageNet model weights](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth) |

Do not extract `17flowers.tgz`. The code reads the images straight from it.
The original dataset notes are in `data/README.txt`. The download links and file
checksums are also saved in `data/download_manifest.json` and `models/source.json`.

Once the files are in place, run:

```sh
python check_project.py
```

### Use the models already trained

To test the existing models without training them again, copy these files from
the full coursework folder into the matching folders here:

```text
results/baseline_20261112_030300_944563/baseline_model.pth
results/two_stage_20261112_030318_406056/two_stage_model.pth
results/final_block_20261112_030815_639295/final_block_model.pth
```

**Full project download:** share link to be added. This will provide the large
files that are not included in this repository.

## Train the models

```sh
python train_baseline.py
python train_two_stage.py --baseline results/YOUR_BASELINE_FOLDER
python train_extension.py --baseline results/YOUR_BASELINE_FOLDER
```

The first command prints the name of its new results folder. Replace
`YOUR_BASELINE_FOLDER` with that name in the next two commands. Both fine-tuning
methods should start from the same baseline.

The default settings are:

- 20 training epochs for the baseline and 10 for each fine-tuning method.
- Batch size 32 and random seed 42.
- Learning rate 0.001 for the baseline and 0.00001 for fine-tuning.
- No image augmentation.

An epoch is one pass through the training data. Fine-tuning also accepts
`--patience N` to stop after N epochs without a lower validation loss.
The default is zero, which runs all epochs.

## Test the models

After adding the original trained model files, run:

```sh
python evaluate_test.py
```

To test new runs, give it all three results folders:

```sh
python evaluate_test.py --baseline results/YOUR_BASELINE_FOLDER --full results/YOUR_FULL_FOLDER --extension results/YOUR_EXTENSION_FOLDER
```

The script saves the test scores, predictions, graphs and confusion matrices in
a new results folder. In each confusion matrix, rows show the true categories
and columns show the predicted categories. Categories are numbered 0 to 16.

Folder paths start from your current terminal location. Full paths also work.

To remake the saved graphs without training or testing again, run:

```sh
python plot_results.py
```

## Coursework files

The report is kept separately. It should link to this repository and the full
project download so the teaching team can access the code, data and trained models.
