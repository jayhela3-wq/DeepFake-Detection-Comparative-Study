# Dataset

The processed dataset used in this project is generated from FaceForensics++ C23.

The generated dataset contains:

- Train set
- Validation set
- Test set
- Real and Fake classes
- 10 frames per video

The processed dataset was published on Kaggle for all project members to use.

## manifest.csv

A list of every frame image in the dataset, with its label and split.
It has 70,000 rows and three columns:

| column | meaning |
|---|---|
| path | full Kaggle path of the image |
| label | 0 = real, 1 = fake |
| split | train, val or test |

Counts: train 49,000 / val 10,500 / test 10,500 (about 6 fake for every 1 real).

**Why it exists:** scanning the image folders takes minutes. Reading this
file takes a second, and all three of us train and test on exactly the same images.

**When it's used:** every time you load data for training or evaluation.
It is built once (see experiments/01_build_manifest.ipynb) and only rebuilt
if the split changes.

**How to use it:**
    from preprocessing.manifest import load_manifest
    paths, labels = load_manifest("train")

**Note:** the paths only work on Kaggle with the ProjectDetectionFrames
dataset attached.