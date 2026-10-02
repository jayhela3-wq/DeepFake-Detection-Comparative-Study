from pathlib import Path

DATASET_DIR = Path("/kaggle/input/datasets/joy1008/projectdetectionframes/deepfake_frames")

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT/"datasets"/"manifest.csv"

SEED = 42
CLASS_NAMES = ["real", "fake"]