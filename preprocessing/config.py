from pathlib import Path

DATASET_DIR = Path("/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23")

ORIGINAL_DIR = DATASET_DIR/"original"

FAKE_DIRS = [
    DATASET_DIR/"DeepFakeDetection",
    DATASET_DIR/"Deepfakes",
    DATASET_DIR/"Face2Face",
    DATASET_DIR/"FaceShifter",
    DATASET_DIR/"FaceSwap",
    DATASET_DIR/"NeuralTextures"

]


SEED = 42
FRAMES_PER_VIDEO = 10

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

TEST_RATIO = 0.15

OUTPUT_DIR = Path("/kaggle/working/deepfake_frames")

JPEG_QUALITY = 95
WORKERS = 8

RESIZE = None