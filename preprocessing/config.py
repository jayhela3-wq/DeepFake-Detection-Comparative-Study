from pathlib import Path


DATASET_DIR = Path("/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23")
ORIGINAL_DIR = DATASET_DIR / "original"
FAKE_DIR_NAMES = [
    "DeepFakeDetection", "Deepfakes", "Face2Face",
    "FaceShifter", "FaceSwap", "NeuralTextures",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

SPLITS_DIR = Path("splits")
OUTPUT_DIR = Path("/kaggle/working/face_crops")
SPLIT_ASSIGNMENT_PATH = Path("split_assignment.csv")

SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42

FRAMES_PER_VIDEO = 10
CROP_SIZE = 299
FACE_MARGIN = 0.30
JPEG_QUALITY = 95


READ_WORKERS = 4
VIDEO_BATCH_SIZE = 8


MIN_FACE_SIZE = 20
DETECTOR_THRESHOLDS = (0.6, 0.7, 0.7)
KEEP_ALL_FACES = True
