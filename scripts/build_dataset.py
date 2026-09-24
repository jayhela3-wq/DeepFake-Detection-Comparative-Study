from pathlib import Path
from preprocessing.find_videos import find_videos
from preprocessing.split_videos import split_videos
from preprocessing.extract_frames import build_zip

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

OUTPUT_DIR = Path("/kaggle/working/deepfake_frames")

JPEG_QUALITY = 95
WORKERS = 8

RESIZE = None

real_videos = find_videos(ORIGINAL_DIR)

fake_videos = []

for folder in FAKE_DIRS:
    fake_videos.append(find_videos(folder))

print("Real videos: ", len(real_videos))
print("Fake Videos: ", len(fake_videos))

real_train, real_val, real_test = split_videos(real_videos, train_ratio = TRAIN_RATIO, val_ratio = VAL_RATIO, seed = SEED )
fake_train, fake_val, fake_test = split_videos(fake_videos, train_ratio = TRAIN_RATIO, val_ratio = VAL_RATIO, seed = SEED)

print("\nREAL")
print("Train: ", len(real_train))
print("Validation: ", len(real_val))
print("Test: ", len(real_test))

print("\nFAKE")
print("Train: ", len(fake_train))
print("Validation: ", len(fake_val))
print("Test: ", len(fake_test))

build_zip("train", real_train, fake_train, OUTPUT_DIR, DATASET_DIR, frames_per_video = FRAMES_PER_VIDEO, resize = RESIZE, jpeg_quality = JPEG_QUALITY, workers = WORKERS)
build_zip("val", real_val, fake_val, OUTPUT_DIR, DATASET_DIR, frames_per_video = FRAMES_PER_VIDEO, resize = RESIZE, jpeg_quality = JPEG_QUALITY, workers = WORKERS)
build_zip("test", real_test, fake_test, OUTPUT_DIR, DATASET_DIR, frames_per_video = FRAMES_PER_VIDEO, resize = RESIZE, jpeg_quality = JPEG_QUALITY, workers = WORKERS)


