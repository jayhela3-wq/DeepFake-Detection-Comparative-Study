import cv2
import numpy as np
import zipfile

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

def video_uid(video_path, dataset_dir):
    rel = Path(video_path).relative_to(dataset_dir).with_suffix("")
    return "_".join(rel.parts)

