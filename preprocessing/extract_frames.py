import cv2
import numpy as np
import zipfile

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

def video_uid(video_path, dataset_dir):
    rel = Path(video_path).relative_to(dataset_dir).with_suffix("")
    return "_".join(rel.parts)

def read_frames(job, dataset_dir, frames_per_video=10, resize=None, jpeg_quality=95):
    video_path, label = job

    uid = video_uid(video_path, dataset_dir)
    output=[]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open: ", video_path)
        return output

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return output

    frame_indices = np.linspace(0, total-1, frames_per_video, dtype=int)

    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        success, frame = cap.read()

        if not success:
            continue

        if resize:
            frame = cv2.resize(frame, resize)

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        if success:
            output.append((f"{label}/{uid}/frame_{i:03d}.jpg", buffer.tobytes())) 

    cap.release()

    return output