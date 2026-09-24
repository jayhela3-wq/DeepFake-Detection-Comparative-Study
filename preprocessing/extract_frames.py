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

def build_zip(split, real_list, fake_list, output_dir, dataset_dir, frames_per_video=10, resize=None, jpeg_quality=95, workers=8):
    jobs = (
        [(video, "real") for video in real_list] +
        [(video, "fake") for video in fake_list]
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir/f"{split}.zip"

    frame_count = 0

    with(zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zip_file, ThreadPoolExecutor(max_workers=workers) as executor):
        results = executor.map(lambda job: read_frames(job, dataset_dir, frames_per_video, resize, jpeg_quality), jobs)

        for frames in tqdm(results, total=len(jobs), desc=split):
            for name, data in frames:
                zip_file.writestr(name, data)
                frame_count += 1


    size_gb = zip_path.stat().st_size/1e9

    print(
        f"{split}: {frame_count} frames ->"
        f"{zip_path.name} ({size_gb:.2f} GB)"
    )
