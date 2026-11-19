import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import (
    FRAMES_PER_VIDEO, JPEG_QUALITY, OUTPUT_DIR,
    READ_WORKERS, VIDEO_BATCH_SIZE,
)
from .face_detection import detect_batch, crop_face
from .manifest import save_split_manifest


def read_frames(video_path, frames_per_video=FRAMES_PER_VIDEO):
    
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        if not capture.isOpened():
            return frames
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return frames

        indices = np.linspace(0, total - 1, frames_per_video, dtype=int)
        for frame_index, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if success:
                frames.append(
                    (frame_index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
    finally:
        capture.release()
    return frames


def process_split(split_name, assignments, detector, output_dir=OUTPUT_DIR,
                  limit=None):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = assignments[assignments["split"] == split_name].copy()
    if limit is not None:
        subset = subset.sample(min(limit, len(subset)), random_state=0)

    jobs = list(subset.itertuples(index=False))
    chunks = [
        jobs[index:index + VIDEO_BATCH_SIZE]
        for index in range(0, len(jobs), VIDEO_BATCH_SIZE)
    ]

    zip_path = output_dir / f"{split_name}.zip"
    rows = []
    started = time.time()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        with ThreadPoolExecutor(max_workers=READ_WORKERS) as executor:
            for chunk in tqdm(chunks, desc=split_name):
                futures = [
                    executor.submit(read_frames, job.video_path)
                    for job in chunk
                ]
                for job, future in zip(chunk, futures):
                    frames = future.result()
                    if not frames:
                        print(f"No frames could be read: {job.video_path}")
                        continue

                    boxes_batch, probs_batch = detect_batch(
                        detector, [image for _, image in frames]
                    )

                    for frame_position, (frame_index, image) in enumerate(frames):
                        boxes = boxes_batch[frame_position]
                        probabilities = probs_batch[frame_position]

                        row = {
                            "path": "",
                            "label": int(job.label == "fake"),
                            "split": split_name,
                            "video_id": job.uid,
                            "frame_idx": frame_index,
                            "face_found": False,
                            "x1": np.nan, "y1": np.nan,
                            "x2": np.nan, "y2": np.nan,
                            "prob": np.nan,
                        }

                        if boxes is not None and len(boxes):
                            areas = [
                                max(0, box[2] - box[0]) *
                                max(0, box[3] - box[1])
                                for box in boxes
                            ]
                            best_index = int(np.argmax(areas))
                            try:
                                face = crop_face(image, boxes[best_index])
                                success, encoded = cv2.imencode(
                                    ".jpg",
                                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                                )
                                if success:
                                    
                                    member = (
                                        f"{job.label}/{job.uid}/"
                                        f"frame_{frame_index:03d}.jpg"
                                    )
                                    archive.writestr(member, encoded.tobytes())
                                    x1, y1, x2, y2 = boxes[best_index]
                                    row.update(
                                        path=member,
                                        face_found=True,
                                        x1=float(x1), y1=float(y1),
                                        x2=float(x2), y2=float(y2),
                                        prob=float(probabilities[best_index]),
                                    )
                            except (ValueError, cv2.error) as error:
                                print(
                                    f"Crop failed for {job.uid}, "
                                    f"frame {frame_index}: {error}"
                                )
                        rows.append(row)

    manifest = save_split_manifest(rows, output_dir, split_name)
    elapsed = time.time() - started
    found = float(manifest["face_found"].mean()) if len(manifest) else 0.0
    print(
        f"{split_name}: {len(manifest)} frame records; "
        f"face found in {found:.1%}; {elapsed:.0f}s"
    )
    return manifest
