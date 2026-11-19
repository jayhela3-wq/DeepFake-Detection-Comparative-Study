import pandas as pd
from pathlib import Path

from .config import (
    DATASET_DIR, SPLITS_DIR, SPLIT_ASSIGNMENT_PATH, SEED, SPLIT_RATIOS
)
from .find_videos import discover_dataset_videos
from .identity_split import load_official_splits, split_by_identity, leakage_report


def make_split_assignment(output_path=SPLIT_ASSIGNMENT_PATH):
    
    real_videos, fake_videos = discover_dataset_videos()
    print(f"Real videos: {len(real_videos)} | Fake videos: {len(fake_videos)}")

    id_to_split = load_official_splits(SPLITS_DIR)
    splits = split_by_identity(
        real_videos, fake_videos, id_to_split,
        seed=SEED, ratios=SPLIT_RATIOS
    )
    leakage_report(splits)

    rows = []
    for split_name, groups in splits.items():
        for label in ("real", "fake"):
            for video in groups[label]:
                uid = "_".join(
                    Path(video).relative_to(DATASET_DIR).with_suffix("").parts
                )
                rows.append((str(video), uid, label, split_name))

    frame = pd.DataFrame(
        rows, columns=["video_path", "uid", "label", "split"]
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    print(f"Saved {len(frame)} video assignments to {output_path}")
    if not frame.empty:
        print(frame.groupby(["split", "label"]).size())
    return frame
