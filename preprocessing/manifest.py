from pathlib import Path
import pandas as pd

MANIFEST_COLUMNS = [
    "path", "label", "split", "video_id", "frame_idx",
    "face_found", "x1", "y1", "x2", "y2", "prob",
]


def save_split_manifest(rows, output_dir, split_name):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    frame.to_csv(output_dir / f"manifest_{split_name}.csv", index=False)
    return frame


def save_combined_manifest(manifests, output_dir):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = (
        pd.concat(manifests, ignore_index=True)
        if manifests else pd.DataFrame(columns=MANIFEST_COLUMNS)
    )
    combined.to_csv(output_dir / "manifest.csv", index=False)
    return combined


def load_manifest(split=None, manifest_path=None):
    
    if manifest_path is None:
        raise ValueError("Provide manifest_path explicitly")
    frame = pd.read_csv(manifest_path)
    if split is not None:
        frame = frame[frame["split"] == split]
    return frame


def valid_face_rows(frame):
    
    return frame[frame["face_found"].astype(bool) & frame["path"].notna()
                 & (frame["path"].astype(str) != "")]
