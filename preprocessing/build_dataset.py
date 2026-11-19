import argparse
import pandas as pd

from .config import (
    OUTPUT_DIR, SPLIT_ASSIGNMENT_PATH, SPLIT_NAMES, SPLITS_DIR
)
from .identity_split import load_official_splits
from .download_splits import download_official_splits
from .split_assignment import make_split_assignment
from .face_detection import create_detector
from .extract_face_crops import process_split
from .manifest import save_combined_manifest


def run_pipeline(rebuild_assignments=True, limit=None):
    
    download_official_splits(SPLITS_DIR)

    if rebuild_assignments:
        assignments = make_split_assignment(SPLIT_ASSIGNMENT_PATH)
    else:
        assignments = pd.read_csv(SPLIT_ASSIGNMENT_PATH)

    
    load_official_splits(SPLITS_DIR)

    detector = create_detector()
    manifests = [
        process_split(
            split_name, assignments, detector,
            output_dir=OUTPUT_DIR, limit=limit
        )
        for split_name in ("val", "test", "train")
    ]
    combined = save_combined_manifest(manifests, OUTPUT_DIR)

    print("\nPipeline complete.")
    print(f"Combined manifest: {OUTPUT_DIR / 'manifest.csv'}")
    if not combined.empty:
        print(combined.groupby(["split", "label", "face_found"]).size())
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Build identity-aware MTCNN face-crop dataset."
    )
    parser.add_argument(
        "--reuse-split-assignment", action="store_true",
        help="Reuse an existing split_assignment.csv instead of rebuilding it."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional number of videos per split for a small test run."
    )
    args = parser.parse_args()
    run_pipeline(
        rebuild_assignments=not args.reuse_split_assignment,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
