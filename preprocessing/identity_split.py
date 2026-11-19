import random
import re
from pathlib import Path

NUMERIC_ID = re.compile(r"^\d{3}$")
SPLIT_NAMES = ("train", "val", "test")


def load_official_splits(folder):
    """Load official FF++ JSON pair lists into {video_id: split_name}."""
    import json

    folder = Path(folder)
    id_to_split = {}

    for split_name in SPLIT_NAMES:
        split_file = folder / f"{split_name}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing official split file: {split_file}")

        pairs = json.loads(split_file.read_text())
        for pair in pairs:
            for video_id in pair:
                video_id = str(video_id)
                previous = id_to_split.setdefault(video_id, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"Video identity {video_id} appears in both "
                        f"{previous} and {split_name}"
                    )
    return id_to_split


def video_ids(path):
    """Extract numeric FF++ IDs from stems such as 200 or 200_189."""
    parts = Path(path).stem.split("_")
    if parts and all(NUMERIC_ID.fullmatch(part) for part in parts):
        return parts
    return []


def _group_key(path):
    """Keep non-numeric videos from the same actor pair together."""
    return Path(path).stem.split("__")[0]


def split_by_identity(real_videos, fake_videos, id_to_split, seed=42,
                      ratios=(0.70, 0.15, 0.15)):
    """Assign videos without splitting known source identities across splits."""
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must contain train/val/test values summing to 1")

    output = {name: {"real": [], "fake": []} for name in SPLIT_NAMES}
    unmatched, cross_or_unknown = [], []

    for label, videos in (("real", real_videos), ("fake", fake_videos)):
        for video in videos:
            ids = video_ids(video)
            if not ids:
                unmatched.append((label, video))
                continue

            assigned = {id_to_split.get(video_id) for video_id in ids}
            if len(assigned) != 1 or None in assigned:
                cross_or_unknown.append((label, video))
                continue

            output[assigned.pop()][label].append(video)

    # Fallback for names that do not use the numeric FF++ naming convention.
    groups = {}
    for label, video in unmatched:
        groups.setdefault(_group_key(video), []).append((label, video))

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_groups = len(keys)
    train_end = int(n_groups * ratios[0])
    val_end = int(n_groups * (ratios[0] + ratios[1]))

    for index, key in enumerate(keys):
        split_name = (
            "train" if index < train_end
            else "val" if index < val_end
            else "test"
        )
        for label, video in groups[key]:
            output[split_name][label].append(video)

    assigned_count = sum(
        len(group["real"]) + len(group["fake"])
        for group in output.values()
    )
    print(f"Assigned videos: {assigned_count}")
    print(f"Non-numeric videos grouped: {len(unmatched)} in {len(groups)} groups")
    print(f"Dropped videos with cross-split/unknown IDs: {len(cross_or_unknown)}")
    return output


def leakage_report(splits):
    """Report fake videos whose known source IDs map to another split."""
    identity_location = {}
    for split_name, group in splits.items():
        for video in group["real"]:
            ids = video_ids(video)
            if ids:
                identity_location[ids[0]] = split_name

    print("\nLeakage check:")
    for split_name, group in splits.items():
        checked = leaked = 0
        for video in group["fake"]:
            ids = video_ids(video)
            if not ids:
                continue
            checked += 1
            if any(identity_location.get(video_id, split_name) != split_name
                   for video_id in ids):
                leaked += 1
        percentage = 100 * leaked / checked if checked else 0.0
        print(f"  {split_name}: {leaked}/{checked} ({percentage:.1f}%)")
