from pathlib import Path
from .config import VIDEO_EXTENSIONS, ORIGINAL_DIR, DATASET_DIR, FAKE_DIR_NAMES


def find_videos(folder):
    
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Video folder does not exist: {folder}")
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def discover_dataset_videos():
    
    real_videos = find_videos(ORIGINAL_DIR)
    fake_videos = []
    for folder_name in FAKE_DIR_NAMES:
        fake_videos.extend(find_videos(DATASET_DIR / folder_name))
    return real_videos, sorted(fake_videos)
