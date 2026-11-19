from pathlib import Path
from urllib.request import urlretrieve

from .config import SPLITS_DIR

OFFICIAL_SPLITS_BASE_URL = (
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/"
)


def download_official_splits(folder=SPLITS_DIR, overwrite=False):
    """Download official train/val/test JSON files when absent."""
    import json

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        destination = folder / f"{split_name}.json"
        if overwrite or not destination.exists():
            urlretrieve(
                OFFICIAL_SPLITS_BASE_URL + f"{split_name}.json",
                destination
            )
        with destination.open("r", encoding="utf-8") as file:
            pairs = json.load(file)
        print(split_name, len(pairs), "pairs")
    return folder
