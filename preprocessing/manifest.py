import os
from pathlib import Path
import pandas as pd
from preprocessing.shared_config import MANIFEST_PATH, CLASS_NAMES

LABELS = {name: i for i, name in enumerate(CLASS_NAMES)}

def scan_split(folder, split):
    rows = []

    for root, dirs, files in os.walk(folder):
        class_name = Path(root).parent.name
        
        if class_name not in LABELS:
            continue

        for f in files:
            if f.endswith(".jpg"):
                rows.append((os.path.join(root, f), LABELS[class_name], split))


    return rows


