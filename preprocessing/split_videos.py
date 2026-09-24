import random

def split_videos(videos, train_ratio=0.70, val_ratio=0.15, test_ratio = 0.15, seed=42):
    videos = list(videos)

    random.Random(seed).shuffle(videos)

    total = len(videos)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_videos = videos[:train_end]
    val_videos = videos[train_end:val_end]
    test_videos = videos[val_end:]

    return train_videos, val_videos, test_videos