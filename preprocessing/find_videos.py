from pathlib import Path

VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
]

def find_videos(folder):
    videos = []

    for path in Path(folder).rglob("*"):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    return videos