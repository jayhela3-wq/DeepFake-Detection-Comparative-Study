import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from .config import (
    CROP_SIZE, FACE_MARGIN, MIN_FACE_SIZE, DETECTOR_THRESHOLDS,
    KEEP_ALL_FACES
)


def create_detector(device=None):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(
        keep_all=KEEP_ALL_FACES,
        device=device,
        min_face_size=MIN_FACE_SIZE,
        thresholds=list(DETECTOR_THRESHOLDS),
    )
    return detector


def detect_batch(detector, images):
    
    if not images:
        return [], []
    try:
        boxes, probabilities = detector.detect(np.stack(images))
        return list(boxes), list(probabilities)
    except RuntimeError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        all_boxes, all_probabilities = [], []
        for image in images:
            boxes, probabilities = detector.detect(image)
            all_boxes.append(boxes)
            all_probabilities.append(probabilities)
        return all_boxes, all_probabilities


def crop_face(image_rgb, box, margin=FACE_MARGIN, size=CROP_SIZE):
   
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    pad = int(side)
    padded = cv2.copyMakeBorder(
        image_rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE
    )
    left = int(center_x - side / 2) + pad
    top = int(center_y - side / 2) + pad
    crop = padded[top:top + int(side), left:left + int(side)]
    return cv2.resize(crop, (size, size))
