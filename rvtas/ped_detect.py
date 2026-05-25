"""
ped_detect.py — HOG + SVM pedestrian detection with Non-Maximum Suppression.

Uses OpenCV's built-in HOG descriptor with the default pre-trained people
detector (Dalal & Triggs 2005). No external model files required.

Theory:
    HOG divides a 64x128 detection window into cells, builds gradient
    direction histograms (9 bins, 0-180°), groups cells into overlapping
    blocks for L2-Hys normalisation, producing a 3780-D descriptor.
    A linear SVM classifies the descriptor: score = w^T * x + b
    Positive score → pedestrian.
"""

import cv2
import numpy as np


def _build_hog(win_stride=(8, 8), padding=(4, 4), scale=1.05) -> cv2.HOGDescriptor:
    """Initialise HOG descriptor with the default people detector SVM."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


# Module-level singleton — avoids re-initialising per frame
_HOG = _build_hog()


def detect_pedestrians(
    frame: np.ndarray,
    win_stride: tuple = (8, 8),
    padding: tuple = (4, 4),
    scale: float = 1.05,
    nms_thresh: float = 0.4,
) -> tuple:
    """
    Detect pedestrians in frame using HOG + default linear SVM.

    Args:
        frame: BGR image (uint8).
        win_stride: HOG window stride in pixels (smaller = more detections, slower).
        padding: Padding around each detection window.
        scale: Pyramid scale factor (smaller = finer scale, slower).
        nms_thresh: IoU threshold for Non-Maximum Suppression.

    Returns:
        (boxes, weights) where:
            boxes: list of (x, y, w, h) tuples for each detected pedestrian.
            weights: list of SVM confidence scores corresponding to each box.
    """
    rects, weights = _HOG.detectMultiScale(
        frame,
        winStride=win_stride,
        padding=padding,
        scale=scale,
    )

    if len(rects) == 0:
        return [], []

    # Non-Maximum Suppression
    indices = non_max_suppression(rects, weights, nms_thresh)
    filtered_boxes = [tuple(rects[i]) for i in indices]
    filtered_weights = [float(weights[i]) for i in indices]
    return filtered_boxes, filtered_weights


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.4,
) -> list:
    """
    Greedy Non-Maximum Suppression (NMS) to remove overlapping detections.

    Args:
        boxes: np.ndarray of shape (N, 4) with columns [x, y, w, h].
        scores: np.ndarray of shape (N,) with detection confidence scores.
        iou_threshold: Suppress boxes whose IoU with a higher-scoring box
                       exceeds this threshold.

    Returns:
        List of indices of surviving boxes in descending score order.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = (boxes[:, 0] + boxes[:, 2]).astype(float)
    y2 = (boxes[:, 1] + boxes[:, 3]).astype(float)
    areas = (x2 - x1) * (y2 - y1)

    order = np.argsort(scores.ravel())[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break

        rest = order[1:]
        inter_x1 = np.maximum(x1[i], x1[rest])
        inter_y1 = np.maximum(y1[i], y1[rest])
        inter_x2 = np.minimum(x2[i], x2[rest])
        inter_y2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        union = areas[i] + areas[rest] - intersection
        iou = intersection / np.where(union > 0, union, 1e-6)

        order = rest[iou <= iou_threshold]

    return keep
