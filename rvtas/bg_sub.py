"""
bg_sub.py — Background subtraction for moving vehicle segmentation.

Uses MOG2 (Mixture of Gaussians v2) to model the static background and
output a foreground mask highlighting moving objects. This mask gates the
optical flow ROI so only genuinely moving pixels are tracked.

Pipeline stage 2: runs after undistortion, before optical flow.
"""

import cv2
import numpy as np


def create_background_subtractor(
    history: int = 500,
    var_threshold: float = 16.0,
    detect_shadows: bool = True,
    method: str = "mog2",
) -> cv2.BackgroundSubtractor:
    """
    Instantiate a background subtractor model.

    Args:
        history: Number of frames used to build the background model.
                 Longer history = more stable background, slower adaptation.
        var_threshold: Mahalanobis distance threshold for foreground/background
                       classification. Lower = more sensitive (more noise too).
        detect_shadows: If True, shadow pixels are marked with value 127.
                        Set False to treat shadows as foreground.
        method: "mog2" (recommended) or "knn".

    Returns:
        Initialised cv2.BackgroundSubtractor instance.
    """
    if method.lower() == "knn":
        return cv2.createBackgroundSubtractorKNN(
            history=history,
            dist2Threshold=var_threshold ** 2,
            detectShadows=detect_shadows,
        )
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )


def apply_background_subtraction(
    subtractor: cv2.BackgroundSubtractor,
    frame: np.ndarray,
    learning_rate: float = -1.0,
) -> np.ndarray:
    """
    Apply the background model to a frame and return the cleaned foreground mask.

    Shadows (value 127) are suppressed. Morphological opening removes small
    noise blobs; dilation fills holes inside vehicles.

    Args:
        subtractor: Background subtractor from create_background_subtractor().
        frame: BGR frame (uint8).
        learning_rate: Model update rate. -1 = automatic. 0 = frozen model.
                       0.01 to 0.05 gives gradual adaptation to lighting changes.

    Returns:
        Binary foreground mask, uint8 with values 0 or 255. Same H, W as frame.
    """
    fg_mask = subtractor.apply(frame, learningRate=learning_rate)

    # Suppress shadow class (value 127 → 0)
    fg_mask[fg_mask == 127] = 0

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    return fg_mask


def mask_to_roi(fg_mask: np.ndarray, min_area: int = 500) -> list:
    """
    Extract bounding boxes of foreground blobs large enough to be vehicles.

    Args:
        fg_mask: Binary foreground mask from apply_background_subtraction().
        min_area: Minimum blob area in pixels to keep. Filters pedestrian-sized
                  and smaller blobs; set lower to also track cyclists.

    Returns:
        List of (x, y, w, h) tuples for each qualifying foreground blob.
    """
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            boxes.append(cv2.boundingRect(cnt))
    return boxes