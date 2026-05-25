"""
flow_tracker.py — Lucas-Kanade sparse optical flow vehicle tracker.

Tracks Shi-Tomasi corner features between consecutive frames. When the
active track count drops below MIN_TRACKS, new corners are detected
to maintain tracking density.

Math:
    LK solves:  [SUM(Ix^2)  SUM(IxIy)] [u]   [-SUM(IxIt)]
                [SUM(IxIy)  SUM(Iy^2) ] [v] = [-SUM(IyIt)]

    using image pyramids (maxLevel=3) to handle large inter-frame motion.
"""

import cv2
import numpy as np


# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
)

# Shi-Tomasi corner detection parameters
CORNER_PARAMS = dict(
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7,
)


def detect_corners(gray: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Detect Shi-Tomasi corners suitable for Lucas-Kanade tracking.

    Args:
        gray: Single-channel grayscale image (uint8).
        mask: Optional uint8 mask — corners are only detected where mask > 0.
              Use this to restrict detection to the road ROI.

    Returns:
        np.ndarray of shape (N, 1, 2), float32, ready for calcOpticalFlowPyrLK.
        Returns empty array if no corners found.
    """
    corners = cv2.goodFeaturesToTrack(gray, mask=mask, **CORNER_PARAMS)
    if corners is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return corners


def track_flow(
    old_gray: np.ndarray,
    new_gray: np.ndarray,
    p0: np.ndarray,
) -> tuple:
    """
    Run Lucas-Kanade optical flow from old_gray to new_gray.

    Only returns points where the forward-backward consistency check
    (status == 1) passes, filtering out lost tracks.

    Args:
        old_gray: Previous frame, single-channel grayscale.
        new_gray: Current frame, single-channel grayscale.
        p0: Tracked points in old_gray, shape (N, 1, 2) float32.

    Returns:
        (good_new, good_old): Two np.ndarrays of shape (M, 2) float32
        where M <= N. good_new[i] is the position in new_gray corresponding
        to good_old[i] in old_gray.
    """
    if p0 is None or len(p0) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    p1, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **LK_PARAMS)

    if p1 is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    good_mask = status.ravel() == 1
    good_new = p1[good_mask].reshape(-1, 2)
    good_old = p0[good_mask].reshape(-1, 2)
    return good_new, good_old


def needs_refresh(points: np.ndarray, min_tracks: int = 80) -> bool:
    """
    Return True if the active track count is below the minimum threshold.

    Args:
        points: Current tracked points array, or None.
        min_tracks: Minimum acceptable number of active tracks.

    Returns:
        True if re-detection should be triggered.
    """
    return points is None or len(points) < min_tracks


def compute_flow_vectors(good_new: np.ndarray, good_old: np.ndarray) -> np.ndarray:
    """
    Compute per-point displacement vectors.

    Args:
        good_new: Current positions, shape (N, 2).
        good_old: Previous positions, shape (N, 2).

    Returns:
        np.ndarray of shape (N, 2) containing (dx, dy) displacement vectors.
    """
    return good_new - good_old


def filter_by_magnitude(
    good_new: np.ndarray,
    good_old: np.ndarray,
    min_mag: float = 1.0,
    max_mag: float = 80.0,
) -> tuple:
    """
    Remove stationary points (likely background) and implausibly large jumps.

    Args:
        good_new: Current positions, shape (N, 2).
        good_old: Previous positions, shape (N, 2).
        min_mag: Minimum pixel displacement to keep (filters static background).
        max_mag: Maximum pixel displacement to keep (filters tracking failures).

    Returns:
        Filtered (good_new, good_old) arrays.
    """
    if len(good_new) == 0:
        return good_new, good_old
    vectors = good_new - good_old
    magnitudes = np.linalg.norm(vectors, axis=1)
    keep = (magnitudes >= min_mag) & (magnitudes <= max_mag)
    return good_new[keep], good_old[keep]
