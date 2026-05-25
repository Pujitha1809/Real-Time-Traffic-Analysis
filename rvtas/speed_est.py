"""
speed_est.py — Vehicle speed estimation using ground-plane homography.

Converts per-frame pixel displacement to real-world speed (km/h) by mapping
tracked pixel positions through the calibrated homography H into metres,
computing Euclidean distance, then scaling by the frame rate.

Formula:
    d_metres = || H*p_(k+1) - H*p_k ||_2
    speed_kmh = d_metres * fps * 3.6
"""

import numpy as np
from homography import pixel_to_ground


def estimate_speed(
    H: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    fps: float,
) -> float:
    """
    Estimate vehicle speed in km/h from one frame of pixel displacement.

    Args:
        H: 3x3 homography matrix (pixels → metres) from build_homography().
        p_old: Pixel position at frame k,   shape (2,) as [u, v].
        p_new: Pixel position at frame k+1, shape (2,) as [u, v].
        fps: Video frame rate in frames per second.

    Returns:
        Speed in km/h (float). Returns 0.0 if the result is negative or NaN.
    """
    g_old = pixel_to_ground(H, p_old)
    g_new = pixel_to_ground(H, p_new)
    dist_m = float(np.linalg.norm(g_new - g_old))
    speed = dist_m * fps * 3.6   # m/frame → km/h
    return max(0.0, speed) if np.isfinite(speed) else 0.0


def estimate_speeds_batch(
    H: np.ndarray,
    pts_old: np.ndarray,
    pts_new: np.ndarray,
    fps: float,
    min_kmh: float = 2.0,
    max_kmh: float = 120.0,
) -> np.ndarray:
    """
    Vectorised speed estimation for multiple tracked points simultaneously.

    Args:
        H: 3x3 homography matrix (pixels → metres).
        pts_old: Previous positions, shape (N, 2).
        pts_new: Current positions, shape (N, 2).
        fps: Frame rate in fps.
        min_kmh: Discard readings below this (filters stationary noise).
        max_kmh: Discard readings above this (filters tracking outliers).

    Returns:
        np.ndarray of valid speed readings (may be shorter than N).
    """
    if len(pts_old) == 0:
        return np.array([], dtype=np.float32)

    # Homogeneous pixel coords: shape (N, 3)
    ones = np.ones((len(pts_old), 1), dtype=np.float64)
    ph_old = np.hstack([pts_old, ones])
    ph_new = np.hstack([pts_new, ones])

    # Apply H: (3x3) @ (3,N) → (3,N), then divide by last row
    g_old = (H @ ph_old.T).T
    g_old = g_old[:, :2] / g_old[:, 2:3]

    g_new = (H @ ph_new.T).T
    g_new = g_new[:, :2] / g_new[:, 2:3]

    dists = np.linalg.norm(g_new - g_old, axis=1)
    speeds = dists * fps * 3.6

    valid = (speeds >= min_kmh) & (speeds <= max_kmh) & np.isfinite(speeds)
    return speeds[valid].astype(np.float32)


def rolling_average_speed(speed_buffer: list, window: int = 10) -> float:
    """
    Return the rolling mean of the last `window` speed readings.

    Args:
        speed_buffer: List of recent per-frame average speeds (km/h).
        window: Number of frames to average over.

    Returns:
        Mean speed (float), or 0.0 if the buffer is empty.
    """
    if not speed_buffer:
        return 0.0
    recent = speed_buffer[-window:]
    return float(np.mean(recent))
