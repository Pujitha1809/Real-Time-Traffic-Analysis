"""
utils.py — Shared helpers for the RVTAS pipeline.
"""

import time
import json
import os
import cv2
import numpy as np


class FPSCounter:
    """Rolling-average FPS counter."""

    def __init__(self, window: int = 30):
        self._times = []
        self._window = window
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Call once per processed frame. Returns current smoothed FPS."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) > self._window:
            self._times.pop(0)
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 0.0


def write_json_log(log: dict, output_path: str) -> None:
    """
    Serialise the run log to a JSON file.

    Args:
        log: Dictionary of run statistics.
        output_path: Destination .json file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(log, f, indent=2)


def clamp(value: float, lo: float, hi: float) -> float:
    """Return value clamped to [lo, hi]."""
    return max(lo, min(hi, value))


def draw_roi_polygon(frame: np.ndarray, polygon: np.ndarray, colour=(0, 255, 200)) -> np.ndarray:
    """
    Draw a semi-transparent ROI polygon overlay on frame.

    Args:
        frame: BGR image (modified in place).
        polygon: np.ndarray of shape (N, 2) int32 vertices.
        colour: Fill/border colour in BGR.

    Returns:
        Annotated frame.
    """
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], colour)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [polygon], True, colour, 2)
    return frame


def parse_yaml_pts(pts_list: list) -> np.ndarray:
    """
    Convert a list-of-lists from YAML into a float32 numpy array.

    Args:
        pts_list: e.g. [[312, 480], [620, 480], ...]

    Returns:
        np.ndarray of shape (N, 2) float32.
    """
    return np.array(pts_list, dtype=np.float32)
