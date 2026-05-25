"""
visualiser.py — Frame annotation, dashboard overlay, and heatmap management.

Handles all drawing operations: flow vectors, speed labels, pedestrian boxes,
lane count overlays, FPS counter, and the per-minute heatmap PNG export.

Pipeline stage 9: final step before writing the annotated frame to video.
"""

import cv2
import numpy as np
import os
import time


# Colours (BGR)
COLOUR_FLOW = (0, 200, 100)
COLOUR_SPEED = (255, 255, 255)
COLOUR_PED = (0, 60, 255)
COLOUR_DASHBOARD = (20, 20, 20)
COLOUR_WARN = (0, 80, 255)


def draw_flow_vectors(
    frame: np.ndarray,
    good_new: np.ndarray,
    good_old: np.ndarray,
    colour: tuple = COLOUR_FLOW,
) -> np.ndarray:
    """
    Draw optical flow displacement arrows on frame.

    Args:
        frame: BGR image to annotate (modified in place).
        good_new: Current positions, shape (N, 2).
        good_old: Previous positions, shape (N, 2).
        colour: Arrow colour in BGR.

    Returns:
        Annotated frame (same object as input).
    """
    for new_pt, old_pt in zip(good_new, good_old):
        a, b = int(new_pt[0]), int(new_pt[1])
        c, d = int(old_pt[0]), int(old_pt[1])
        cv2.arrowedLine(frame, (c, d), (a, b), colour, 1, tipLength=0.4)
        cv2.circle(frame, (a, b), 2, colour, -1)
    return frame


def draw_speed_labels(
    frame: np.ndarray,
    good_new: np.ndarray,
    speeds: list,
    colour: tuple = COLOUR_SPEED,
) -> np.ndarray:
    """
    Overlay speed estimates next to each tracked point.

    Args:
        frame: BGR image to annotate (modified in place).
        good_new: Current tracked positions, shape (N, 2).
        speeds: List of speed values (km/h) corresponding to good_new.
        colour: Text colour in BGR.

    Returns:
        Annotated frame.
    """
    for pt, spd in zip(good_new[:len(speeds)], speeds):
        x, y = int(pt[0]), int(pt[1])
        label = f"{spd:.0f}"
        cv2.putText(frame, label, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
    return frame


def draw_pedestrian_boxes(
    frame: np.ndarray,
    boxes: list,
    weights: list = None,
    colour: tuple = COLOUR_PED,
) -> np.ndarray:
    """
    Draw bounding boxes around detected pedestrians.

    Args:
        frame: BGR image to annotate (modified in place).
        boxes: List of (x, y, w, h) tuples from detect_pedestrians().
        weights: Optional list of confidence scores for labelling.
        colour: Box colour in BGR.

    Returns:
        Annotated frame.
    """
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        label = "PED"
        if weights is not None and i < len(weights):
            label = f"PED {weights[i]:.2f}"
        cv2.putText(frame, label, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return frame


def draw_dashboard(
    frame: np.ndarray,
    fps: float,
    vehicle_count: int,
    avg_speed_kmh: float,
    ped_count: int,
    frame_number: int = 0,
) -> np.ndarray:
    """
    Draw a semi-transparent HUD dashboard in the top-left corner.

    Shows: FPS counter, cumulative vehicle count, rolling average speed,
    pedestrian event count, and elapsed frame number.

    Args:
        frame: BGR image to annotate (modified in place).
        fps: Current processing frame rate.
        vehicle_count: Total vehicles counted since start.
        avg_speed_kmh: Rolling average speed over recent frames.
        ped_count: Total pedestrian detections since start.
        frame_number: Current frame index.

    Returns:
        Annotated frame.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (250, 120), COLOUR_DASHBOARD, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        f"FPS:      {fps:.1f}",
        f"Vehicles: {vehicle_count}",
        f"Avg Speed:{avg_speed_kmh:.1f} km/h",
        f"Peds:     {ped_count}",
        f"Frame:    {frame_number}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (16, 30 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def update_heatmap(
    heatmap: np.ndarray,
    good_new: np.ndarray,
    good_old: np.ndarray,
    decay: float = 0.995,
) -> np.ndarray:
    """
    Accumulate motion intensity onto a floating-point heatmap.

    Each tracked point contributes a Gaussian splat at its current position,
    weighted by its displacement magnitude. A slow decay keeps the heatmap
    from saturating on long-duration runs.

    Args:
        heatmap: Accumulated heatmap (float32, same H, W as video frame).
                 Updated in place.
        good_new: Current positions, shape (N, 2).
        good_old: Previous positions, shape (N, 2).
        decay: Per-frame decay multiplier (0.995 = gentle fade over ~200 frames).

    Returns:
        Updated heatmap (same object).
    """
    heatmap *= decay
    for new_pt, old_pt in zip(good_new, good_old):
        x, y = int(new_pt[0]), int(new_pt[1])
        if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
            mag = float(np.linalg.norm(new_pt - old_pt))
            heatmap[y, x] += mag
    return heatmap


def save_heatmap(
    heatmap: np.ndarray,
    output_path: str,
    blur_radius: int = 21,
) -> str:
    """
    Normalise, apply Gaussian blur, colour-map, and save the heatmap as PNG.

    Args:
        heatmap: Accumulated floating-point heatmap.
        output_path: Destination file path (should end in .png).
        blur_radius: Gaussian blur kernel size for smoothing the heatmap.
                     Must be odd. Larger = smoother, less localised.

    Returns:
        output_path (for logging convenience).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    normalised = np.zeros_like(heatmap, dtype=np.uint8)
    cv2.normalize(heatmap, normalised, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blurred = cv2.GaussianBlur(normalised, (blur_radius, blur_radius), 0)
    coloured = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, coloured)
    return output_path


def draw_overlay(
    frame: np.ndarray,
    good_new: np.ndarray,
    good_old: np.ndarray,
    speeds: list,
    ped_boxes: list,
    ped_weights: list = None,
) -> np.ndarray:
    """
    Composite all visual annotations onto the frame.

    Convenience wrapper that calls draw_flow_vectors, draw_speed_labels,
    and draw_pedestrian_boxes in the correct layering order.

    Args:
        frame: BGR image (modified in place).
        good_new: Current tracked positions.
        good_old: Previous tracked positions.
        speeds: Speed values corresponding to tracked positions.
        ped_boxes: Pedestrian bounding boxes from detect_pedestrians().
        ped_weights: Optional pedestrian confidence scores.

    Returns:
        Fully annotated frame.
    """
    draw_flow_vectors(frame, good_new, good_old)
    draw_speed_labels(frame, good_new, speeds)
    draw_pedestrian_boxes(frame, ped_boxes, ped_weights)
    return frame
