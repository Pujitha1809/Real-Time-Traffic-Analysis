"""
homography.py — Homography-based perspective transform (bird's-eye warp).

Calibrates a mapping from image pixel coordinates to real-world ground-plane
metres using 4 known correspondence points (e.g. lane markings of known size).

Math:
    p_ground = H * p_pixel  (homogeneous coords, then divide by w)
    Calibrated so that distances in the ground plane are in metres.
"""

import cv2
import numpy as np


def build_homography(src_pts: list, dst_pts_metres: list) -> tuple:
    """
    Compute the homography matrix H mapping image pixels to ground metres.

    Args:
        src_pts: List of 4 [u, v] pixel coordinates in the source image.
                 These should correspond to clearly identifiable road features
                 (lane markings, painted lines) whose real-world positions
                 you know (e.g. standard 3.5 m lane width, 10 m spacing).
        dst_pts_metres: List of 4 [x, y] real-world coordinates in metres,
                        corresponding to src_pts.

    Returns:
        (H, scale_mpp) where:
            H: 3x3 homography matrix (float64).
            scale_mpp: Approximate metres-per-pixel at the reference plane
                       (informational — H already encodes the full transform).

    Raises:
        ValueError: If fewer than 4 point pairs are provided.
    """
    if len(src_pts) < 4 or len(dst_pts_metres) < 4:
        raise ValueError("build_homography requires at least 4 point correspondences.")

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts_metres, dtype=np.float32)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("findHomography failed — check src_pts / dst_pts_metres are correct.")

    # Estimate mean scale (metres per pixel) from horizontal span
    pixel_span = np.linalg.norm(src[1] - src[0])
    metre_span = np.linalg.norm(dst[1] - dst[0])
    scale_mpp = metre_span / pixel_span if pixel_span > 0 else 0.01

    return H, scale_mpp


def pixel_to_ground(H: np.ndarray, pixel_pt: np.ndarray) -> np.ndarray:
    """
    Map a single image pixel coordinate to a 2-D ground-plane position in metres.

    Args:
        H: 3x3 homography matrix from build_homography().
        pixel_pt: [u, v] pixel coordinate, shape (2,) or (1, 2).

    Returns:
        np.ndarray of shape (2,): [x_metres, y_metres] in ground plane.
    """
    pt = np.array([pixel_pt[0], pixel_pt[1], 1.0], dtype=np.float64)
    gp = H @ pt
    return gp[:2] / gp[2]


def warp_frame(frame: np.ndarray, H: np.ndarray, output_size: tuple = (400, 400)) -> np.ndarray:
    """
    Warp a full video frame to a bird's-eye view using H.

    Note: H maps pixels → metres. For a display bird's-eye view we need the
    inverse perspective transform (image → top-down image), computed as
    cv2.getPerspectiveTransform using the src/dst points directly.

    This function wraps warpPerspective for visualisation purposes.

    Args:
        frame: BGR source frame.
        H: 3x3 homography from build_homography() (pixels → metres).
        output_size: (width, height) of the output bird's-eye image in pixels.

    Returns:
        Warped BGR image of shape (output_size[1], output_size[0], 3).
    """
    return cv2.warpPerspective(frame, H, output_size)


def build_display_homography(src_pts: list, dst_pts_pixels: list) -> np.ndarray:
    """
    Compute a homography for display-only bird's-eye view warp (pixel → pixel).

    Use this to render a top-down preview overlay. For speed computation,
    use build_homography() with real-world metre coordinates.

    Args:
        src_pts: 4 source pixel points (same as for build_homography).
        dst_pts_pixels: 4 destination pixel points in the output top-down image.

    Returns:
        H_display: 3x3 homography matrix (float64), pixel → pixel.
    """
    src = np.float32(src_pts)
    dst = np.float32(dst_pts_pixels)
    return cv2.getPerspectiveTransform(src, dst)