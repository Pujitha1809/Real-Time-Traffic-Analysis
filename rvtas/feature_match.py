

import cv2
import numpy as np


# ORB parameters — tune nfeatures for speed vs density trade-off
_ORB = cv2.ORB_create(nfeatures=500)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def extract_orb_features(gray: np.ndarray) -> tuple:
    """
    Detect ORB keypoints and compute binary descriptors.

    Args:
        gray: Single-channel grayscale image (uint8).

    Returns:
        (keypoints, descriptors):
            keypoints: list of cv2.KeyPoint objects.
            descriptors: np.ndarray of shape (N, 32), dtype uint8, or None
                         if no keypoints were found.
    """
    keypoints, descriptors = _ORB.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(
    desc_ref: np.ndarray,
    desc_cur: np.ndarray,
    ratio_thresh: float = 0.75,
) -> list:
    """
    Match ORB descriptors between a reference and current frame using
    Brute-Force Hamming distance matching with cross-check.

    Args:
        desc_ref: Reference frame descriptors, shape (N, 32) uint8.
        desc_cur: Current frame descriptors, shape (M, 32) uint8.
        ratio_thresh: Not used with cross-check; retained for API consistency.
                      For Lowe ratio test, disable crossCheck in BFMatcher.

    Returns:
        Sorted list of cv2.DMatch objects (best matches first).
        Returns empty list if either descriptor set is None.
    """
    if desc_ref is None or desc_cur is None:
        return []
    matches = _BF.match(desc_ref, desc_cur)
    return sorted(matches, key=lambda m: m.distance)


def matched_point_pairs(
    kp_ref: list,
    kp_cur: list,
    matches: list,
    top_n: int = 50,
) -> tuple:
    """
    Extract matched pixel coordinates from keypoints and matches.

    Args:
        kp_ref: Keypoints from the reference frame.
        kp_cur: Keypoints from the current frame.
        matches: List of cv2.DMatch from match_features().
        top_n: Keep only the top N best matches.

    Returns:
        (pts_ref, pts_cur): Two np.ndarrays of shape (K, 2) float32
        where K = min(top_n, len(matches)).
    """
    best = matches[:top_n]
    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in best])
    pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in best])
    return pts_ref, pts_cur


def estimate_homography_from_matches(
    kp_ref: list,
    kp_cur: list,
    matches: list,
    ransac_thresh: float = 5.0,
) -> np.ndarray:
    """
    Estimate a homography between two frames using RANSAC on matched keypoints.

    Useful for detecting/correcting camera shake or verifying the static
    camera assumption underlying the optical flow speed estimator.

    Args:
        kp_ref: Reference frame keypoints.
        kp_cur: Current frame keypoints.
        matches: DMatch list from match_features().
        ransac_thresh: RANSAC reprojection error threshold in pixels.

    Returns:
        3x3 homography matrix (float64), or None if estimation fails.
    """
    if len(matches) < 4:
        return None
    pts_ref, pts_cur = matched_point_pairs(kp_ref, kp_cur, matches)
    H, mask = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, ransac_thresh)
    return H
