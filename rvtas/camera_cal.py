"""
camera_cal.py — Camera calibration and frame undistortion.

Phase 1 of the RVTAS pipeline. Run calibrate() once to generate
calibration/camera_params.npz, then call undistort_frame() per frame.
"""

import cv2
import numpy as np
import glob
import os


CHESSBOARD_SIZE = (9, 6)   # inner corners (cols, rows) — adjust for your board


def calibrate(image_dir: str, board_size: tuple = CHESSBOARD_SIZE) -> tuple:
    """
    Find chessboard corners in all images under image_dir and compute
    the camera intrinsic matrix K and distortion coefficients.

    Args:
        image_dir: Path to folder containing chessboard calibration images.
        board_size: Number of inner corners (cols, rows) on the chessboard.

    Returns:
        (K, dist, reprojection_error) where K is 3x3 float64 and dist is
        the distortion coefficient vector.

    Raises:
        ValueError: If fewer than 5 valid chessboard images are found.
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    obj_points = []   # 3-D points in real world space
    img_points = []   # 2-D points in image plane
    img_size = None

    pattern = os.path.join(image_dir, "*.jpg")
    paths = glob.glob(pattern) + glob.glob(os.path.join(image_dir, "*.png"))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)

    if len(obj_points) < 5:
        raise ValueError(
            f"Only {len(obj_points)} valid calibration images found in {image_dir}. "
            "Need at least 5. Check CHESSBOARD_SIZE matches your printed board."
        )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    # Compute mean reprojection error
    total_error = 0.0
    for i, (objp_i, imgp_i) in enumerate(zip(obj_points, img_points)):
        projected, _ = cv2.projectPoints(objp_i, rvecs[i], tvecs[i], K, dist)
        total_error += cv2.norm(imgp_i, projected, cv2.NORM_L2) / len(projected)
    reprojection_error = total_error / len(obj_points)

    print(f"[calibrate] Used {len(obj_points)} images. "
          f"Reprojection error: {reprojection_error:.4f} px")

    output_path = os.path.join(os.path.dirname(image_dir), "calibration", "camera_params.npz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, K=K, dist=dist)
    print(f"[calibrate] Saved to {output_path}")

    return K, dist, reprojection_error


def undistort_frame(frame: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Return an undistorted copy of frame using pre-computed intrinsics.

    Uses getOptimalNewCameraMatrix with alpha=0 to crop out black borders
    while preserving all valid image pixels.

    Args:
        frame: BGR image as np.ndarray (H, W, 3).
        K: 3x3 camera intrinsic matrix.
        dist: Distortion coefficient vector (4, 5, or 8 elements).

    Returns:
        Undistorted BGR image, same shape as input.
    """
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    undistorted = cv2.undistort(frame, K, dist, None, new_K)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]
        undistorted = cv2.resize(undistorted, (w, h))
    return undistorted


def load_params(npz_path: str) -> tuple:
    """
    Load camera intrinsics saved by calibrate().

    Args:
        npz_path: Path to .npz file produced by calibrate().

    Returns:
        (K, dist) as numpy arrays.
    """
    data = np.load(npz_path)
    return data["K"], data["dist"]


if __name__ == "__main__":
    import sys
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "calibration/images"
    calibrate(image_dir)
