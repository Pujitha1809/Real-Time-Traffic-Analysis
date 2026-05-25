"""
preprocess.py — DFT-based low-pass spatial frequency filter.

Suppresses high-frequency sensor noise (common in night-time CCTV) without
the spatial-domain blurring artefacts introduced by a Gaussian kernel.
This preserves fine vehicle edges needed by the optical flow tracker.

Theory:
    DFT converts the image to frequency domain. A circular mask zeroes
    out all frequencies beyond radius R from the DC component (centre of
    the shifted spectrum). IDFT then reconstructs the filtered image.
"""

import cv2
import numpy as np


def dft_lowpass(frame: np.ndarray, radius: int = 45) -> np.ndarray:
    """
    Apply a circular low-pass filter in the DFT domain.

    H_lp(u,v) = 1  if sqrt(u^2 + v^2) <= radius
                0  otherwise

    Args:
        frame: Grayscale image (uint8, single channel).
        radius: Cut-off frequency radius in pixels. Larger = less filtering.
                Typical range: 30 (aggressive) to 80 (mild).

    Returns:
        Filtered grayscale image as uint8, same shape as input.
    """
    assert len(frame.shape) == 2, "dft_lowpass expects a single-channel (grayscale) image"

    rows, cols = frame.shape
    # Zero-pad to optimal DFT size for speed
    opt_rows = cv2.getOptimalDFTSize(rows)
    opt_cols = cv2.getOptimalDFTSize(cols)
    padded = np.zeros((opt_rows, opt_cols), np.float32)
    padded[:rows, :cols] = frame.astype(np.float32)

    # Forward DFT → shift DC to centre
    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft, axes=(0, 1))

    # Build circular low-pass mask
    cy, cx = opt_rows // 2, opt_cols // 2
    Y, X = np.ogrid[:opt_rows, :opt_cols]
    mask = ((X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2).astype(np.float32)
    mask_2ch = np.stack([mask, mask], axis=-1)   # apply to both real + imaginary

    # Multiply in frequency domain (convolution theorem)
    filtered_shifted = dft_shifted * mask_2ch

    # Inverse shift → IDFT → magnitude
    filtered = np.fft.ifftshift(filtered_shifted, axes=(0, 1))
    img_back = cv2.idft(filtered)
    img_magnitude = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Crop back to original size and normalise to uint8
    img_magnitude = img_magnitude[:rows, :cols]
    cv2.normalize(img_magnitude, img_magnitude, 0, 255, cv2.NORM_MINMAX)
    return img_magnitude.astype(np.uint8)


def apply_roi_mask(frame: np.ndarray, roi_polygon: np.ndarray) -> np.ndarray:
    """
    Zero out pixels outside a polygonal region of interest.

    Useful for restricting optical flow to the road surface only,
    ignoring sky, buildings, and pedestrian footpaths.

    Args:
        frame: Grayscale or BGR image.
        roi_polygon: Numpy array of shape (N, 2) with polygon vertices in
                     pixel coordinates, dtype int32.

    Returns:
        Image with everything outside roi_polygon set to zero.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)
