"""
tests/test_speed_est.py — Unit tests for speed estimation and homography modules.

Run with:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from homography import build_homography, pixel_to_ground
from speed_est import estimate_speed, estimate_speeds_batch, rolling_average_speed


# ── Homography fixtures ────────────────────────────────────────────────────────

SRC_PTS = [[100, 200], [200, 200], [200, 300], [100, 300]]
DST_PTS = [[0.0, 0.0], [3.5, 0.0], [3.5, 10.0], [0.0, 10.0]]


@pytest.fixture
def homography():
    H, scale = build_homography(SRC_PTS, DST_PTS)
    return H, scale


# ── build_homography tests ─────────────────────────────────────────────────────

class TestBuildHomography:
    def test_returns_3x3_matrix(self, homography):
        H, _ = homography
        assert H.shape == (3, 3), "Homography matrix must be 3x3"

    def test_scale_is_positive(self, homography):
        _, scale = homography
        assert scale > 0, "Scale (metres per pixel) must be positive"

    def test_raises_with_fewer_than_4_points(self):
        with pytest.raises(ValueError):
            build_homography([[0, 0], [1, 0]], [[0.0, 0.0], [1.0, 0.0]])

    def test_known_point_maps_correctly(self, homography):
        """The top-left src point should map to [0, 0] metres."""
        H, _ = homography
        result = pixel_to_ground(H, np.array([100.0, 200.0]))
        assert abs(result[0]) < 0.5, f"Expected x≈0, got {result[0]:.4f}"
        assert abs(result[1]) < 0.5, f"Expected y≈0, got {result[1]:.4f}"

    def test_horizontal_distance_is_correct(self, homography):
        """The two top points are 3.5 m apart — verify the mapping."""
        H, _ = homography
        g_left = pixel_to_ground(H, np.array([100.0, 200.0]))
        g_right = pixel_to_ground(H, np.array([200.0, 200.0]))
        dist = np.linalg.norm(g_right - g_left)
        assert abs(dist - 3.5) < 0.5, f"Expected ~3.5 m, got {dist:.4f} m"


# ── pixel_to_ground tests ──────────────────────────────────────────────────────

class TestPixelToGround:
    def test_output_shape(self, homography):
        H, _ = homography
        result = pixel_to_ground(H, np.array([150.0, 250.0]))
        assert result.shape == (2,), "pixel_to_ground must return a (2,) array"

    def test_all_finite(self, homography):
        H, _ = homography
        result = pixel_to_ground(H, np.array([150.0, 250.0]))
        assert np.all(np.isfinite(result)), "pixel_to_ground returned non-finite values"


# ── estimate_speed tests ───────────────────────────────────────────────────────

class TestEstimateSpeed:
    def test_stationary_point_returns_zero(self, homography):
        H, _ = homography
        spd = estimate_speed(H, np.array([150.0, 250.0]), np.array([150.0, 250.0]), fps=25.0)
        assert spd == pytest.approx(0.0, abs=0.1), "Stationary point should give ~0 km/h"

    def test_moving_point_returns_positive(self, homography):
        H, _ = homography
        spd = estimate_speed(H, np.array([100.0, 200.0]), np.array([105.0, 200.0]), fps=25.0)
        assert spd > 0, "Moving point must return positive speed"

    def test_speed_scale_with_fps(self, homography):
        """Doubling fps should double the estimated speed."""
        H, _ = homography
        p_old = np.array([100.0, 200.0])
        p_new = np.array([110.0, 200.0])
        spd_25 = estimate_speed(H, p_old, p_new, fps=25.0)
        spd_50 = estimate_speed(H, p_old, p_new, fps=50.0)
        assert abs(spd_50 - 2 * spd_25) < 0.1, "Speed should scale linearly with fps"

    def test_does_not_return_negative(self, homography):
        H, _ = homography
        spd = estimate_speed(H, np.array([150.0, 250.0]), np.array([148.0, 251.0]), fps=25.0)
        assert spd >= 0.0, "Speed must never be negative"


# ── estimate_speeds_batch tests ────────────────────────────────────────────────

class TestEstimateSpeedsBatch:
    def test_empty_input_returns_empty(self, homography):
        H, _ = homography
        result = estimate_speeds_batch(H, np.empty((0, 2)), np.empty((0, 2)), fps=25.0)
        assert len(result) == 0

    def test_filters_implausible_speeds(self, homography):
        """A huge pixel jump should be filtered out by max_kmh."""
        H, _ = homography
        pts_old = np.array([[100.0, 200.0]])
        pts_new = np.array([[9000.0, 200.0]])   # absurdly large jump
        result = estimate_speeds_batch(H, pts_old, pts_new, fps=25.0, max_kmh=120.0)
        assert len(result) == 0, "Implausibly fast reading should be filtered"

    def test_valid_batch_returns_array(self, homography):
        H, _ = homography
        pts_old = np.array([[100.0, 200.0], [150.0, 250.0]])
        pts_new = np.array([[103.0, 200.0], [153.0, 250.0]])
        result = estimate_speeds_batch(H, pts_old, pts_new, fps=25.0)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


# ── rolling_average_speed tests ────────────────────────────────────────────────

class TestRollingAverageSpeed:
    def test_empty_buffer(self):
        assert rolling_average_speed([]) == 0.0

    def test_single_value(self):
        assert rolling_average_speed([50.0]) == pytest.approx(50.0)

    def test_window_clipping(self):
        """Window of 3 over a 5-element buffer should use only last 3."""
        buf = [10.0, 20.0, 30.0, 40.0, 50.0]
        avg = rolling_average_speed(buf, window=3)
        expected = (30.0 + 40.0 + 50.0) / 3
        assert avg == pytest.approx(expected, abs=0.01)
