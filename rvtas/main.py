"""
main.py — RVTAS pipeline entry point.

Wires all modules together in a single processing loop:
    1. Undistort frame (camera_cal)
    2. DFT low-pass filter (preprocess)
    3. Background subtraction (bg_sub)
    4. Lucas-Kanade optical flow (flow_tracker)
    5. Speed estimation via homography (speed_est)
    6. HOG pedestrian detection (ped_detect)
    7. Heatmap accumulation + periodic save (visualiser)
    8. Dashboard overlay and annotated video write (visualiser)

Usage:
    python main.py [--config config.yaml]
    python main.py --input samples/myvideo.mp4 --output reports/out.mp4
"""

import cv2
import yaml
import json
import time
import argparse
import os
import numpy as np

from camera_cal import undistort_frame, load_params
from preprocess import dft_lowpass
from bg_sub import create_background_subtractor, apply_background_subtraction
from flow_tracker import detect_corners, track_flow, needs_refresh, filter_by_magnitude
from homography import build_homography
from speed_est import estimate_speeds_batch, rolling_average_speed
from ped_detect import detect_pedestrians
from visualiser import draw_overlay, draw_dashboard, update_heatmap, save_heatmap
from utils import FPSCounter, write_json_log, parse_yaml_pts


def process_video(input_path: str, output_path: str, cfg: dict) -> dict:
    """
    Main pipeline. Processes input_path frame-by-frame and writes annotated
    video to output_path. Returns a run-log dictionary.

    Args:
        input_path: Path to input video file, or RTSP stream URL.
        output_path: Path for the annotated output .mp4.
        cfg: Parsed config.yaml dictionary.

    Returns:
        Dictionary with keys: total_vehicles, avg_speed_kmh, pedestrian_events,
        frames_processed, duration_sec.
    """
    # ── Load calibration ──────────────────────────────────────────────────────
    cal_path = cfg.get("calibration_path", "calibration/camera_params.npz")
    if os.path.exists(cal_path):
        K, dist = load_params(cal_path)
        print(f"[main] Loaded calibration from {cal_path}")
    else:
        print(f"[main] WARNING: No calibration file at {cal_path}. Skipping undistortion.")
        K, dist = None, None

    # ── Build homography ──────────────────────────────────────────────────────
    H, scale_mpp = build_homography(cfg["src_pts"], cfg["dst_pts_metres"])
    print(f"[main] Homography built. Scale ≈ {scale_mpp*100:.2f} cm/px")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[main] Source: {w}x{h} @ {fps_video:.1f} fps")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_video,
        (w, h),
    )

    # ── Init modules ──────────────────────────────────────────────────────────
    subtractor = create_background_subtractor()
    heatmap = np.zeros((h, w), dtype=np.float32)
    fps_counter = FPSCounter(window=30)
    speed_buffer = []

    p0 = None
    old_gray = None

    # ── Run log ───────────────────────────────────────────────────────────────
    log = {
        "input": input_path,
        "output": output_path,
        "total_vehicles": 0,
        "speed_readings": [],
        "avg_speed_kmh": 0.0,
        "pedestrian_events": 0,
        "frames_processed": 0,
        "duration_sec": 0.0,
        "heatmap_saves": [],
    }

    start_time = time.perf_counter()
    heatmap_timer = start_time
    heatmap_interval = cfg.get("heatmap_interval_sec", 60)
    frame_num = 0

    print("[main] Processing — press Q to quit early if showing preview.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Stage 1 — Undistort
        if K is not None:
            frame = undistort_frame(frame, K, dist)

        # Stage 2 — Background subtraction (generates motion mask for corner detect)
        fg_mask = apply_background_subtraction(subtractor, frame)

        # Stage 3 — DFT low-pass filter (grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = dft_lowpass(gray, radius=cfg.get("dft_radius", 45))

        # Stage 4 — Lucas-Kanade optical flow
        if old_gray is None:
            old_gray = gray
            p0 = detect_corners(gray, mask=fg_mask)
            continue

        if needs_refresh(p0, min_tracks=cfg.get("min_tracks", 80)):
            p0 = detect_corners(gray, mask=fg_mask)

        good_new, good_old = track_flow(old_gray, gray, p0)
        good_new, good_old = filter_by_magnitude(good_new, good_old, min_mag=1.0, max_mag=80.0)

        # Stage 5 & 6 — Speed estimation
        speeds_arr = estimate_speeds_batch(
            H, good_old, good_new, fps_video,
            min_kmh=2.0,
            max_kmh=cfg.get("max_speed_kmh", 120),
        )
        speeds_list = speeds_arr.tolist()
        if speeds_list:
            speed_buffer.extend(speeds_list)
            log["speed_readings"].extend([round(s, 1) for s in speeds_list])
            # Rough vehicle count: each cluster of readings is a vehicle
            log["total_vehicles"] += max(1, len(speeds_list) // 10)

        avg_speed = rolling_average_speed(speed_buffer, window=30)

        # Stage 7 — HOG pedestrian detection (every 3rd frame to hit 15fps)
        ped_boxes, ped_weights = [], []
        if frame_num % 3 == 0:
            ped_boxes, ped_weights = detect_pedestrians(
                frame,
                win_stride=tuple(cfg.get("hog_win_stride", [8, 8])),
                padding=tuple(cfg.get("hog_padding", [4, 4])),
                scale=cfg.get("hog_scale", 1.05),
                nms_thresh=cfg.get("nms_iou_thresh", 0.4),
            )
            log["pedestrian_events"] += len(ped_boxes)

        # Stage 8 — Update heatmap; save every heatmap_interval seconds
        update_heatmap(heatmap, good_new, good_old)
        now = time.perf_counter()
        if now - heatmap_timer >= heatmap_interval:
            hm_path = f"reports/heatmap_{int(now)}.png"
            save_heatmap(heatmap, hm_path)
            log["heatmap_saves"].append(hm_path)
            print(f"[main] Heatmap saved: {hm_path}")
            heatmap_timer = now

        # Stage 9 — Annotate and write output frame
        current_fps = fps_counter.tick()
        draw_overlay(frame, good_new, good_old, speeds_list, ped_boxes, ped_weights)
        draw_dashboard(frame, current_fps, log["total_vehicles"], avg_speed,
                       log["pedestrian_events"], frame_num)
        out.write(frame)

        # Advance state
        old_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

        log["frames_processed"] = frame_num

    # ── Finalise ──────────────────────────────────────────────────────────────
    duration = time.perf_counter() - start_time
    log["duration_sec"] = round(duration, 2)
    log["avg_speed_kmh"] = round(
        float(np.mean(log["speed_readings"])) if log["speed_readings"] else 0.0, 1
    )
    # Trim raw readings from final log (keep summary only)
    log.pop("speed_readings")

    cap.release()
    out.release()

    # Save final heatmap
    final_hm = "reports/sample_heatmap.png"
    save_heatmap(heatmap, final_hm)
    log["heatmap_saves"].append(final_hm)

    print(f"\n[main] Done. {frame_num} frames in {duration:.1f}s "
          f"({frame_num/duration:.1f} fps throughput)")
    print(f"[main] Output: {output_path}")
    return log


def main():
    parser = argparse.ArgumentParser(description="RVTAS — Real-Time Vehicle Traffic Analytics")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", default=None, help="Override config input path")
    parser.add_argument("--output", default=None, help="Override config output path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.input:
        cfg["input"] = args.input
    if args.output:
        cfg["output"] = args.output

    log = process_video(cfg["input"], cfg["output"], cfg)

    log_path = "reports/run_log.json"
    write_json_log(log, log_path)
    print(f"\n[main] Run log: {log_path}")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
