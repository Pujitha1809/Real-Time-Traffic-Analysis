# RVTAS — Real-Time Vehicle Traffic Analytics System

Classical computer vision pipeline for vehicle speed estimation, pedestrian detection, and traffic density heatmapping — no deep learning, runs on CPU.

---

## Setup

```bash
git clone <your-repo>
cd rvtas
pip install -r requirements.txt
```

**Python 3.9+ required.**

---

## Quickstart

```bash
# 1. Calibrate your camera (skip if using pre-saved calibration)
python camera_cal.py calibration/images/

# 2. Edit config.yaml — set your src_pts / dst_pts_metres for your camera view
# 3. Run the pipeline
python main.py --config config.yaml

# Override input/output without editing config:
python main.py --input samples/myvideo.mp4 --output reports/out.mp4
```

---

## Pipeline stages

| # | Module | What it does |
|---|--------|-------------|
| 1 | `camera_cal.py` | Undistorts frames using chessboard calibration |
| 2 | `bg_sub.py` | MOG2 background subtraction — isolates moving objects |
| 3 | `preprocess.py` | DFT low-pass filter — removes sensor noise |
| 4 | `flow_tracker.py` | Lucas-Kanade sparse optical flow — tracks vehicle features |
| 5 | `feature_match.py` | ORB + BFMatcher — validates homography / handles occlusion |
| 6 | `homography.py` | Perspective warp — converts image → ground-plane metres |
| 7 | `speed_est.py` | Converts pixel displacement to km/h via homography |
| 8 | `ped_detect.py` | HOG + SVM pedestrian detector with NMS |
| 9 | `visualiser.py` | Annotated video output + heatmap PNG export |

---

## Configuration (`config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dft_radius` | 45 | Low-pass filter cut-off (pixels). Lower = more aggressive smoothing |
| `min_tracks` | 80 | Re-detect corners when active tracks drop below this |
| `max_speed_kmh` | 120 | Discard implausible speed readings above this |
| `src_pts` | — | 4 pixel coordinates of known road features (set for your camera) |
| `dst_pts_metres` | — | Corresponding real-world positions in metres |
| `hog_scale` | 1.05 | HOG detection pyramid scale |
| `nms_iou_thresh` | 0.4 | IoU threshold for pedestrian NMS |
| `heatmap_interval_sec` | 60 | Save heatmap PNG every N seconds |

---

## Calibration

1. Print a chessboard (9×6 inner corners recommended).
2. Take 25+ photos from varied angles/distances.
3. Place images in `calibration/images/`.
4. Run `python camera_cal.py calibration/images/`.
5. Target reprojection error < 0.8 px.

---

## Setting `src_pts` / `dst_pts_metres`

On your target camera view, identify 4 road features whose real positions you know (e.g. lane markings 3.5 m apart, stop lines 10 m apart). Record their pixel coordinates in `src_pts` and their ground-plane metre coordinates in `dst_pts_metres`.

Tip: use `cv2.imshow` on a still frame and note pixel coords by hovering. A correct calibration will show vehicles' speed readings within ~15% of their actual speed.

---

## Running tests

```bash
pytest tests/ -v
```

All tests must pass for `speed_est.py` and `homography.py`.

---

## Outputs

| File | Description |
|------|-------------|
| `reports/output_annotated.mp4` | Annotated video with flow vectors, speed labels, pedestrian boxes |
| `reports/sample_heatmap.png` | Final traffic density heatmap (COLORMAP_JET) |
| `reports/heatmap_<timestamp>.png` | Per-interval heatmap saves |
| `reports/run_log.json` | Run statistics: vehicle count, avg speed, pedestrian events |

---

## Performance tips

- Target ≥ 15 fps on 720p CPU. If slower:
  - Increase `dft_radius` (less DFT padding needed)
  - Increase `hog_win_stride` to `[16, 16]`
  - Run HOG every 5th frame instead of every 3rd
  - Reduce `maxCorners` in `flow_tracker.py`

---

## Interview talking points

> *"I built a classical CV pipeline for a smart city traffic analytics scenario — no deep learning, so it could run on the client's existing edge hardware. One key trade-off I made was using DFT-based filtering over Gaussian blur, because I needed precise frequency control to preserve the fine vehicle edge detail required by the optical flow tracker."*
