# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

YF-IPS: a low-cost indoor positioning system for tracking 1–100 robots in a ~5×5 m area at ≥10 fps, returning `(id, x, y, yaw)`. Vision-based (webcam) with two detection modes — **AprilTag** and **reference-image (ORB)**. Early-stage research code.

## Setup & Run

Package management uses [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` (`requirements.txt` is kept for reference but no longer authoritative).

```bash
uv sync                                               # create .venv + install deps
uv run python -m yfips.calibration --camera cam0      # calibrate a camera (default: first in cameras[])
uv run python -m yfips.detection --mode apriltag      # default mode
uv run python -m yfips.detection --mode image         # reference-image mode
uv run pytest                                         # unit tests
uv run ruff check .                                   # lint
```

Mode can also be set via `"mode"` in `config.json`. Add deps with `uv add <pkg>` (runtime) or `uv add --group dev <pkg>` (dev).

Inside the window: double-click 4 world corners in the order of `world_corners_m` (default `(0,0), (5,0), (5,5), (0,5)` m) to compute and persist the image→world homography. 5th click resets. ESC quits.

## Architecture

All state flows through `config.json` (at repo root). Source lives under `src/yfips/` as an installable package — run modules with `python -m yfips.<name>`.

- `src/yfips/config.py` — load/save `config.json`. Holds a `cameras[]` array (each entry: `name`, `index`, `width`, `height`, `fps`, `camera_matrix`, `dist_coeffs`, `image_corners_px`), plus global `world_corners_m`, `udp`, `tracker`, `mode`, `references_dir`, `image_mode`, `apriltag_mode`, `undistort`, `ros`. Legacy flat single-camera configs are auto-migrated into `cameras=[{name:"cam0",...}]` on first load.
- `src/yfips/calibration.py` — takes `--camera <name>`; reads `images/<name>/calibration_*.jpg` and writes intrinsics into the matching `cameras[]` entry. Only that camera's `image_corners_px` is cleared.
- `src/yfips/camera_worker.py` — `CameraWorker` thread per camera: owns its own `VideoCapture`, `FrameGrabber`, undistort maps, detector, homography (`_PerCameraClicker`), and preview window `YFIPS:<name>`. Emits `Detection(rid, x, y, yaw, t, camera)` items onto a shared queue with drop-oldest backpressure.
- `src/yfips/detection.py` — orchestrator.
  - Loads + validates config, spawns one `CameraWorker` per `cameras[]` entry.
  - Main thread drains the shared detection queue, updates a shared tracker under a `tracker_lock`, and publishes `{id, x, y, yaw, t, camera}` to UDP + ROS.
  - A daemon prediction-timer thread fires every `tracker.prediction_interval_s` (default 50 ms) and emits `{predicted:true}` payloads for ids whose last measurement is older than that interval (prevents spam right after a real detection).
  - Still owns the shared detector builder (`build_detector`), homography helpers (`compute_homography`, `image_to_world`, `yaw_from_forward`), publisher (`Publisher`), frame grabber + fps meter (`FrameGrabber`, `FpsMeter`), and AprilTag adapter.
- `src/yfips/image_detector.py` — `ImageRefDetector`: ORB + BFMatcher/FLANN + RANSAC homography. Loads references from `references/<id>.{png,jpg}` where the filename stem is the integer robot id. Emits the same `{id, center, forward, corners}` shape as the AprilTag adapter so downstream world-transform code is shared. Set `image_mode.use_flann=true` for LSH-based matching at scale.
- `src/yfips/tracker.py` / `kalman_tracker.py` — per-id smoothers. Tracker-level fusion happens implicitly: same-id detections from different cameras land in the same `tracker.update()` call keyed on `rid`. Both expose `last_measurement_t(rid)` for the prediction cooldown.
- `src/yfips/ros_publisher.py` — optional ROS 2 publisher (std_msgs/String JSON). No-ops if `rclpy` isn't installed. Enable via `ros.enabled=true`.
- `src/yfips/gui.py` — matplotlib live visualizer; listens to the UDP stream and plots each tracked robot as an arrow on the world plane. Run in a second terminal: `uv run python -m yfips.gui`. Payloads now carry a `"camera"` field on detections (omitted on predicted packets).
- `tests/` — pytest unit tests over pure-logic modules (trackers, homography, guards, config migration, camera worker, fusion, orchestrator).

## Multi-camera runbook

To add a second camera:

1. Append an entry to `config.json`'s `cameras[]`:
   ```json
   {"name": "cam1", "index": 2, "width": 640, "height": 480, "fps": 60,
    "camera_matrix": null, "dist_coeffs": null, "image_corners_px": null}
   ```
2. Drop chessboard images into `images/cam1/calibration_*.jpg`.
3. `uv run python -m yfips.calibration --camera cam1`.
4. Start detection. A `YFIPS:cam1` window opens; double-click 4 world corners in that window.
5. Verify via the UDP listener: payloads should appear with `"camera": "cam1"`.

CLI flags `--camera-index/--width/--height/--fps` are only honoured with a single camera in `cameras[]`; with N>1, edit `config.json` instead.

UDP listener (debug):
```bash
python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(('127.0.0.1',9999))
while True: print(s.recvfrom(4096)[0].decode())"
```

## Gotchas

- `apriltag` package name on PyPI varies by platform (swatbotics binding).
- Running `yfips.calibration --camera <name>` auto-clears that camera's saved `image_corners_px`, because the old pixel coordinates refer to an image rectified with the old intrinsics. Re-click the 4 world corners after calibration.
- Camera index, resolution and fps live per-entry in `config.json`'s `cameras[]`; CLI flags `--camera-index --width --height --fps` override at runtime **only** when there's a single camera in `cameras[]`.
- Image mode cost scales linearly with number of references; for ≳50 robots swap BFMatcher for FLANN.
- World positioning assumes robots move on a **plane**; tall tags/robots get parallax error even after undistortion.
- `ros` mode requires a ROS 2 install (Humble+) with `rclpy` on PYTHONPATH; otherwise it no-ops with a warning.
- Toggling `undistort` after clicking world corners still misaligns them — only a recalibration auto-invalidates them. If you flip `undistort` manually, re-click every camera's corners.
- Duplicate `name` or `index` in `cameras[]` is a config error; startup fails with a clear message.
- `cv2.imshow` / `waitKey` must run on the main thread. Workers only draw into a frame buffer; the orchestrator calls `imshow` and `waitKey` for every worker's latest frame once per main-loop tick.
- Per-camera detector instances are a requirement, not a convenience: the AprilTag and ORB detectors are not safe to share across threads.
