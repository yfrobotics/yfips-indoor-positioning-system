# YF-IPS: Robot Indoor Positioning System
![](https://img.shields.io/github/stars/yfrobotics/yfips-indoor-positioning-system) ![](https://img.shields.io/github/issues/yfrobotics/yfips-indoor-positioning-system) ![](https://img.shields.io/github/license/yfrobotics/yfips-indoor-positioning-system)

## 1. Introduction
Traditional localisation systems are unfriendly to robot researchers and learners: they're expensive and take a lot of effort to set up. YF-IPS targets a low-cost indoor positioning system (IPS) with reasonable accuracy, usable for tracking and navigation beyond robotics as well.

The system is designed to locate 1–100 robots at ≥10 fps over a 5×5 m area, returning `(id, x, y, yaw)` — where `id` is the robot identifier, `(x, y)` is the 2D position relative to a world origin (in metres), and `yaw` is the rotation about z (in radians).

Two planned variants exist: vision-based and ToF/tag-based, usable together or separately. This repo currently implements the **vision** variant, with two selectable detection modes:

1. **AprilTag mode** — robots carry an AprilTag fiducial; detection uses the [swatbotics AprilTag](https://github.com/swatbotics/apriltag) binding.
2. **Reference-image mode** — no fiducials; each robot is localized by matching an overhead reference image against the live scene using ORB features + RANSAC homography.

## 2. Design specification
- Detect robots in a 5×5 m region.
- Simplest setup: ±20 cm precision. More sensors can improve to ±10 cm.
- 1–10 robots simultaneously at ≥10 fps.
- Output publishable as UDP JSON (implemented) or a ROS topic (planned).

## 3. Hardware
- **PC** (desktop or laptop) to run the program.
- **Camera** — a USB webcam or an IP camera with an RTSP stream. Multiple cameras can run together; configure each input in `config.json` under `cameras`.
- **AprilTag printouts** — for AprilTag mode.
- **Reference images** — for image mode (top-down crops of each robot, one file per id).
- Wireless anchors/tags — TBD (ToF variant not yet implemented).

## 4. Install and run

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                                            # create .venv and install deps
uv run python -m yfips.calibration                 # one-time camera calibration
uv run python -m yfips.detection --mode apriltag   # or --mode image
```

Mode can also be set permanently via `"mode": "apriltag" | "image"` in `config.json`.

### RTSP cameras

For a single-camera configuration, override the input with:

```bash
uv run python -m yfips.detection --camera-url 'rtsp://user:password@192.168.1.100:554/stream1'
```

For persistent or mixed webcam/IP-camera setups, add `rtsp_url` to each
network camera entry in `config.json`:

```json
{
  "name": "overhead",
  "rtsp_url": "rtsp://192.168.1.100:554/stream1",
  "width": 1920,
  "height": 1080,
  "fps": 30,
  "camera_matrix": null,
  "dist_coeffs": null,
  "image_corners_px": null
}
```

Place this entry in the `cameras` array. Each camera needs a unique name and
input. `rtsp_url` takes precedence over `index`; local webcams continue to use
`index`. `--camera-index` switches a single-camera setup back to a webcam.
`--camera-url` and `--camera-index` are mutually exclusive. All camera CLI
overrides (`--camera-url`, `--camera-index`, `--width`, `--height`, `--fps`)
require exactly one configured camera; edit `cameras` for multi-camera setups.

Configure stream resolution and FPS on the IP camera. YF-IPS does not change
these stream settings; set `width` and `height` to the stream's actual size
for lens undistortion, and calibrate using images from that same stream.
RTSP capture prioritizes responsiveness: it explicitly uses OpenCV's FFmpeg
backend with reduced startup buffering, low-delay decoding and one decoder
thread. Capture runs continuously in the background and detection takes only
the newest decoded frame, skipping intermediate frames when it falls behind.
Connection attempts time out after 5 seconds and individual reads after 3
seconds; these are failure timeouts, not a bound on end-to-end video delay.

TCP is the default transport for compatibility. TCP retransmissions and
camera-side encoding can still introduce delay. For a server and network that
support UDP, you can favor packet loss over waiting for retransmission:

```bash
OPENCV_FFMPEG_CAPTURE_OPTIONS='rtsp_transport;udp|fflags;nobuffer|flags;low_delay|threads;1|max_delay;0|reorder_queue_size;0' \
  uv run python -m yfips.detection
```

This example uses the RTSP URL saved in `config.json`; alternatively add
`--camera-url`. UDP can produce damaged or missing frames. An explicit
`OPENCV_FFMPEG_CAPTURE_OPTIONS` value replaces the defaults for all RTSP
cameras in the process. See the [OpenCV options reference](https://docs.opencv.org/4.x/d6/dea/tutorial_env_reference.html)
and [FFmpeg RTSP documentation](https://ffmpeg.org/ffmpeg-protocols.html#rtsp).

If opening fails,
check the camera's stream URL, credentials, network reachability and backend
support. Failed cameras are skipped so other inputs can continue.

### UDP output
Detections are published as JSON datagrams `{"id", "x", "y", "yaw", "t"}` to `127.0.0.1:9999` by default (configurable under `udp` in `config.json`). Quick listener:

```bash
python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(('127.0.0.1',9999))
while True: print(s.recvfrom(4096)[0].decode())"
```

## 5. Calibration

### 5.1 Camera calibration
For each camera, capture ~15 images of a 7×6-inner-corner chessboard at the
resolution used for detection. Save them as
`images/<camera_name>/calibration_*.jpg`, using the camera's configured `name`.
For the RTSP example above:

```bash
uv run python -m yfips.calibration --camera overhead
```

Without `--camera`, calibration uses the first configured camera. It writes
the camera matrix and distortion coefficients to that camera's entry in
`config.json` and clears its saved world-corner calibration. Re-select those
corners in the detection preview after recalibrating.

### 5.2 Environment calibration
Run `uv run python -m yfips.detection` and **double-click the 4 world corners** of your playing area inside each camera's video window, in the same order as `world_corners_m` in `config.json` (default: `(0,0), (5,0), (5,5), (0,5)` metres). The image→world homography is computed and persisted; subsequent runs reuse it. A 5th click resets and re-captures.

## 6. Detection modes

### 6.1 AprilTag mode (default)
Print AprilTags (any supported family), mount one per robot with the tag id corresponding to the robot id. Position is recovered via the world-plane homography (no PnP), so the printed side length doesn't affect localization — but keep tags large enough to detect reliably (≥5 cm for a 1080p camera at ~3 m height).

Tunables under `apriltag_mode` in `config.json` map onto `apriltag.DetectorOptions`:

```json
"apriltag_mode": {
  "family": "tag36h11",
  "nthreads": 4,
  "quad_decimate": 1.0,
  "refine_edges": true
}
```

`quad_decimate > 1.0` trades detection range for speed (downsamples before quad detection). `nthreads` parallelizes across cores.

### 6.2 Reference-image mode
Create a `references/` directory at the repo root and drop one image per robot:

```
references/
  1.png     # top-down image of robot with id=1
  2.png
  7.jpg
```

The filename stem must be the integer robot id. Requirements for good detection:
- Planar, textured top-down view of the robot (the feature matcher assumes a plane).
- Enough texture — flat colours or reflective surfaces produce few ORB features.
- Ideally the same approximate scale as the robot will appear in the scene.

Tunables under `image_mode` in `config.json`:
- `min_inliers` — RANSAC inlier threshold (default 15). Raise for robustness, lower for faint matches.
- `use_flann` — use FLANN-LSH instead of brute-force matcher (default `false`). Faster with many references.

Optional **sidecar JSON** per reference declares which way is forward in the reference image. Drop `references/<id>.json` next to the image:

```json
{ "forward_deg": 90 }
```

`forward_deg` is measured CCW from the image's +x axis (math convention; 0 = right, 90 = up, 180 = left, 270 = down). Default `0` matches pre-1.0 behavior (forward = right edge).

## 7. Live visualizer

A second terminal can subscribe to the UDP stream and show each robot as an arrow on the world plane:

```bash
uv run python -m yfips.gui
```

## 8. Extras in `config.json`

- `undistort` — apply lens undistortion per frame using the calibrated intrinsics (default `true`).
- `tracker` — per-id smoothing across frames. Two types:
  - **EMA** (default, low overhead):
    ```json
    "tracker": { "enabled": true, "type": "ema", "alpha": 0.4, "timeout_s": 1.0 }
    ```
  - **Kalman** (constant-velocity model on `[x, y, yaw, vx, vy, ω]`; predicts through brief dropouts, wraps yaw residuals):
    ```json
    "tracker": {
      "enabled": true,
      "type": "kalman",
      "q_accel": 1.0,
      "r_pos": 0.05,
      "r_yaw": 0.1,
      "timeout_s": 1.0
    }
    ```
    Raise `q_accel` to react faster; raise `r_pos`/`r_yaw` to smooth harder. The filter re-initializes for an id if no measurement arrives within `timeout_s`.

    On frames where a tracked id is missing (briefly occluded, missed match), the Kalman tracker emits a UDP/ROS payload with `"predicted": true` derived from the constant-velocity model. EMA cannot extrapolate and stays silent on dropouts.
- `ros` — optional ROS 2 publisher: `{ "enabled": false, "topic": "/yfips/detections" }`. Requires `rclpy`; otherwise gracefully disables itself.

## 9. Credits
- Built on [OpenCV](https://opencv.org/).
- AprilTag detection via the swatbotics binding: https://github.com/swatbotics/apriltag

## 10. Contributors
- [automaticdai](https://github.com/automaticdai)
- [xinyu-xu-dev](https://github.com/xinyu-xu-dev)

## License
MIT
