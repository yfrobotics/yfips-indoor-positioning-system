# Multi-Camera Integration — Design

Date: 2026-04-17
Branch: `feature/multi-camera`

## Goal

Extend YF-IPS from a single webcam to **N cameras** with partially overlapping
fields of view, all mapped into the same 5×5 m world frame, producing a single
fused `(id, x, y, yaw)` stream.

## Non-goals (v1)

- Per-frame averaging or confidence-weighted fusion (tracker-level fusion only).
- Cross-camera extrinsic calibration via a shared target (every camera gets its
  own 4-corner click into the world frame).
- Full 3D `solvePnP` — planar-motion assumption stays.
- `multiprocessing`-based workers — stays single process.
- Headless / no-preview mode.
- Auto-discovery of cameras.
- Hardware-synchronised capture.

## Architecture

Single process, N threads. Each camera owns a capture + detection pipeline
feeding a shared, locked tracker; one orchestrator publishes to UDP and ROS.

```
  cam0 thread ──┐
  cam1 thread ──┼─► queue ─► orchestrator ─► tracker ─► publisher (UDP, ROS)
  cam2 thread ──┤            (main thread)   (locked)
  ...         ──┘
                                         ▲
                                         │
                            prediction timer thread
```

Key invariants:

- Exactly **one world frame** (`world_corners_m`) and **one tracker**.
- Each camera's homography is private to its worker.
- Detector instances are per-worker (AprilTag and ORB detectors are not
  guaranteed thread-safe across parallel `detect()` calls).

## Components

### `config.py` — schema change + legacy migration

New shape:

```json
{
  "cameras": [
    {"name": "cam0", "index": 0, "width": 640, "height": 480, "fps": 60,
     "camera_matrix": [...], "dist_coeffs": [...],
     "image_corners_px": [[...], ...]},
    {"name": "cam1", "index": 2, "width": 640, "height": 480, "fps": 60,
     "camera_matrix": [...], "dist_coeffs": [...],
     "image_corners_px": null}
  ],
  "world_corners_m": [[0,0],[5,0],[5,5],[0,5]],
  "mode": "apriltag",
  "udp": {...},
  "tracker": {...},
  "ros": {...},
  "undistort": true,
  "references_dir": null,
  "image_mode": {...},
  "apriltag_mode": {...}
}
```

Top-level keys that stay global: `world_corners_m`, `mode`, `udp`, `tracker`,
`ros`, `undistort`, `references_dir`, `image_mode`, `apriltag_mode`,
`tag_size_m`.

`tracker` gains one new field:

- `prediction_interval_s` (float, default `0.05`) — cadence of the prediction
  timer thread.

**Legacy migration** runs on `load()`:

- If `cameras` is absent and any of `camera_matrix` / `dist_coeffs` /
  `image_corners_px` / `camera` exist at top level, build
  `cameras = [{name: "cam0", index: camera.index, width, height, fps,
  camera_matrix, dist_coeffs, image_corners_px}]` and drop those keys from the
  top level. Re-save immediately.
- If `cameras` is absent and no legacy keys exist, default to
  `cameras=[{name:"cam0", index:0, width:640, height:480, fps:60,
  camera_matrix:null, dist_coeffs:null, image_corners_px:null}]`.

**Validation** (`validate_cameras(cameras)`):

- Non-empty list.
- Unique `name` across entries.
- Unique `index` across entries.

Raises `ValueError` on violation; orchestrator surfaces as `SystemExit` with a
clear message.

### `camera_worker.py` (new)

```python
class CameraWorker:
    def __init__(self, cam_cfg, world_corners_m, detector, undistort_enabled,
                 out_queue, stop_event, config_lock): ...
    def start(self) -> CameraWorker: ...
    def stop(self) -> None: ...
    # click callback persists the camera's own image_corners_px under config_lock
```

Owns per-instance: `cv2.VideoCapture`, `FrameGrabber`, undistort maps,
`CalibClicker` (scoped to its own `image_corners_px`), OpenCV window
`YFIPS:<name>`, a detector. Emits
`Detection(rid:int, x:float, y:float, yaw:float, t:float, camera:str)` onto the
shared queue. Uses `queue.put_nowait` + drop-oldest on full; increments a
`dropped` counter logged periodically.

When homography is not yet set, the worker still reads frames and draws
overlays but pushes **nothing** onto the queue (no world coordinates available).

### `detection.py` — orchestrator

- Parse CLI, load+validate config.
- Build one detector per camera via `build_detector(mode, cfg)`.
- Construct `CameraWorker` per `cameras[]` entry, each with the shared queue,
  stop event, and `config_lock`.
- Start all workers. If a worker fails to open its capture, log and drop it;
  exit only if none open.
- Enter the orchestrator loop (main thread):
  - `waitKey(1)` (required on the main thread for OpenCV UIs); ESC sets
    `stop_event`.
  - For each worker, `cv2.imshow("YFIPS:<name>", worker.latest_drawn_frame())`.
  - Drain queue non-blockingly. For each `Detection`:
    - `with tracker_lock: x,y,yaw = tracker.update(rid,x,y,yaw,t)`
    - Publish `{id, x, y, yaw, t, camera}` via UDP + ROS.
- Prediction timer thread (daemon) wakes every
  `tracker.prediction_interval_s` (default 50 ms) and, under `tracker_lock`,
  emits predictions for every tracked id whose last real update was **older
  than `prediction_interval_s`** but still within `timeout_s`. This "one
  cooldown tick" rule prevents emitting a prediction packet immediately after
  a real detection arrived from any camera. `emit_predictions` is extended to
  accept a `min_age_s` threshold and apply it against the tracker's per-id
  last-update timestamp (exposed via a new `tracker.last_update_t(rid)`
  accessor; both `EMATracker` and `KalmanTracker` already store this
  internally). Each prediction publishes with `"predicted": true`; the
  `"camera"` field is omitted (no single camera owns it).
- Shutdown: set stop, join workers, release captures, destroy windows, shut
  down ROS publisher.

### `calibration.py`

- New flag `--camera <name>` (default: first camera in `cameras[]`).
- Reads from `images/<name>/calibration_*.jpg` (was `images/calibration_*.jpg`;
  the new default for a single migrated camera named `cam0` is
  `images/cam0/…`). The README/CLAUDE.md runbook will document the new layout.
- Writes `camera_matrix` + `dist_coeffs` into that camera's entry.
- Clears **only** that camera's `image_corners_px`.
- Respects `config_lock`-style single-writer semantics by loading, updating,
  saving the whole `cameras[]` array.

### `tracker.py` / `kalman_tracker.py`

APIs unchanged. Concurrency: the orchestrator guards `update()` and
`predict_only()` under a single `tracker_lock`. Both are O(1) per id, so
contention is negligible.

### `gui.py`

Unchanged. Already listens to UDP and plots whatever arrives. Will start
showing the `"camera"` field naturally if we ever add a per-camera color, but
that's cosmetic and not part of this branch.

### CLI

Preserved: `--mode`, `--camera-index`, `--width`, `--height`, `--fps`.

- `--mode` is global (as today).
- `--camera-index`, `--width`, `--height`, `--fps` are honoured only when
  `len(cameras) == 1`; with more than one camera they raise a clear error
  ("edit config.json cameras[] instead"). This keeps today's single-camera
  workflow trivially working.

## Data flow

Per camera worker thread, per frame:

```
FrameGrabber → latest frame
  → undistort (per-camera maps)
  → grayscale → detector.detect(gray) → [det, ...]
  → for each det:
        cw  = image_to_world(H_cam, det.center)
        fw  = image_to_world(H_cam, det.forward)
        yaw = atan2(fw.y - cw.y, fw.x - cw.x)
        queue.put(Detection(rid, cw.x, cw.y, yaw, now, cam.name))
  → draw overlays on its own frame
  → publish frame buffer to `worker.latest_drawn_frame()`
```

Orchestrator (main thread):

```
while not stop_event.is_set():
    waitKey(1)  # ESC check, required for OpenCV UI
    for w in workers:
        frame = w.latest_drawn_frame()
        if frame is not None:
            imshow(f"YFIPS:{w.name}", frame)
    while not queue.empty():
        det = queue.get_nowait()
        with tracker_lock:
            x, y, yaw = tracker.update(det.rid, det.x, det.y, det.yaw, det.t)
        pub.send({"id": det.rid, "x": x, "y": y, "yaw": yaw,
                  "t": det.t, "camera": det.camera})
        ros_pub.send(...)
```

Prediction timer thread:

```
while not stop_event.is_set():
    sleep(prediction_interval)
    with tracker_lock:
        preds = list(emit_predictions(tracker, set(), now))
    for rid, x, y, yaw in preds:
        pub.send({"id": rid, "x": x, "y": y, "yaw": yaw,
                  "t": now, "predicted": True})
        ros_pub.send(...)
```

Notes:

- **Timestamp `t`** is stamped in the worker at detection time, so interleaved
  detections from different cameras remain correctly ordered even if the
  queue backs up briefly.
- **Fusion is implicit.** Both cameras' detections flow into the same tracker
  keyed on `rid`; Kalman `R` down-weights noisier updates, EMA blends.
- **Predictions on a timer** decouple them from any one camera's cadence.

## Error handling & lifecycle

### Startup

- Empty `cameras[]` → `SystemExit`.
- Duplicate `name` or `index` → `SystemExit`.
- Each camera opens independently. Per-camera `open_camera()` failure: log and
  skip, keep others. All-fail → `SystemExit`.
- Camera without `image_corners_px`: worker runs, window shows the "double-
  click 4 corners" hint, no queue pushes from that worker.
- Camera without intrinsics: undistort skipped for that camera only (matches
  today's behaviour).

### Runtime

- Worker hitting `CAPTURE_FAILURE_LIMIT` consecutive misses exits its thread
  and sets a per-worker `failed` flag. Orchestrator notices, logs, and
  continues with the remaining workers. If the **last** worker dies,
  orchestrator shuts down.
- `config.save()` is called from two places (click handler in each worker,
  calibration script). All writes from the running process are serialised
  through `config_lock`. `calibration.py` runs in its own process and does not
  contend — it's expected to be run while detection is stopped.
- Queue backpressure: bounded `queue.Queue(maxsize)` with drop-oldest on full,
  per-worker `dropped` counter logged at ~1 Hz.
- Tracker lock held only for one `update()` / `predict_only()` call.

### Shutdown

- ESC on any OpenCV window (captured by the single `waitKey` on the main
  thread) sets `stop_event`.
- Orchestrator joins all workers (1 s timeout each), joins the prediction
  timer thread, releases captures, destroys all windows, calls
  `ros_pub.shutdown()`.

### Edge cases

- **Undistort toggled after calibration** — gotcha is now per-camera. Clearing
  a single camera's corners on recalibration is scoped correctly.
- **Conflicting world-frame views** (one camera's corners clicked wrong):
  single id oscillates between clusters. Not prevented in code;
  `"camera"` field in UDP payload makes it diagnosable.
- **Robot only under one camera**: works (single-stream input to tracker).
- **Robot under two cameras simultaneously**: two detections within ~1 frame
  interval, tracker smooths across them. Yaw: EMA via unit vector handles
  wraparound, Kalman uses yaw measurement with `R = r_yaw`.
- **Stale homography after re-click**: `CalibClicker`'s 5th click still
  resets; per-camera.

## Testing

All tests live under `tests/` and follow the existing pure-logic pytest style.
No hardware, no real OpenCV windows, no real AprilTag.

### `test_config_migration.py` (new)

- Flat legacy config with `camera_matrix`, `dist_coeffs`, `image_corners_px`,
  `camera` at top level → after `load()`, produces
  `cameras=[{name:"cam0", ...}]` with values moved into index 0 and legacy
  keys removed from top level.
- Config that already has `cameras` is left untouched.
- Empty/new config → default `cameras=[{name:"cam0", index:0, …}]`.
- Duplicate `name` → `ValueError`.
- Duplicate `index` → `ValueError`.
- Empty `cameras=[]` → `ValueError`.

### `test_camera_worker.py` (new)

- Fake `VideoCapture` returning a canned gray frame + fake detector returning
  one detection → worker pushes a `Detection` with correct `camera` name,
  `t` set, and world coordinates matching the identity-`H` mapping.
- Worker constructed without homography (no `image_corners_px`) → nothing
  pushed onto the queue after N frames, but worker thread runs cleanly.
- Queue full → `dropped` counter increments; no exception raised.
- `stop()` joins the worker thread within timeout.

### `test_fusion.py` (new)

- Two synthetic detections for the same `rid` from `cam0` and `cam1`, close
  timestamps, slightly different `(x, y)`. Feed through a shared `EMATracker`
  in order; assert final state is within expected EMA bounds of the two
  inputs. Repeat for `KalmanTracker`.
- Prediction is emitted **once per tracked id** per timer tick regardless of
  how many cameras saw it that tick: feed two same-id detections, advance
  time, call `emit_predictions`, assert exactly one `(rid, x, y, yaw)` tuple
  for that id.

### `test_orchestrator.py` (new, light)

- Two mock workers push known detections into a real `queue.Queue`. Drive the
  orchestrator's queue-drain logic (extracted into a testable function) and
  assert: (a) `tracker.update` is called in order, (b) a mock `Publisher.send`
  receives payloads carrying `"camera": <name>`.

### Updated existing tests

- `test_trackers.py` — add coverage for `last_update_t(rid)` accessor.
- `test_calibration.py` — add coverage for `--camera <name>` flag writing
  into the correct `cameras[]` entry and clearing only that entry's
  `image_corners_px`.
- `test_publisher.py`, `test_apriltag_adapter.py`, `test_apriltag_options.py`,
  `test_camera_open.py`, `test_camera_settings.py`, `test_fps_meter.py`,
  `test_frame_grabber.py`, `test_image_detector.py`, `test_smoke.py` —
  unchanged.

### Not tested

- Real camera capture; real `cv2.imshow`; real AprilTag / ORB detection;
  real thread timing under load. Manual verification only (run it, see N
  windows, see the GUI).

## Migration path for users

1. `git pull` → `uv sync` (no new deps unless we add any).
2. First run loads the flat `config.json`, auto-migrates it into
   `cameras=[{name:"cam0", ...}]`, and re-saves.
3. To add a second camera: append an entry to `cameras[]` with a unique
   `name` and `index`, leave `camera_matrix` / `image_corners_px` null.
4. Run `uv run python -m yfips.calibration --camera cam1` with chessboard
   images in `images/cam1/`.
5. Start detection; a `YFIPS:cam1` window opens; double-click the 4 world
   corners in that window. Homography persists.
6. UDP consumers see detections tagged with `"camera": "cam1"`.

## Risks

- `cv2.imshow` / `waitKey` are single-threaded (main thread) on macOS. The
  design handles this: workers only draw into their frame buffer; the
  orchestrator calls `imshow` and `waitKey` on the main thread.
- Detector thread safety: each worker gets its own detector instance. No
  shared detector state.
- USB bandwidth with many cameras on one host: operational concern documented
  in the runbook, not in code.
- Prediction-timer interval tuning: too slow = stale predictions; too fast =
  spam. Default 50 ms is a reasonable starting point; exposed via
  `config.tracker.prediction_interval_s`.
