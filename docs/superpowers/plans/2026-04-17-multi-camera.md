# Multi-Camera Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend YF-IPS from a single webcam to N cameras with partially overlapping FOVs, all mapped into one 5×5 m world frame, producing a single fused `(id, x, y, yaw)` UDP stream.

**Architecture:** Single process, one `CameraWorker` thread per camera (own `VideoCapture`, `FrameGrabber`, undistort maps, detector, homography, OpenCV window). All workers push `Detection` items onto a shared `queue.Queue` drained by the orchestrator on the main thread, which updates a shared tracker (under a `threading.Lock`) and publishes to UDP + ROS. A separate timer thread emits predictions.

**Tech Stack:** Python 3.11+, OpenCV, numpy, `apriltag`, `rclpy` (optional), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-04-17-multi-camera-design.md`

---

## File Structure

- Create: `src/yfips/camera_worker.py` — `Detection` dataclass + `CameraWorker` thread class.
- Modify: `src/yfips/config.py` — add `cameras[]` schema, legacy migration, `validate_cameras`, `prediction_interval_s` default.
- Modify: `src/yfips/tracker.py` — add `last_measurement_t(rid)` accessor; keep `predict_only` returning `None`.
- Modify: `src/yfips/kalman_tracker.py` — track per-filter `last_measurement_t` separate from filter clock; add tracker-level accessor.
- Modify: `src/yfips/detection.py` — extend `emit_predictions` with `min_age_s`; replace single-camera loop with orchestrator (workers + queue drain + prediction timer).
- Modify: `src/yfips/calibration.py` — add `--camera <name>` flag; read from `images/<name>/calibration_*.jpg`; write into the matching `cameras[]` entry.
- Create: `tests/test_config_migration.py` — migration + validation tests.
- Create: `tests/test_camera_worker.py` — worker logic tests (fake capture, fake detector).
- Create: `tests/test_fusion.py` — multi-camera tracker fusion + single-prediction-per-id tests.
- Create: `tests/test_orchestrator.py` — drain-queue logic with mocks.
- Modify: `tests/test_trackers.py` — coverage for `last_measurement_t` + `emit_predictions` `min_age_s`.
- Modify: `tests/test_calibration.py` — coverage for the `--camera <name>` path.
- Modify: `CLAUDE.md` — architecture section + multi-camera runbook + gotchas.

---

## Task 1: Config schema — `cameras[]` + legacy migration + validation

**Files:**
- Modify: `src/yfips/config.py`
- Create: `tests/test_config_migration.py`

- [ ] **Step 1: Write the failing tests for migration + validation**

Create `tests/test_config_migration.py`:

```python
"""Tests for the cameras[] schema, legacy-flat migration, and validation."""

import pytest

from yfips.config import (
    DEFAULT_CAMERA_ENTRY,
    migrate_cameras,
    validate_cameras,
)


def test_migrate_flat_legacy_config_moves_fields_into_cameras_zero():
    legacy = {
        "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "dist_coeffs": [0.1, 0.0, 0.0, 0.0, 0.0],
        "image_corners_px": [[10, 20], [30, 40], [50, 60], [70, 80]],
        "camera": {"index": 2, "width": 1280, "height": 720, "fps": 30},
        "world_corners_m": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "mode": "apriltag",
    }
    migrated = migrate_cameras(legacy)
    assert migrated["cameras"] == [{
        "name": "cam0",
        "index": 2,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "dist_coeffs": [0.1, 0.0, 0.0, 0.0, 0.0],
        "image_corners_px": [[10, 20], [30, 40], [50, 60], [70, 80]],
    }]
    # Legacy keys must be removed from top level.
    for k in ("camera_matrix", "dist_coeffs", "image_corners_px", "camera"):
        assert k not in migrated
    # Unrelated top-level keys are preserved.
    assert migrated["world_corners_m"] == [[0, 0], [5, 0], [5, 5], [0, 5]]
    assert migrated["mode"] == "apriltag"


def test_migrate_already_multi_camera_is_idempotent():
    cfg = {
        "cameras": [{"name": "cam0", "index": 0, "width": 640, "height": 480,
                     "fps": 60, "camera_matrix": None, "dist_coeffs": None,
                     "image_corners_px": None}],
        "mode": "image",
    }
    out = migrate_cameras(dict(cfg))
    assert out == cfg


def test_migrate_empty_config_creates_default_cam0():
    out = migrate_cameras({})
    assert out["cameras"] == [DEFAULT_CAMERA_ENTRY]


def test_migrate_partial_legacy_fills_defaults_for_missing_fields():
    # Only camera_matrix present, no "camera" block.
    legacy = {"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    out = migrate_cameras(legacy)
    entry = out["cameras"][0]
    assert entry["name"] == "cam0"
    assert entry["index"] == 0
    assert entry["width"] == 640
    assert entry["height"] == 480
    assert entry["fps"] == 60
    assert entry["camera_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert entry["dist_coeffs"] is None
    assert entry["image_corners_px"] is None


def test_validate_cameras_requires_non_empty():
    with pytest.raises(ValueError, match="at least one camera"):
        validate_cameras([])


def test_validate_cameras_rejects_duplicate_name():
    with pytest.raises(ValueError, match="duplicate camera name"):
        validate_cameras([
            {"name": "cam0", "index": 0, "width": 640, "height": 480, "fps": 60,
             "camera_matrix": None, "dist_coeffs": None, "image_corners_px": None},
            {"name": "cam0", "index": 1, "width": 640, "height": 480, "fps": 60,
             "camera_matrix": None, "dist_coeffs": None, "image_corners_px": None},
        ])


def test_validate_cameras_rejects_duplicate_index():
    with pytest.raises(ValueError, match="duplicate camera index"):
        validate_cameras([
            {"name": "cam0", "index": 0, "width": 640, "height": 480, "fps": 60,
             "camera_matrix": None, "dist_coeffs": None, "image_corners_px": None},
            {"name": "cam1", "index": 0, "width": 640, "height": 480, "fps": 60,
             "camera_matrix": None, "dist_coeffs": None, "image_corners_px": None},
        ])


def test_validate_cameras_accepts_valid():
    validate_cameras([
        {"name": "cam0", "index": 0, "width": 640, "height": 480, "fps": 60,
         "camera_matrix": None, "dist_coeffs": None, "image_corners_px": None},
        {"name": "cam1", "index": 2, "width": 640, "height": 480, "fps": 60,
         "camera_matrix": None, "dist_coeffs": None, "image_corners_px": None},
    ])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_migration.py -v`
Expected: FAIL with `ImportError: cannot import name 'migrate_cameras'` (or similar).

- [ ] **Step 3: Implement migration, validation, and updated defaults**

Replace `src/yfips/config.py` with:

```python
from __future__ import annotations

import copy
import json
import os
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(_REPO_ROOT, "config.json")

DEFAULT_CAMERA_ENTRY: dict[str, Any] = {
    "name": "cam0",
    "index": 0,
    "width": 640,
    "height": 480,
    "fps": 60,
    "camera_matrix": None,
    "dist_coeffs": None,
    "image_corners_px": None,
}

DEFAULTS: dict[str, Any] = {
    "cameras": [dict(DEFAULT_CAMERA_ENTRY)],
    "world_corners_m": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
    "udp": {"host": "127.0.0.1", "port": 9999, "enabled": True},
    "mode": "apriltag",
    "references_dir": None,
    "image_mode": {"min_inliers": 15, "use_flann": False},
    "undistort": True,
    "tracker": {
        "enabled": True, "type": "ema", "alpha": 0.4, "timeout_s": 1.0,
        "q_accel": 1.0, "r_pos": 0.05, "r_yaw": 0.1,
        "prediction_interval_s": 0.05,
    },
    "ros": {"enabled": False, "topic": "/yfips/detections"},
}

_LEGACY_FLAT_KEYS = ("camera_matrix", "dist_coeffs", "image_corners_px", "camera")


def migrate_cameras(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with a single-camera flat layout converted to
    the cameras[] layout. Idempotent if cfg already has cameras[]."""
    out = copy.deepcopy(cfg)
    if "cameras" in out and out["cameras"] is not None:
        return out
    cam_block = out.pop("camera", None) or {}
    entry = dict(DEFAULT_CAMERA_ENTRY)
    entry["index"] = cam_block.get("index", entry["index"])
    entry["width"] = cam_block.get("width", entry["width"])
    entry["height"] = cam_block.get("height", entry["height"])
    entry["fps"] = cam_block.get("fps", entry["fps"])
    for k in ("camera_matrix", "dist_coeffs", "image_corners_px"):
        if k in out:
            entry[k] = out.pop(k)
    out["cameras"] = [entry]
    return out


def validate_cameras(cameras: list[dict[str, Any]]) -> None:
    if not cameras:
        raise ValueError("config.cameras must contain at least one camera")
    names, indices = set(), set()
    for cam in cameras:
        name = cam["name"]
        idx = cam["index"]
        if name in names:
            raise ValueError(f"duplicate camera name: {name!r}")
        if idx in indices:
            raise ValueError(f"duplicate camera index: {idx}")
        names.add(name)
        indices.add(idx)


def load() -> dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return copy.deepcopy(DEFAULTS)
    with open(CONFIG_PATH) as f:
        raw = json.load(f)
    merged = copy.deepcopy(DEFAULTS)
    merged.update(raw)
    migrated = migrate_cameras(merged)
    # Preserve the default tracker fields that aren't in raw
    tracker_defaults = dict(DEFAULTS["tracker"])
    tracker_defaults.update(migrated.get("tracker") or {})
    migrated["tracker"] = tracker_defaults
    if migrated != raw:
        save(migrated)
    return migrated


def save(cfg: dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def update(**kwargs: Any) -> dict[str, Any]:
    cfg = load()
    cfg.update(kwargs)
    save(cfg)
    return cfg
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_migration.py -v`
Expected: 7 PASSED.

- [ ] **Step 5: Run the full suite to catch regressions**

Run: `uv run pytest -x`
Expected: All tests pass. If `test_camera_settings.py` or other tests touch top-level `camera_matrix`/`camera` keys, they'll break — if so, update them to consume `cfg["cameras"][0]` instead. Commit any such fixes together with this task.

- [ ] **Step 6: Commit**

```bash
git add src/yfips/config.py tests/test_config_migration.py tests/
git commit -m "Introduce cameras[] config schema with legacy migration"
```

---

## Task 2: Tracker — `last_measurement_t(rid)` accessor

**Files:**
- Modify: `src/yfips/tracker.py`
- Modify: `src/yfips/kalman_tracker.py`
- Modify: `tests/test_trackers.py`

Rationale: the prediction-timer cooldown rule needs the timestamp of the last *measurement* (tracker `update`), not the filter's internal clock (which Kalman's `predict_only` advances).

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_trackers.py`:

```python
def test_ema_last_measurement_t_returns_update_time():
    ema = EMATracker()
    assert ema.last_measurement_t(1) is None
    ema.update(1, 0.0, 0.0, 0.0, t=3.25)
    assert ema.last_measurement_t(1) == 3.25
    ema.update(1, 0.1, 0.1, 0.1, t=4.50)
    assert ema.last_measurement_t(1) == 4.50


def test_kalman_last_measurement_t_ignores_predict_only():
    k = KalmanTracker(q_accel=0.05, r_pos=0.01, r_yaw=0.01, timeout_s=10.0)
    assert k.last_measurement_t(7) is None
    k.update(7, 0.0, 0.0, 0.0, t=1.0)
    assert k.last_measurement_t(7) == 1.0
    # predict_only advances the filter's internal clock but not the
    # measurement timestamp.
    k.predict_only(7, t=2.0)
    assert k.last_measurement_t(7) == 1.0
    k.update(7, 1.0, 0.0, 0.0, t=3.0)
    assert k.last_measurement_t(7) == 3.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trackers.py -v -k last_measurement_t`
Expected: FAIL with `AttributeError: … last_measurement_t`.

- [ ] **Step 3: Implement `last_measurement_t` on `EMATracker`**

Edit `src/yfips/tracker.py`. The existing state tuple already stores `t` at index 4 — expose it.

Append this method inside `class EMATracker`:

```python
    def last_measurement_t(self, rid: int) -> float | None:
        row = self.state.get(rid)
        return None if row is None else row[4]
```

- [ ] **Step 4: Implement `last_measurement_t` on `KalmanTracker`**

Edit `src/yfips/kalman_tracker.py`. Add a measurement timestamp field on `_Filter` and expose it on the tracker.

In `_Filter.__init__`, after `self.t = t` add:

```python
        self.last_measurement_t = t
```

In `_Filter.update`, at the end (after the existing Kalman update), add:

```python
        # t is advanced in KalmanTracker.update; record the measurement
        # timestamp from the outer scope via the caller.
```

Wait — `_Filter.update` doesn't receive `t`. Instead update it where `KalmanTracker.update` sets `f.t`. Change `KalmanTracker.update` to set `f.last_measurement_t = t` on both the new-filter branch (already set in `__init__`) and after the update branch:

Replace the body of `KalmanTracker.update` with:

```python
    def update(self, rid: int, x: float, y: float, yaw: float, t: float
               ) -> tuple[float, float, float]:
        f = self.filters.get(rid)
        if f is None or (t - f.t) > self.timeout_s:
            f = _Filter(x, y, yaw, t, self.q_accel, self.r_pos, self.r_yaw)
            self.filters[rid] = f
        else:
            f.predict(t)
            f.update(x, y, yaw)
            f.last_measurement_t = t
        return float(f.x[0]), float(f.x[1]), float(f.x[2])
```

Add to `class KalmanTracker`:

```python
    def last_measurement_t(self, rid: int) -> float | None:
        f = self.filters.get(rid)
        return None if f is None else f.last_measurement_t
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trackers.py -v`
Expected: All tests pass, including the two new `last_measurement_t` ones.

- [ ] **Step 6: Commit**

```bash
git add src/yfips/tracker.py src/yfips/kalman_tracker.py tests/test_trackers.py
git commit -m "Add last_measurement_t accessor on both trackers"
```

---

## Task 3: `emit_predictions` — `min_age_s` cooldown

**Files:**
- Modify: `src/yfips/detection.py` (only the `emit_predictions` function)
- Modify: `tests/test_trackers.py`

- [ ] **Step 1: Add failing tests for `min_age_s`**

Append to `tests/test_trackers.py`:

```python
def test_emit_predictions_skips_recently_updated_ids():
    k = KalmanTracker(q_accel=0.05, r_pos=0.01, r_yaw=0.01, timeout_s=10.0)
    # Warm the filter so predict_only would return a value.
    k.update(1, 0.0, 0.0, 0.0, t=0.0)
    k.update(1, 1.0, 0.0, 0.0, t=1.0)
    # At t=1.02, last measurement is 20ms old — below min_age_s.
    out = list(emit_predictions(k, detected_ids=set(), t=1.02, min_age_s=0.05))
    assert out == []
    # At t=1.10, measurement is 100ms old — above min_age_s.
    out = list(emit_predictions(k, detected_ids=set(), t=1.10, min_age_s=0.05))
    assert [row[0] for row in out] == [1]


def test_emit_predictions_min_age_defaults_to_zero():
    # Backwards-compatible call without min_age_s still works as before.
    k = KalmanTracker(q_accel=0.05, r_pos=0.01, r_yaw=0.01, timeout_s=10.0)
    k.update(1, 0.0, 0.0, 0.0, t=0.0)
    k.update(1, 1.0, 0.0, 0.0, t=1.0)
    out = list(emit_predictions(k, detected_ids=set(), t=1.5))
    assert [row[0] for row in out] == [1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trackers.py -v -k emit_predictions`
Expected: FAIL — `emit_predictions` doesn't accept `min_age_s`.

- [ ] **Step 3: Update `emit_predictions` signature**

In `src/yfips/detection.py`, replace the existing `emit_predictions` function with:

```python
def emit_predictions(tracker: Any, detected_ids: set[int], t: float,
                     min_age_s: float = 0.0
                     ) -> Iterator[tuple[int, float, float, float]]:
    """Yield (rid, x, y, yaw) for tracked ids absent this frame whose
    tracker can extrapolate. Trackers without a velocity model (EMA)
    silently produce nothing.

    min_age_s: skip ids whose last real measurement is newer than this
    threshold. Used by the prediction timer to avoid emitting a
    prediction immediately after a detection lands for the same id."""
    if tracker is None:
        return
    for rid in tracker.ids():
        if rid in detected_ids:
            continue
        if min_age_s > 0.0:
            last = tracker.last_measurement_t(rid)
            if last is not None and (t - last) < min_age_s:
                continue
        out = tracker.predict_only(rid, t)
        if out is None:
            continue
        yield (rid, *out)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trackers.py -v`
Expected: All green including the two new cases.

- [ ] **Step 5: Commit**

```bash
git add src/yfips/detection.py tests/test_trackers.py
git commit -m "Add min_age_s cooldown to emit_predictions"
```

---

## Task 4: Calibration — `--camera <name>` + per-camera images dir

**Files:**
- Modify: `src/yfips/calibration.py`
- Modify: `tests/test_calibration.py`

- [ ] **Step 1: Add failing tests for the new helper shape**

Append to `tests/test_calibration.py`:

```python
from yfips.calibration import apply_intrinsics_to_camera, images_glob_for_camera


def test_apply_intrinsics_to_camera_writes_into_matching_entry():
    cfg = {
        "cameras": [
            {"name": "cam0", "index": 0, "width": 640, "height": 480, "fps": 60,
             "camera_matrix": None, "dist_coeffs": None,
             "image_corners_px": None},
            {"name": "cam1", "index": 2, "width": 640, "height": 480, "fps": 60,
             "camera_matrix": None, "dist_coeffs": None,
             "image_corners_px": [[1, 2], [3, 4], [5, 6], [7, 8]]},
        ],
    }
    out = apply_intrinsics_to_camera(
        cfg, name="cam1",
        mtx=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        dist=[0.5, 0.0, 0.0, 0.0, 0.0],
    )
    assert out["cameras"][1]["camera_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    assert out["cameras"][1]["dist_coeffs"] == [0.5, 0.0, 0.0, 0.0, 0.0]
    # Only cam1's corners are cleared.
    assert out["cameras"][1]["image_corners_px"] is None
    # cam0 is untouched.
    assert out["cameras"][0]["camera_matrix"] is None


def test_apply_intrinsics_to_camera_raises_on_unknown_name():
    cfg = {"cameras": [{"name": "cam0", "index": 0, "width": 640, "height": 480,
                        "fps": 60, "camera_matrix": None, "dist_coeffs": None,
                        "image_corners_px": None}]}
    import pytest
    with pytest.raises(ValueError, match="no camera named"):
        apply_intrinsics_to_camera(cfg, name="nope", mtx=[[1, 0, 0]],
                                   dist=[0.0])


def test_images_glob_for_camera_uses_camera_subdir():
    g = images_glob_for_camera("cam1")
    assert g.endswith("/images/cam1/calibration_*.jpg")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_intrinsics_to_camera'`.

- [ ] **Step 3: Rewrite `calibration.py` to work against `cameras[]`**

Replace `src/yfips/calibration.py` with:

```python
"""Camera intrinsic calibration from a chessboard pattern.

Reads images/<camera_name>/calibration_*.jpg at the repo root, runs
OpenCV's chessboard corner finder + calibrateCamera, and writes the
intrinsics into config.json's matching cameras[] entry."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any

import cv2 as cv
import numpy as np

from yfips import config

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CHESSBOARD = (7, 6)  # inner corners (cols, rows)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def images_glob_for_camera(name: str) -> str:
    return os.path.join(_REPO_ROOT, "images", name, "calibration_*.jpg")


def _make_object_points() -> np.ndarray:
    cols, rows = CHESSBOARD
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return pts


def apply_intrinsics_to_camera(cfg: dict[str, Any], name: str,
                               mtx: Any, dist: Any) -> dict[str, Any]:
    """Update cfg in place: write intrinsics into cameras[<name>] and clear
    only that camera's image_corners_px. The previously saved homography
    no longer matches the rectified image, so the user must re-click."""
    for entry in cfg["cameras"]:
        if entry["name"] == name:
            entry["camera_matrix"] = mtx
            entry["dist_coeffs"] = dist
            if entry.get("image_corners_px") is not None:
                print(f"[calibration] WARNING: image_corners_px for {name} "
                      "cleared — re-click 4 world corners after restart")
                entry["image_corners_px"] = None
            return cfg
    raise ValueError(f"no camera named {name!r} in config.cameras")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=None,
                        help="name of the camera to calibrate "
                             "(default: first camera in config.cameras)")
    args = parser.parse_args()

    cfg = config.load()
    cam_name = args.camera or cfg["cameras"][0]["name"]
    if not any(c["name"] == cam_name for c in cfg["cameras"]):
        raise SystemExit(
            f"[calibration] no camera named {cam_name!r} in config.cameras — "
            f"available: {[c['name'] for c in cfg['cameras']]}"
        )

    glob_path = images_glob_for_camera(cam_name)
    images = glob.glob(glob_path)
    if not images:
        raise SystemExit(f"No calibration images found at {glob_path}")

    objp = _make_object_points()
    objpoints, imgpoints = [], []
    gray = None
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(500)

    ret, mtx, dist, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print(f"[calibration] {cam_name}: RMS reprojection error: {ret}")
    print("Camera matrix:\n", mtx)
    print("Distortion coeffs:", dist.ravel())

    apply_intrinsics_to_camera(cfg, name=cam_name,
                               mtx=mtx.tolist(), dist=dist.ravel().tolist())
    config.save(cfg)
    print(f"Saved intrinsics for {cam_name} to {config.CONFIG_PATH}")

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Delete the now-obsolete single-camera tests and add migration coverage**

In `tests/test_calibration.py`, delete the four `test_apply_intrinsics_*` tests that reference the old flat shape (`apply_intrinsics` is gone). The three new tests from Step 1 replace them.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: 3 PASSED (the three tests from Step 1).

- [ ] **Step 6: Commit**

```bash
git add src/yfips/calibration.py tests/test_calibration.py
git commit -m "Scope calibration to a named camera entry"
```

---

## Task 5: `camera_worker.py` — `Detection` dataclass + `CameraWorker`

**Files:**
- Create: `src/yfips/camera_worker.py`
- Create: `tests/test_camera_worker.py`

- [ ] **Step 1: Add failing tests**

Create `tests/test_camera_worker.py`:

```python
"""Logic tests for CameraWorker. Uses a fake VideoCapture and fake
detector — no real OpenCV windows, no real cameras."""

import queue
import threading
import time

import numpy as np

from yfips.camera_worker import CameraWorker, Detection


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in. Always returns the same frame."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.released = False

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame

    def release(self) -> None:
        self.released = True

    def set(self, *args: object, **kwargs: object) -> bool:
        return True

    def isOpened(self) -> bool:
        return True


class FakeDetector:
    """Emits one detection per `detect()` call."""

    def __init__(self, rid: int = 7) -> None:
        self.rid = rid
        self.calls = 0

    def detect(self, gray: np.ndarray) -> list[dict]:
        self.calls += 1
        # Tag center at (100, 100), forward point at (110, 100) → along +x.
        return [{
            "id": self.rid,
            "center": (100.0, 100.0),
            "forward": (110.0, 100.0),
            "corners": np.array([[90, 90], [110, 90], [110, 110], [90, 110]],
                                dtype=np.float32),
        }]


def _make_worker(out_queue: queue.Queue, stop_event: threading.Event,
                 image_corners_px=None, cam_name: str = "cam0",
                 detector=None) -> CameraWorker:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = FakeCapture(frame)
    cam_cfg = {
        "name": cam_name,
        "index": 0,
        "width": 640,
        "height": 480,
        "fps": 60,
        "camera_matrix": None,
        "dist_coeffs": None,
        "image_corners_px": image_corners_px,
    }
    return CameraWorker(
        cam_cfg=cam_cfg,
        world_corners_m=[[0, 0], [640, 0], [640, 480], [0, 480]],
        detector=detector or FakeDetector(),
        out_queue=out_queue,
        stop_event=stop_event,
        config_lock=threading.Lock(),
        capture=cap,
    )


def test_worker_with_homography_emits_detections_with_camera_name():
    # World corners == image corners, so H is identity in pixel coords.
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    image_corners = [[0, 0], [640, 0], [640, 480], [0, 480]]
    worker = _make_worker(q, stop, image_corners_px=image_corners,
                          cam_name="camX")
    worker.start()
    try:
        det = q.get(timeout=1.0)
    finally:
        stop.set()
        worker.stop()
    assert isinstance(det, Detection)
    assert det.rid == 7
    assert det.camera == "camX"
    assert det.x == 100.0 and det.y == 100.0
    assert abs(det.yaw) < 1e-6  # forward along +x → yaw 0


def test_worker_without_homography_pushes_nothing():
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    worker = _make_worker(q, stop, image_corners_px=None)
    worker.start()
    time.sleep(0.1)  # give the worker a few frames
    stop.set()
    worker.stop()
    assert q.empty()


def test_worker_drops_oldest_when_queue_full():
    q: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()
    image_corners = [[0, 0], [640, 0], [640, 480], [0, 480]]
    worker = _make_worker(q, stop, image_corners_px=image_corners)
    worker.start()
    time.sleep(0.2)  # queue fills, drops accumulate
    stop.set()
    worker.stop()
    assert worker.dropped > 0


def test_worker_stop_joins_within_timeout():
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    worker = _make_worker(q, stop)
    worker.start()
    stop.set()
    worker.stop()
    assert not worker.thread.is_alive()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_camera_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'yfips.camera_worker'`.

- [ ] **Step 3: Implement `CameraWorker`**

Create `src/yfips/camera_worker.py`:

```python
"""Per-camera capture + detection thread.

Owns a VideoCapture, a FrameGrabber, a detector, undistort maps, and an
OpenCV preview window for its camera. Emits Detection items onto a shared
queue. Never blocks on the queue: when full, drops the oldest item and
increments `self.dropped`.
"""

from __future__ import annotations

import dataclasses
import math
import queue
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from yfips import config
from yfips.detection import (  # existing pure helpers
    CAPTURE_FAILURE_LIMIT,
    FrameGrabber,
    compute_homography,
    image_to_world,
    yaw_from_forward,
)


@dataclasses.dataclass
class Detection:
    rid: int
    x: float
    y: float
    yaw: float
    t: float
    camera: str


def _build_undistort_maps(cam_cfg: dict[str, Any]) -> Any:
    K = cam_cfg.get("camera_matrix")
    D = cam_cfg.get("dist_coeffs")
    if not K or not D:
        return None
    K = np.array(K, dtype=np.float32)
    D = np.array(D, dtype=np.float32)
    size = (cam_cfg["width"], cam_cfg["height"])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=0.0, newImgSize=size)
    return cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_16SC2)


class CameraWorker:
    def __init__(
        self,
        cam_cfg: dict[str, Any],
        world_corners_m: list[list[float]],
        detector: Any,
        out_queue: queue.Queue,
        stop_event: threading.Event,
        config_lock: threading.Lock,
        capture: Any | None = None,
        undistort_enabled: bool = True,
    ) -> None:
        self.cam_cfg = cam_cfg
        self.world_corners_m = world_corners_m
        self.detector = detector
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.config_lock = config_lock
        self.name = cam_cfg["name"]
        self.dropped = 0
        self.failed = False
        self._latest_drawn: np.ndarray | None = None
        self._latest_lock = threading.Lock()

        if capture is None:
            capture = cv2.VideoCapture(cam_cfg["index"])
            if not capture.isOpened():
                self.failed = True
            else:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
                capture.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
        self.capture = capture

        self.undistort_maps = (
            _build_undistort_maps(cam_cfg) if undistort_enabled else None
        )

        self.clicker = _PerCameraClicker(self)

        self.grabber = FrameGrabber(capture) if not self.failed else None
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"cam-{self.name}")

    def start(self) -> "CameraWorker":
        if self.failed:
            return self
        self.grabber.start()
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        if self.grabber is not None:
            self.grabber.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def latest_drawn_frame(self) -> np.ndarray | None:
        with self._latest_lock:
            return None if self._latest_drawn is None else self._latest_drawn.copy()

    def _emit(self, det: Detection) -> None:
        try:
            self.out_queue.put_nowait(det)
        except queue.Full:
            try:
                self.out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.out_queue.put_nowait(det)
            except queue.Full:
                pass
            self.dropped += 1

    def _run(self) -> None:
        last_counter = 0
        while not self.stop_event.is_set():
            frame, counter, misses = self.grabber.get_new(last_counter,
                                                         timeout=0.5)
            if misses >= CAPTURE_FAILURE_LIMIT:
                print(f"[{self.name}] camera returned no frame for "
                      f"{CAPTURE_FAILURE_LIMIT} consecutive reads — stopping")
                self.failed = True
                return
            if frame is None:
                continue
            last_counter = counter
            now = time.time()

            if self.undistort_maps is not None:
                frame = cv2.remap(frame, self.undistort_maps[0],
                                  self.undistort_maps[1], cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = self.detector.detect(gray)

            H = self.clicker.H
            for det in detections:
                cx, cy = det["center"]
                for cn in det["corners"]:
                    cv2.circle(frame, (int(cn[0]), int(cn[1])), 3, (0, 0, 255), -1)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                fx_, fy_ = det["forward"]
                cv2.arrowedLine(frame, (int(cx), int(cy)), (int(fx_), int(fy_)),
                                (0, 255, 255), 1, tipLength=0.3)

                if H is not None:
                    x_w, y_w = image_to_world(H, det["center"])
                    yaw = yaw_from_forward(H, det["center"], det["forward"])
                    if math.isfinite(x_w) and math.isfinite(y_w) and math.isfinite(yaw):
                        self._emit(Detection(rid=int(det["id"]), x=x_w, y=y_w,
                                             yaw=yaw, t=now, camera=self.name))
                    label = f"id={det['id']} x={x_w:.2f} y={y_w:.2f}"
                else:
                    label = f"id={det['id']}"
                cv2.putText(frame, label, (int(cx) + 6, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            if H is None:
                cv2.putText(frame, "double-click 4 corners in world_corners_m order",
                            (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255))

            with self._latest_lock:
                self._latest_drawn = frame


class _PerCameraClicker:
    """CalibClicker variant scoped to this worker's cameras[] entry.
    Persists image_corners_px under the shared config_lock."""

    def __init__(self, worker: "CameraWorker") -> None:
        self.worker = worker
        existing = worker.cam_cfg.get("image_corners_px")
        self.points: list[list[int]] = list(existing) if existing else []
        self.H: np.ndarray | None = None
        if len(self.points) == 4:
            self.H = compute_homography(self.points, worker.world_corners_m)

    def __call__(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDBLCLK:
            return
        if len(self.points) >= 4:
            self.points = []
            self.H = None
        self.points.append([x, y])
        print(f"[{self.worker.name}] calib point "
              f"{len(self.points)}/4: ({x},{y})")
        if len(self.points) == 4:
            self.H = compute_homography(self.points, self.worker.world_corners_m)
            with self.worker.config_lock:
                cfg = config.load()
                for entry in cfg["cameras"]:
                    if entry["name"] == self.worker.name:
                        entry["image_corners_px"] = self.points
                        break
                config.save(cfg)
            self.worker.cam_cfg["image_corners_px"] = self.points
            print(f"[{self.worker.name}] homography computed and saved")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_camera_worker.py -v`
Expected: 4 PASSED. If queue-drop test is flaky (depends on timing), raise the sleep to 0.3 s. If the homography test yields non-zero `yaw` due to y-axis flip in image coordinates, verify: `atan2(fw.y - cw.y, fw.x - cw.x)` with `fw=(110,100)`, `cw=(100,100)` gives `atan2(0, 10) = 0`. Correct.

- [ ] **Step 5: Commit**

```bash
git add src/yfips/camera_worker.py tests/test_camera_worker.py
git commit -m "Add CameraWorker thread and Detection dataclass"
```

---

## Task 6: Orchestrator refactor — multi-camera `detection.py`

**Files:**
- Modify: `src/yfips/detection.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Add failing tests for the drain-queue helper**

Create `tests/test_orchestrator.py`:

```python
"""Tests for the orchestrator's pure queue-drain logic. No threads,
no cameras — just the state-transition function."""

import queue
import threading

from yfips.camera_worker import Detection
from yfips.detection import drain_detections
from yfips.tracker import EMATracker


class MockPublisher:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)


def test_drain_detections_updates_tracker_and_publishes_with_camera_name():
    q: queue.Queue = queue.Queue()
    q.put(Detection(rid=1, x=1.0, y=2.0, yaw=0.0, t=1.0, camera="cam0"))
    q.put(Detection(rid=1, x=1.1, y=2.1, yaw=0.0, t=1.05, camera="cam1"))
    q.put(Detection(rid=2, x=3.0, y=4.0, yaw=1.57, t=1.10, camera="cam0"))
    tracker = EMATracker(alpha=0.5, timeout_s=10.0)
    pub = MockPublisher()
    ros_pub = MockPublisher()

    drain_detections(q, tracker, threading.Lock(), pub, ros_pub)

    # Three payloads, each tagged with its source camera.
    assert [p["camera"] for p in pub.sent] == ["cam0", "cam1", "cam0"]
    assert [p["id"] for p in pub.sent] == [1, 1, 2]
    # Same id from two cameras produced two tracker updates.
    assert sorted(tracker.ids()) == [1, 2]
    # ROS got the same traffic.
    assert len(ros_pub.sent) == 3


def test_drain_detections_handles_empty_queue():
    q: queue.Queue = queue.Queue()
    tracker = EMATracker()
    pub = MockPublisher()
    drain_detections(q, tracker, threading.Lock(), pub, pub)
    assert pub.sent == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ImportError: cannot import name 'drain_detections'`.

- [ ] **Step 3: Extract `drain_detections` and refactor `main()`**

In `src/yfips/detection.py`:

**(a)** Remove the old single-camera loop body inside `main()` (everything from `cap = open_camera(cam)` down to the `finally:` block).

**(b)** Add these helpers before `main()`:

```python
def drain_detections(
    q: "queue.Queue[Any]",
    tracker: Any,
    tracker_lock: threading.Lock,
    pub: Any,
    ros_pub: Any,
) -> set[int]:
    """Drain all available Detection items from q, update the tracker
    under tracker_lock, and publish. Returns the set of ids seen this
    drain (used by the prediction timer's skip logic, though the timer
    itself relies on last_measurement_t)."""
    seen: set[int] = set()
    while True:
        try:
            det = q.get_nowait()
        except queue.Empty:
            return seen
        with tracker_lock:
            if tracker is not None:
                x, y, yaw = tracker.update(det.rid, det.x, det.y, det.yaw, det.t)
            else:
                x, y, yaw = det.x, det.y, det.yaw
        payload = {"id": det.rid, "x": x, "y": y, "yaw": yaw,
                   "t": det.t, "camera": det.camera}
        pub.send(payload)
        ros_pub.send(payload)
        seen.add(det.rid)


def _prediction_timer_loop(
    tracker: Any,
    tracker_lock: threading.Lock,
    pub: Any,
    ros_pub: Any,
    stop_event: threading.Event,
    interval_s: float,
) -> None:
    while not stop_event.wait(interval_s):
        now = time.time()
        with tracker_lock:
            preds = list(emit_predictions(
                tracker, detected_ids=set(), t=now, min_age_s=interval_s,
            ))
        for rid, x, y, yaw in preds:
            payload = {"id": rid, "x": x, "y": y, "yaw": yaw,
                       "t": now, "predicted": True}
            pub.send(payload)
            ros_pub.send(payload)
```

Add `import queue` and `import threading` at the top of the file if not already present.

**(c)** Replace `main()` with the orchestrator:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["apriltag", "image"], default=None,
                        help="detection mode; overrides config.mode")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="camera index; single-camera configs only")
    parser.add_argument("--width", type=int, default=None,
                        help="frame width; single-camera configs only")
    parser.add_argument("--height", type=int, default=None,
                        help="frame height; single-camera configs only")
    parser.add_argument("--fps", type=int, default=None,
                        help="capture fps; single-camera configs only")
    args = parser.parse_args()

    cfg = config.load()
    try:
        config.validate_cameras(cfg["cameras"])
    except ValueError as e:
        raise SystemExit(f"[yfips] invalid cameras config: {e}")

    if len(cfg["cameras"]) > 1 and any(v is not None for v in (
            args.camera_index, args.width, args.height, args.fps)):
        raise SystemExit(
            "[yfips] --camera-index/--width/--height/--fps are only "
            "supported with a single camera in config.cameras[]; "
            "edit config.json for multi-camera setups"
        )

    if len(cfg["cameras"]) == 1:
        cam = cfg["cameras"][0]
        if args.camera_index is not None:
            cam["index"] = args.camera_index
        if args.width is not None: cam["width"] = args.width
        if args.height is not None: cam["height"] = args.height
        if args.fps is not None: cam["fps"] = args.fps

    mode = args.mode or cfg.get("mode", "apriltag")
    print(f"[yfips] mode={mode} cameras={[c['name'] for c in cfg['cameras']]}")

    # Lazy import to avoid a circular dependency with camera_worker.py,
    # which imports helpers from this module at import time.
    from yfips.camera_worker import CameraWorker

    pub = Publisher(cfg["udp"])
    ros_pub = RosPublisher(cfg.get("ros", {}))
    tracker_cfg = cfg.get("tracker", {})
    if not tracker_cfg.get("enabled", True):
        tracker = None
    elif tracker_cfg.get("type", "ema") == "kalman":
        tracker = KalmanTracker(
            q_accel=float(tracker_cfg.get("q_accel", 1.0)),
            r_pos=float(tracker_cfg.get("r_pos", 0.05)),
            r_yaw=float(tracker_cfg.get("r_yaw", 0.1)),
            timeout_s=float(tracker_cfg.get("timeout_s", 1.0)),
        )
    else:
        tracker = EMATracker(
            alpha=float(tracker_cfg.get("alpha", 0.4)),
            timeout_s=float(tracker_cfg.get("timeout_s", 1.0)),
        )
    prediction_interval_s = float(tracker_cfg.get("prediction_interval_s", 0.05))

    stop_event = threading.Event()
    tracker_lock = threading.Lock()
    config_lock = threading.Lock()
    det_queue: queue.Queue = queue.Queue(maxsize=1024)
    undistort_enabled = bool(cfg.get("undistort", True))

    workers = []
    for cam_cfg in cfg["cameras"]:
        detector = build_detector(mode, cfg)
        w = CameraWorker(
            cam_cfg=cam_cfg,
            world_corners_m=cfg["world_corners_m"],
            detector=detector,
            out_queue=det_queue,
            stop_event=stop_event,
            config_lock=config_lock,
            undistort_enabled=undistort_enabled,
        )
        if w.failed:
            print(f"[yfips] {cam_cfg['name']} failed to open — skipping")
            continue
        workers.append(w)

    if not workers:
        raise SystemExit("[yfips] no cameras could be opened")

    for w in workers:
        w.start()
        cv2.namedWindow(f"YFIPS:{w.name}")
        cv2.setMouseCallback(f"YFIPS:{w.name}", w.clicker)

    pred_thread = threading.Thread(
        target=_prediction_timer_loop,
        args=(tracker, tracker_lock, pub, ros_pub, stop_event,
              prediction_interval_s),
        daemon=True, name="prediction-timer",
    )
    if tracker is not None:
        pred_thread.start()

    try:
        while not stop_event.is_set():
            if all(w.failed for w in workers):
                print("[yfips] all cameras failed — exiting")
                break
            for w in workers:
                frame = w.latest_drawn_frame()
                if frame is not None:
                    cv2.imshow(f"YFIPS:{w.name}", frame)
            drain_detections(det_queue, tracker, tracker_lock, pub, ros_pub)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        stop_event.set()
        for w in workers:
            w.stop()
        if tracker is not None:
            pred_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        ros_pub.shutdown()
```

**(d)** Add `import queue` at the top of `detection.py` if absent. Do **not** add `from yfips.camera_worker import CameraWorker` at module top — that would create a circular import because `camera_worker.py` imports helpers from `detection.py`. Instead, import `CameraWorker` lazily inside `main()` (shown in the replacement `main()` body above). `drain_detections` only reads `Detection` attributes via duck-typing, so no import of `Detection` is needed in `detection.py`.

Keep the old `FrameGrabber`, `FpsMeter`, `open_camera`, `Publisher`, `compute_homography`, `image_to_world`, `CalibClicker`, `AprilTagAdapter`, `apriltag_options_kwargs`, `_APRILTAG_KEY_MAP`, `build_detector`, `yaw_from_forward`, `CAPTURE_FAILURE_LIMIT`, and `emit_predictions` exactly as they are (they're imported by `camera_worker.py` and tests).

**(e)** Delete the now-unused `camera_settings()` function from `detection.py` (it read the legacy flat `cfg["camera"]` key which no longer exists). Delete `tests/test_camera_settings.py` in the same commit — all four tests in that file exercise the deleted function.

- [ ] **Step 4: Run the orchestrator tests and the full suite**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: 2 PASSED.

Run: `uv run pytest -x`
Expected: All green. Any older test that referenced `main()` or single-camera orchestration is updated or removed in the same commit.

- [ ] **Step 5: Lint**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/yfips/detection.py tests/test_orchestrator.py
git commit -m "Refactor detection.py into multi-camera orchestrator"
```

---

## Task 7: Fusion sanity test across cameras

**Files:**
- Create: `tests/test_fusion.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fusion.py`:

```python
"""Cross-camera fusion sanity: two detections for the same id from two
cameras, fed through a shared tracker, produce a sensibly blended state.
Also: the prediction timer emits one payload per id per tick regardless
of how many cameras saw it."""

from yfips.detection import emit_predictions
from yfips.kalman_tracker import KalmanTracker
from yfips.tracker import EMATracker


def test_ema_fuses_two_camera_updates_on_same_id():
    ema = EMATracker(alpha=0.5, timeout_s=10.0)
    # cam0 reports (1.0, 2.0); cam1 reports (1.4, 2.2) 10ms later.
    ema.update(7, 1.0, 2.0, 0.0, t=0.0)
    x, y, yaw = ema.update(7, 1.4, 2.2, 0.0, t=0.01)
    # Expected EMA with alpha=0.5: prev = (1.0, 2.0); new = 0.5*new + 0.5*prev
    assert abs(x - 1.2) < 1e-6
    assert abs(y - 2.1) < 1e-6
    assert abs(yaw) < 1e-6


def test_kalman_fuses_two_camera_updates_on_same_id():
    k = KalmanTracker(q_accel=0.05, r_pos=0.05, r_yaw=0.05, timeout_s=10.0)
    k.update(7, 1.0, 2.0, 0.0, t=0.0)
    x, y, _ = k.update(7, 1.4, 2.2, 0.0, t=0.01)
    # Kalman should land between the two measurements.
    assert 1.0 <= x <= 1.4
    assert 2.0 <= y <= 2.2


def test_emit_predictions_once_per_id_after_multiple_camera_updates():
    k = KalmanTracker(q_accel=0.05, r_pos=0.01, r_yaw=0.01, timeout_s=10.0)
    # Same id seen by two cameras at t≈0 and t≈0.01.
    k.update(7, 1.0, 2.0, 0.0, t=0.0)
    k.update(7, 1.1, 2.1, 0.0, t=0.01)
    # After the cooldown, prediction should fire exactly once for id=7.
    out = list(emit_predictions(k, detected_ids=set(), t=0.2, min_age_s=0.05))
    assert [row[0] for row in out] == [7]
```

- [ ] **Step 2: Run the tests to verify they pass immediately**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: 3 PASSED (no new production code needed — Tasks 2 and 3 already laid the foundation).

If the Kalman bounded-fusion test fails because the filter's first-measurement step initialises to the raw input, verify behaviour: `update` call 1 creates `_Filter` with `x=1.0`. Call 2 does `predict` then `update`; the Kalman gain at this stage is close to 1 (high P), so the state lands near the second measurement, which is still within `[1.0, 1.4]`. Test should pass.

- [ ] **Step 3: Run the full suite and lint**

Run: `uv run pytest`
Expected: All green.

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_fusion.py
git commit -m "Add cross-camera tracker fusion tests"
```

---

## Task 8: Docs — CLAUDE.md multi-camera runbook

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Setup & Run section**

In `CLAUDE.md`, replace the calibration example line:

```
uv run python -m yfips.calibration                    # chessboard calibration → config.json
```

with:

```
uv run python -m yfips.calibration --camera cam0      # calibrate a camera (default: first in cameras[])
```

- [ ] **Step 2: Update the Architecture section**

Replace the description of `detection.py` and related modules with:

```
- `src/yfips/config.py` — load/save `config.json`. Holds a `cameras[]` array
  (each entry: `name`, `index`, `width`, `height`, `fps`, `camera_matrix`,
  `dist_coeffs`, `image_corners_px`), plus global `world_corners_m`, UDP,
  tracker, mode, references dir. Legacy flat single-camera configs are
  auto-migrated into `cameras=[{name:"cam0",...}]` on first load.
- `src/yfips/calibration.py` — takes `--camera <name>`; reads
  `images/<name>/calibration_*.jpg` and writes intrinsics into the
  matching `cameras[]` entry. Only that camera's `image_corners_px` is
  cleared.
- `src/yfips/camera_worker.py` — `CameraWorker` thread per camera: owns
  its own `VideoCapture`, `FrameGrabber`, undistort maps, detector,
  homography (`CalibClicker`), and preview window `YFIPS:<name>`. Emits
  `Detection(rid, x, y, yaw, t, camera)` items onto a shared queue.
- `src/yfips/detection.py` — orchestrator. Loads + validates config,
  spawns one `CameraWorker` per `cameras[]` entry, drains the shared
  detection queue on the main thread, updates a shared tracker under a
  lock, and publishes `{id, x, y, yaw, t, camera}` to UDP + ROS. A
  prediction-timer thread emits `{predicted:true}` payloads at
  `tracker.prediction_interval_s` (default 50 ms), skipping ids whose
  last measurement is newer than that interval.
- `src/yfips/tracker.py`, `kalman_tracker.py` — per-id smoothers;
  tracker-level fusion happens implicitly because same-id detections
  from different cameras land in the same `tracker.update()` call keyed
  on `rid`. Both expose `last_measurement_t(rid)` for the prediction
  cooldown.
- `src/yfips/gui.py` — unchanged; listens to UDP. Payloads now carry a
  `"camera"` field for detections (omitted on predicted packets).
```

- [ ] **Step 3: Add a "Multi-camera runbook" subsection**

Add to `CLAUDE.md` after the Architecture section:

```
## Multi-camera runbook

To add a second camera:

1. Append an entry to `config.json`'s `cameras[]`:
   ```json
   {"name": "cam1", "index": 2, "width": 640, "height": 480, "fps": 60,
    "camera_matrix": null, "dist_coeffs": null, "image_corners_px": null}
   ```
2. Drop chessboard images into `images/cam1/calibration_*.jpg`.
3. `uv run python -m yfips.calibration --camera cam1`.
4. Start detection. A `YFIPS:cam1` window opens; double-click 4 world
   corners in that window.
5. Verify via the UDP listener: payloads should appear with
   `"camera": "cam1"`.

CLI flags `--camera-index/--width/--height/--fps` are only honoured
with a single camera in `cameras[]`; with N>1, edit `config.json` instead.
```

- [ ] **Step 4: Update Gotchas**

Add the following bullets to the existing Gotchas section:

```
- Duplicate `name` or `index` in `cameras[]` is a config error; startup
  fails with a clear message.
- `cv2.imshow` / `waitKey` must run on the main thread. Workers only
  draw into a frame buffer; the orchestrator calls `imshow` and
  `waitKey` for every worker's latest frame once per main-loop tick.
- Each camera's homography is cleared **only for that camera** when its
  intrinsics are re-calibrated. Flipping the global `undistort` flag
  still invalidates clicks across all cameras — re-click each.
- Per-camera detector instances are a requirement, not a convenience:
  the AprilTag and ORB detectors are not safe to share across threads.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "Document multi-camera architecture and runbook"
```

---

## Final verification

- [ ] **Run the full suite one more time**

Run: `uv run pytest -v`
Expected: All tests pass.

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Manual smoke test** (not scripted — document result in the PR)

1. With existing single-camera `config.json` (unmodified flat layout), run `uv run python -m yfips.detection` — it should auto-migrate the config, open one window `YFIPS:cam0`, detect tags, and publish UDP with `"camera": "cam0"`.
2. Add a second camera entry, restart detection. Two windows open. Double-click 4 world corners in `YFIPS:cam1`. Verify UDP payloads carry both `"camera": "cam0"` and `"camera": "cam1"`.
3. Cover `cam1`'s FOV with a hand while a tag is visible to `cam0` only. The UDP stream continues for that tag with `"camera": "cam0"`. Uncover; detections from `cam1` join back in.
4. ESC on either window shuts the whole thing down cleanly.

- [ ] **Open PR**

```bash
git push -u origin feature/multi-camera
gh pr create --title "Multi-camera integration (N cameras, one world frame)" --body "$(cat <<'EOF'
## Summary
- Extends YF-IPS to N cameras with partial overlap, all in one 5×5 m world frame
- Tracker-level fusion on shared id; per-camera homography + undistort
- `cameras[]` config schema with backward-compatible auto-migration from the flat layout
- UDP payload gains `"camera"` field; prediction timer decoupled from any single camera's cadence

Spec: `docs/superpowers/specs/2026-04-17-multi-camera-design.md`
Plan: `docs/superpowers/plans/2026-04-17-multi-camera.md`

## Test plan
- [x] `uv run pytest -v` (new: `test_config_migration`, `test_camera_worker`, `test_fusion`, `test_orchestrator`; updated: `test_trackers`, `test_calibration`)
- [x] `uv run ruff check .`
- [ ] Manual smoke test with existing single-camera config (auto-migrates)
- [ ] Manual smoke test with two USB cameras (independent detection, fused tracker, ESC shuts down)
EOF
)"
```
