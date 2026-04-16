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
