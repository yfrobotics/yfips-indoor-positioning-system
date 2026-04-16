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
    for k in ("camera_matrix", "dist_coeffs", "image_corners_px", "camera"):
        assert k not in migrated
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
