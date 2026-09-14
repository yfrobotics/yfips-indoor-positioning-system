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


def test_validate_mixed_sources_with_shared_unused_index():
    validate_cameras([
        {"name": "usb", "index": 0},
        {"name": "ip1", "index": 0, "rtsp_url": "rtsp://192.0.2.1/live"},
        {"name": "ip2", "rtsp_url": "rtsp://192.0.2.2/live"},
    ])


@pytest.mark.parametrize("url", ["", "http://camera/live", "rtsp:///live", 12])
def test_validate_rejects_invalid_rtsp_url(url):
    with pytest.raises(ValueError, match="rtsp_url"):
        validate_cameras([{"name": "ip", "rtsp_url": url}])


def test_duplicate_rtsp_source_does_not_expose_credentials():
    url = "rtsp://user:secret@192.0.2.1/live"
    with pytest.raises(ValueError, match="duplicate camera RTSP URL") as error:
        validate_cameras([
            {"name": "ip1", "rtsp_url": url}, {"name": "ip2", "rtsp_url": url},
        ])
    assert "secret" not in str(error.value)


def test_load_migrates_legacy_rtsp_camera(tmp_path, monkeypatch):
    import json

    from yfips import config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera": {"rtsp_url": "rtsp://192.0.2.1/live"}}))
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    loaded = config.load()
    assert loaded["cameras"][0]["rtsp_url"] == "rtsp://192.0.2.1/live"
    assert config.load() == loaded
