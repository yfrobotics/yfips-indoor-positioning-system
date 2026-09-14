"""Tests for the pure-logic helpers in yfips.calibration."""

import pytest

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
    with pytest.raises(ValueError, match="no camera named"):
        apply_intrinsics_to_camera(cfg, name="nope", mtx=[[1, 0, 0]],
                                   dist=[0.0])


def test_images_glob_for_camera_uses_camera_subdir():
    g = images_glob_for_camera("cam1")
    assert g.endswith("/images/cam1/calibration_*.jpg")
