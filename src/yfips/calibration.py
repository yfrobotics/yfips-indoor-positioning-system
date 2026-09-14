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
