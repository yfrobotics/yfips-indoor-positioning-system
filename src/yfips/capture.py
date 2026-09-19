"""Shared capture setup for local devices and RTSP streams."""

import os
from typing import Any

import cv2

from yfips.config import camera_source


# OpenCV exposes FFmpeg options process-wide. Set one stable default instead
# of swapping the environment while other camera threads may be opening.
# One decoder thread avoids frame-threading latency; the background
# FrameGrabber continuously drains capture and keeps only the newest frame.
RTSP_CAPTURE_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|threads;1"


def create_capture(cam: dict[str, Any]) -> Any:
    source = camera_source(cam)
    if isinstance(source, str):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_CAPTURE_OPTIONS)
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
        ])
    cap = cv2.VideoCapture(source)
    if cap.isOpened() and isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
        cap.set(cv2.CAP_PROP_FPS, cam["fps"])
    return cap
