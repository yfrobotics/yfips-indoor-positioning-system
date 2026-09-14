"""Shared capture setup for local devices and RTSP streams."""

from typing import Any

import cv2

from yfips.config import camera_source


def create_capture(cam: dict[str, Any]) -> Any:
    source = camera_source(cam)
    cap = cv2.VideoCapture(source)
    if cap.isOpened() and isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
        cap.set(cv2.CAP_PROP_FPS, cam["fps"])
    return cap
