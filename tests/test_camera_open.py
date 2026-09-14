"""Tests for the camera-open error path."""

import pytest

from yfips import detection


class _FakeCap:
    def __init__(self, opened=True):
        self._opened = opened
        self.props = {}

    def release(self):
        self.released = True

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.props[prop] = value
        return True


def test_open_camera_raises_when_not_opened(monkeypatch):
    monkeypatch.setattr(detection.cv2, "VideoCapture", lambda idx: _FakeCap(opened=False))
    with pytest.raises(SystemExit, match="index 3"):
        detection.open_camera({"index": 3, "width": 640, "height": 480, "fps": 60})


def test_open_camera_returns_cap_when_opened(monkeypatch):
    fake = _FakeCap(opened=True)
    monkeypatch.setattr(detection.cv2, "VideoCapture", lambda idx: fake)
    cap = detection.open_camera({"index": 0, "width": 1280, "height": 720, "fps": 30})
    assert cap is fake
    assert fake.props[detection.cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert fake.props[detection.cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert fake.props[detection.cv2.CAP_PROP_FPS] == 30


def test_open_rtsp_passes_url_without_device_properties(monkeypatch):
    fake = _FakeCap()
    sources = []
    def open_capture(source):
        sources.append(source)
        return fake
    monkeypatch.setattr(detection.cv2, "VideoCapture", open_capture)
    url = "rtsp://user:password@192.0.2.1:554/stream"
    assert detection.open_camera({"rtsp_url": url}) is fake
    assert sources == [url]
    assert fake.props == {}


def test_failed_rtsp_releases_capture_and_omits_credentials(monkeypatch):
    fake = _FakeCap(opened=False)
    monkeypatch.setattr(detection.cv2, "VideoCapture", lambda source: fake)
    with pytest.raises(SystemExit) as error:
        detection.open_camera({"rtsp_url": "rtsp://user:secret@192.0.2.1/stream"})
    assert "RTSP stream" in str(error.value)
    assert "secret" not in str(error.value)
    assert fake.released
