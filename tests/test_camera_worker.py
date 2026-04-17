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
        return True, self.frame.copy()

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
    assert abs(det.x - 100.0) < 1e-6
    assert abs(det.y - 100.0) < 1e-6
    assert abs(det.yaw) < 1e-6  # forward along +x → yaw 0


def test_worker_without_homography_pushes_nothing():
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    worker = _make_worker(q, stop, image_corners_px=None)
    worker.start()
    time.sleep(0.1)
    stop.set()
    worker.stop()
    assert q.empty()


def test_worker_drops_oldest_when_queue_full():
    q: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()
    image_corners = [[0, 0], [640, 0], [640, 480], [0, 480]]
    worker = _make_worker(q, stop, image_corners_px=image_corners)
    worker.start()
    time.sleep(0.3)
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
