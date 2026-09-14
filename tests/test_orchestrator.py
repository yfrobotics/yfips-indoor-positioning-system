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

    assert [p["camera"] for p in pub.sent] == ["cam0", "cam1", "cam0"]
    assert [p["id"] for p in pub.sent] == [1, 1, 2]
    assert sorted(tracker.ids()) == [1, 2]
    assert len(ros_pub.sent) == 3


def test_drain_detections_handles_empty_queue():
    q: queue.Queue = queue.Queue()
    tracker = EMATracker()
    pub = MockPublisher()
    drain_detections(q, tracker, threading.Lock(), pub, pub)
    assert pub.sent == []
