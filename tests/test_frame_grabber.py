"""Tests for the threaded FrameGrabber."""

import threading
import time

import numpy as np

from yfips.detection import FrameGrabber


class _FakeCap:
    """Fake VideoCapture that yields a list of (ret, frame) pairs then loops the last."""

    def __init__(self, frames, fail_after=None, frame_delay_s=0.005):
        self.frames = frames
        self.fail_after = fail_after
        self.delay = frame_delay_s
        self.calls = 0
        self.released = False
        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            i = self.calls
            self.calls += 1
        time.sleep(self.delay)
        if self.fail_after is not None and i >= self.fail_after:
            return False, None
        if i < len(self.frames):
            return True, self.frames[i]
        # Loop the last frame forever to avoid starving consumers.
        return True, self.frames[-1]

    def release(self):
        self.released = True


def _frame(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(8, 8), dtype=np.uint8)


def test_grabber_yields_first_frame():
    cap = _FakeCap([_frame(1)])
    g = FrameGrabber(cap).start()
    try:
        frame, counter, misses = g.get_new(last_counter=0, timeout=0.5)
        assert frame is not None
        assert counter >= 1
        assert misses == 0
    finally:
        g.stop()


def test_grabber_advances_counter_on_each_frame():
    cap = _FakeCap([_frame(1), _frame(2), _frame(3)])
    g = FrameGrabber(cap).start()
    try:
        seen = []
        last = 0
        for _ in range(3):
            _, last, _ = g.get_new(last_counter=last, timeout=0.5)
            seen.append(last)
        assert seen == sorted(seen)  # monotonic
        assert seen[-1] >= 3
    finally:
        g.stop()


def test_grabber_counts_misses_on_failure():
    cap = _FakeCap([], fail_after=0)  # always fails
    g = FrameGrabber(cap).start()
    try:
        time.sleep(0.05)
        frame, _, misses = g.get_new(last_counter=0, timeout=0.1)
        assert frame is None
        assert misses > 0
    finally:
        g.stop()


def test_grabber_stop_releases_cap():
    cap = _FakeCap([_frame(1)])
    g = FrameGrabber(cap).start()
    g.stop()
    assert cap.released


def test_grabber_get_new_times_out_when_no_progress():
    # Grabber has captured frame 1; asking for newer than the latest counter
    # should time out without crashing.
    cap = _FakeCap([_frame(1)], fail_after=1)
    g = FrameGrabber(cap).start()
    try:
        # Wait for the single frame to land
        _, latest, _ = g.get_new(last_counter=0, timeout=0.5)
        # Now ask for a NEWER frame; should time out
        frame, counter, _ = g.get_new(last_counter=latest, timeout=0.1)
        assert frame is None
        assert counter == latest
    finally:
        g.stop()


def test_slow_consumer_receives_latest_frame_without_backlog():
    drained = threading.Event()
    resume = threading.Event()
    frames = [_frame(1), _frame(2), _frame(3)]

    class BurstCapture:
        def __init__(self):
            self.index = 0

        def read(self):
            if self.index < len(frames):
                frame = frames[self.index]
                self.index += 1
                return True, frame
            # Reaching the next read means all three frames were published.
            drained.set()
            resume.wait(timeout=2)
            return False, None

        def release(self):
            pass

    grabber = FrameGrabber(BurstCapture()).start()
    try:
        assert drained.wait(timeout=1)
        frame, counter, misses = grabber.get_new(last_counter=0, timeout=0.1)
        assert frame is frames[-1]
        assert counter == 3
        assert misses == 0
    finally:
        grabber._stop.set()
        resume.set()
        grabber.stop()
