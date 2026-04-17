"""Per-camera capture + detection thread.

Owns a VideoCapture, a FrameGrabber, a detector, undistort maps, and an
OpenCV preview window for its camera. Emits Detection items onto a shared
queue. Never blocks on the queue: when full, drops the oldest item and
increments `self.dropped`.
"""

from __future__ import annotations

import dataclasses
import math
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np

from yfips import config
from yfips.detection import (
    CAPTURE_FAILURE_LIMIT,
    FrameGrabber,
    compute_homography,
    image_to_world,
    yaw_from_forward,
)


@dataclasses.dataclass
class Detection:
    rid: int
    x: float
    y: float
    yaw: float
    t: float
    camera: str


def _build_undistort_maps(cam_cfg: dict[str, Any]) -> Any:
    K = cam_cfg.get("camera_matrix")
    D = cam_cfg.get("dist_coeffs")
    if not K or not D:
        return None
    K = np.array(K, dtype=np.float32)
    D = np.array(D, dtype=np.float32)
    size = (cam_cfg["width"], cam_cfg["height"])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=0.0, newImgSize=size)
    return cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_16SC2)


class CameraWorker:
    def __init__(
        self,
        cam_cfg: dict[str, Any],
        world_corners_m: list[list[float]],
        detector: Any,
        out_queue: queue.Queue,
        stop_event: threading.Event,
        config_lock: threading.Lock,
        capture: Any | None = None,
        undistort_enabled: bool = True,
    ) -> None:
        self.cam_cfg = cam_cfg
        self.world_corners_m = world_corners_m
        self.detector = detector
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.config_lock = config_lock
        self.name = cam_cfg["name"]
        self.dropped = 0
        self.failed = False
        self._latest_drawn: np.ndarray | None = None
        self._latest_lock = threading.Lock()

        if capture is None:
            capture = cv2.VideoCapture(cam_cfg["index"])
            if not capture.isOpened():
                self.failed = True
            else:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
                capture.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
        self.capture = capture

        self.undistort_maps = (
            _build_undistort_maps(cam_cfg) if undistort_enabled else None
        )

        self.clicker = _PerCameraClicker(self)

        self.grabber = FrameGrabber(capture) if not self.failed else None
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"cam-{self.name}")

    def start(self) -> CameraWorker:
        if self.failed:
            return self
        self.grabber.start()
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        if self.grabber is not None:
            self.grabber.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def latest_drawn_frame(self) -> np.ndarray | None:
        with self._latest_lock:
            return None if self._latest_drawn is None else self._latest_drawn.copy()

    def _emit(self, det: Detection) -> None:
        try:
            self.out_queue.put_nowait(det)
        except queue.Full:
            try:
                self.out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.out_queue.put_nowait(det)
            except queue.Full:
                pass
            self.dropped += 1

    def _run(self) -> None:
        last_counter = 0
        while not self.stop_event.is_set():
            frame, counter, misses = self.grabber.get_new(last_counter,
                                                         timeout=0.5)
            if misses >= CAPTURE_FAILURE_LIMIT:
                print(f"[{self.name}] camera returned no frame for "
                      f"{CAPTURE_FAILURE_LIMIT} consecutive reads — stopping")
                self.failed = True
                return
            if frame is None:
                continue
            last_counter = counter
            now = time.time()

            if self.undistort_maps is not None:
                frame = cv2.remap(frame, self.undistort_maps[0],
                                  self.undistort_maps[1], cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = self.detector.detect(gray)

            H = self.clicker.H
            for det in detections:
                cx, cy = det["center"]
                for cn in det["corners"]:
                    cv2.circle(frame, (int(cn[0]), int(cn[1])), 3, (0, 0, 255), -1)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                fx_, fy_ = det["forward"]
                cv2.arrowedLine(frame, (int(cx), int(cy)), (int(fx_), int(fy_)),
                                (0, 255, 255), 1, tipLength=0.3)

                if H is not None:
                    x_w, y_w = image_to_world(H, det["center"])
                    yaw = yaw_from_forward(H, det["center"], det["forward"])
                    if math.isfinite(x_w) and math.isfinite(y_w) and math.isfinite(yaw):
                        self._emit(Detection(rid=int(det["id"]), x=x_w, y=y_w,
                                             yaw=yaw, t=now, camera=self.name))
                    label = f"id={det['id']} x={x_w:.2f} y={y_w:.2f}"
                else:
                    label = f"id={det['id']}"
                cv2.putText(frame, label, (int(cx) + 6, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            if H is None:
                cv2.putText(frame, "double-click 4 corners in world_corners_m order",
                            (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255))

            with self._latest_lock:
                self._latest_drawn = frame


class _PerCameraClicker:
    """CalibClicker variant scoped to this worker's cameras[] entry.
    Persists image_corners_px under the shared config_lock."""

    def __init__(self, worker: CameraWorker) -> None:
        self.worker = worker
        existing = worker.cam_cfg.get("image_corners_px")
        self.points: list[list[int]] = list(existing) if existing else []
        self.H: np.ndarray | None = None
        if len(self.points) == 4:
            self.H = compute_homography(self.points, worker.world_corners_m)

    def __call__(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDBLCLK:
            return
        if len(self.points) >= 4:
            self.points = []
            self.H = None
        self.points.append([x, y])
        print(f"[{self.worker.name}] calib point "
              f"{len(self.points)}/4: ({x},{y})")
        if len(self.points) == 4:
            self.H = compute_homography(self.points, self.worker.world_corners_m)
            with self.worker.config_lock:
                cfg = config.load()
                for entry in cfg["cameras"]:
                    if entry["name"] == self.worker.name:
                        entry["image_corners_px"] = self.points
                        break
                config.save(cfg)
            self.worker.cam_cfg["image_corners_px"] = self.points
            print(f"[{self.worker.name}] homography computed and saved")
