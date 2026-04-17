from __future__ import annotations

import argparse
import collections
import json
import math
import os
import queue
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

from yfips import config
from yfips.image_detector import ImageRefDetector
from yfips.kalman_tracker import KalmanTracker
from yfips.ros_publisher import RosPublisher
from yfips.tracker import EMATracker

WINDOW_NAME = "YFIPS"
DEFAULT_CAMERA = {"index": 0, "width": 640, "height": 480, "fps": 60}


CAPTURE_FAILURE_LIMIT = 30  # consecutive read() failures before bailing


class FrameGrabber:
    """Background thread that calls cap.read() in a loop and exposes only
    the latest frame. Decouples capture latency from processing latency
    so a slow detector doesn't pile up frames."""

    def __init__(self, cap: Any) -> None:
        self.cap = cap
        self._cv = threading.Condition()
        self._frame: np.ndarray | None = None
        self._counter = 0
        self._misses = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> FrameGrabber:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._thread.join(timeout=1.0)
        self.cap.release()

    def _run(self) -> None:
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            with self._cv:
                if not ret:
                    self._misses += 1
                else:
                    self._frame = frame
                    self._counter += 1
                    self._misses = 0
                self._cv.notify_all()

    def get_new(self, last_counter: int, timeout: float = 1.0
                ) -> tuple[np.ndarray | None, int, int]:
        """Block until a frame newer than last_counter arrives, or timeout.
        Returns (frame, counter, misses). frame is None on timeout."""
        with self._cv:
            self._cv.wait_for(
                lambda: self._counter > last_counter or self._stop.is_set(),
                timeout=timeout,
            )
            if self._counter <= last_counter:
                return None, last_counter, self._misses
            return self._frame, self._counter, self._misses


class FpsMeter:
    """Rolling-window fps meter measured from loop-tick timestamps.

    Reports the actual loop interval, not single-frame compute time, so
    capture stalls and dropped frames are visible."""

    def __init__(self, window: int = 30) -> None:
        self._times: collections.deque[float] = collections.deque(maxlen=window)

    def tick(self, t: float) -> float:
        self._times.append(t)
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span


def open_camera(cam: dict[str, Any]) -> Any:
    """Open a cv2.VideoCapture with the given settings dict.
    Raises SystemExit with a clear message if the device fails to open."""
    cap = cv2.VideoCapture(cam["index"])
    if not cap.isOpened():
        raise SystemExit(
            f"[yfips] failed to open camera at index {cam['index']} — "
            "check the device, change config.camera.index, or pass --camera-index"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    cap.set(cv2.CAP_PROP_FPS, cam["fps"])
    return cap


class Publisher:
    _FINITE_FIELDS = ("x", "y", "yaw")

    def __init__(self, udp_cfg: dict[str, Any]) -> None:
        self.enabled = udp_cfg.get("enabled", False)
        self.addr = (udp_cfg.get("host", "127.0.0.1"), int(udp_cfg.get("port", 9999)))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.enabled else None
        self._warned_ids: set[int | None] = set()

    def send(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for field in self._FINITE_FIELDS:
            v = payload.get(field)
            if v is None or not math.isfinite(v):
                rid = payload.get("id")
                if rid not in self._warned_ids:
                    print(f"[publisher] dropping non-finite {field}={v!r} for id={rid}")
                    self._warned_ids.add(rid)
                return
        # allow_nan=False is a defensive backstop in case a non-numeric sneaks in.
        self.sock.sendto(json.dumps(payload, allow_nan=False).encode("utf-8"), self.addr)


def compute_homography(image_corners_px: Any, world_corners_m: Any) -> np.ndarray:
    src = np.array(image_corners_px, dtype=np.float32)
    dst = np.array(world_corners_m, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def image_to_world(H: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    p = np.array([pt[0], pt[1], 1.0])
    w = H @ p
    return float(w[0] / w[2]), float(w[1] / w[2])


class CalibClicker:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        existing = cfg.get("image_corners_px")
        self.points: list[list[int]] = list(existing) if existing else []
        self.H: np.ndarray | None = None
        if len(self.points) == 4:
            self.H = compute_homography(self.points, cfg["world_corners_m"])

    def __call__(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDBLCLK:
            return
        if len(self.points) >= 4:
            self.points = []
            self.H = None
        self.points.append([x, y])
        print(f"calib point {len(self.points)}/4: ({x},{y})")
        if len(self.points) == 4:
            self.H = compute_homography(self.points, self.cfg["world_corners_m"])
            self.cfg["image_corners_px"] = self.points
            config.save(self.cfg)
            print("Homography computed and saved.")


_APRILTAG_KEY_MAP = {
    "family": "families",
    "nthreads": "nthreads",
    "quad_decimate": "quad_decimate",
    "quad_blur": "quad_blur",
    "refine_edges": "refine_edges",
    "refine_decode": "refine_decode",
    "refine_pose": "refine_pose",
}


def apriltag_options_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate config.apriltag_mode into kwargs for apriltag.DetectorOptions.
    Unknown keys are dropped; missing block returns {}."""
    at_cfg = cfg.get("apriltag_mode") or {}
    return {dst: at_cfg[src] for src, dst in _APRILTAG_KEY_MAP.items() if src in at_cfg}


class AprilTagAdapter:
    """Wraps the apriltag library into the common detection dict shape."""

    def __init__(self, options_kwargs: dict[str, Any] | None = None) -> None:
        import apriltag  # imported lazily so image-mode users don't need it
        if options_kwargs:
            self.detector = apriltag.Detector(apriltag.DetectorOptions(**options_kwargs))
        else:
            self.detector = apriltag.Detector()

    @staticmethod
    def _adapt(det: Any) -> dict[str, Any]:
        # swatbotics corner order: 0 back-left, 1 back-right, 2 front-right, 3 front-left.
        # The right edge (midpoint of c1+c2) sits ahead of center along the tag's +x.
        c = np.asarray(det.corners)
        forward = 0.5 * (c[1] + c[2])
        return {
            "id": int(det.tag_id),
            "center": tuple(det.center),
            "forward": (float(forward[0]), float(forward[1])),
            "corners": c,
        }

    def detect(self, gray: np.ndarray) -> list[dict[str, Any]]:
        return [self._adapt(det) for det in self.detector.detect(gray)]


def build_detector(mode: str, cfg: dict[str, Any]) -> Any:
    if mode == "apriltag":
        return AprilTagAdapter(options_kwargs=apriltag_options_kwargs(cfg))
    if mode == "image":
        ref_dir = cfg.get("references_dir") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "references")
        im_cfg = cfg.get("image_mode", {})
        return ImageRefDetector(
            ref_dir,
            min_inliers=int(im_cfg.get("min_inliers", 15)),
            use_flann=bool(im_cfg.get("use_flann", False)),
        )
    raise ValueError(f"unknown mode: {mode}")


def yaw_from_forward(H: np.ndarray, center_px: tuple[float, float],
                     forward_px: tuple[float, float]) -> float:
    cw = image_to_world(H, center_px)
    fw = image_to_world(H, forward_px)
    return math.atan2(fw[1] - cw[1], fw[0] - cw[0])


def emit_predictions(tracker: Any, detected_ids: set[int], t: float,
                     min_age_s: float = 0.0
                     ) -> Iterator[tuple[int, float, float, float]]:
    """Yield (rid, x, y, yaw) for tracked ids absent this frame whose
    tracker can extrapolate. Trackers without a velocity model (EMA)
    silently produce nothing.

    min_age_s: skip ids whose last real measurement is newer than this
    threshold. Used by the prediction timer to avoid emitting a
    prediction immediately after a detection lands for the same id."""
    if tracker is None:
        return
    for rid in tracker.ids():
        if rid in detected_ids:
            continue
        if min_age_s > 0.0:
            last = tracker.last_measurement_t(rid)
            if last is not None and (t - last) < min_age_s:
                continue
        out = tracker.predict_only(rid, t)
        if out is None:
            continue
        yield (rid, *out)


def drain_detections(
    q: queue.Queue,
    tracker: Any,
    tracker_lock: threading.Lock,
    pub: Any,
    ros_pub: Any,
) -> set[int]:
    """Drain all available Detection items from q, update the tracker
    under tracker_lock, and publish. Returns the set of ids seen this
    drain."""
    seen: set[int] = set()
    while True:
        try:
            det = q.get_nowait()
        except queue.Empty:
            return seen
        with tracker_lock:
            if tracker is not None:
                x, y, yaw = tracker.update(det.rid, det.x, det.y, det.yaw, det.t)
            else:
                x, y, yaw = det.x, det.y, det.yaw
        payload = {"id": det.rid, "x": x, "y": y, "yaw": yaw,
                   "t": det.t, "camera": det.camera}
        pub.send(payload)
        ros_pub.send(payload)
        seen.add(det.rid)


def _prediction_timer_loop(
    tracker: Any,
    tracker_lock: threading.Lock,
    pub: Any,
    ros_pub: Any,
    stop_event: threading.Event,
    interval_s: float,
) -> None:
    while not stop_event.wait(interval_s):
        now = time.time()
        with tracker_lock:
            preds = list(emit_predictions(
                tracker, detected_ids=set(), t=now, min_age_s=interval_s,
            ))
        for rid, x, y, yaw in preds:
            payload = {"id": rid, "x": x, "y": y, "yaw": yaw,
                       "t": now, "predicted": True}
            pub.send(payload)
            ros_pub.send(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["apriltag", "image"], default=None,
                        help="detection mode; overrides config.mode")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="camera index; single-camera configs only")
    parser.add_argument("--width", type=int, default=None,
                        help="frame width; single-camera configs only")
    parser.add_argument("--height", type=int, default=None,
                        help="frame height; single-camera configs only")
    parser.add_argument("--fps", type=int, default=None,
                        help="capture fps; single-camera configs only")
    args = parser.parse_args()

    cfg = config.load()
    try:
        config.validate_cameras(cfg["cameras"])
    except ValueError as e:
        raise SystemExit(f"[yfips] invalid cameras config: {e}") from e

    if len(cfg["cameras"]) > 1 and any(v is not None for v in (
            args.camera_index, args.width, args.height, args.fps)):
        raise SystemExit(
            "[yfips] --camera-index/--width/--height/--fps are only "
            "supported with a single camera in config.cameras[]; "
            "edit config.json for multi-camera setups"
        )

    if len(cfg["cameras"]) == 1:
        cam = cfg["cameras"][0]
        if args.camera_index is not None:
            cam["index"] = args.camera_index
        if args.width is not None:
            cam["width"] = args.width
        if args.height is not None:
            cam["height"] = args.height
        if args.fps is not None:
            cam["fps"] = args.fps

    mode = args.mode or cfg.get("mode", "apriltag")
    print(f"[yfips] mode={mode} cameras={[c['name'] for c in cfg['cameras']]}")

    # Lazy import to avoid a circular dependency with camera_worker.py,
    # which imports helpers from this module at import time.
    from yfips.camera_worker import CameraWorker

    pub = Publisher(cfg["udp"])
    ros_pub = RosPublisher(cfg.get("ros", {}))
    tracker_cfg = cfg.get("tracker", {})
    if not tracker_cfg.get("enabled", True):
        tracker = None
    elif tracker_cfg.get("type", "ema") == "kalman":
        tracker = KalmanTracker(
            q_accel=float(tracker_cfg.get("q_accel", 1.0)),
            r_pos=float(tracker_cfg.get("r_pos", 0.05)),
            r_yaw=float(tracker_cfg.get("r_yaw", 0.1)),
            timeout_s=float(tracker_cfg.get("timeout_s", 1.0)),
        )
    else:
        tracker = EMATracker(
            alpha=float(tracker_cfg.get("alpha", 0.4)),
            timeout_s=float(tracker_cfg.get("timeout_s", 1.0)),
        )
    prediction_interval_s = float(tracker_cfg.get("prediction_interval_s", 0.05))

    stop_event = threading.Event()
    tracker_lock = threading.Lock()
    config_lock = threading.Lock()
    det_queue: queue.Queue = queue.Queue(maxsize=1024)
    undistort_enabled = bool(cfg.get("undistort", True))

    workers: list[CameraWorker] = []
    for cam_cfg in cfg["cameras"]:
        detector = build_detector(mode, cfg)
        w = CameraWorker(
            cam_cfg=cam_cfg,
            world_corners_m=cfg["world_corners_m"],
            detector=detector,
            out_queue=det_queue,
            stop_event=stop_event,
            config_lock=config_lock,
            undistort_enabled=undistort_enabled,
        )
        if w.failed:
            print(f"[yfips] {cam_cfg['name']} failed to open — skipping")
            continue
        workers.append(w)

    if not workers:
        raise SystemExit("[yfips] no cameras could be opened")

    for w in workers:
        w.start()
        cv2.namedWindow(f"YFIPS:{w.name}")
        cv2.setMouseCallback(f"YFIPS:{w.name}", w.clicker)

    pred_thread = threading.Thread(
        target=_prediction_timer_loop,
        args=(tracker, tracker_lock, pub, ros_pub, stop_event,
              prediction_interval_s),
        daemon=True, name="prediction-timer",
    )
    if tracker is not None:
        pred_thread.start()

    try:
        while not stop_event.is_set():
            if all(w.failed for w in workers):
                print("[yfips] all cameras failed — exiting")
                break
            for w in workers:
                frame = w.latest_drawn_frame()
                if frame is not None:
                    cv2.imshow(f"YFIPS:{w.name}", frame)
            drain_detections(det_queue, tracker, tracker_lock, pub, ros_pub)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        stop_event.set()
        for w in workers:
            w.stop()
        if tracker is not None:
            pred_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        ros_pub.shutdown()


if __name__ == "__main__":
    main()
