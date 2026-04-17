from __future__ import annotations

import argparse
import collections
import json
import math
import os
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


def camera_settings(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Resolve camera index/width/height/fps with CLI > config > default precedence."""
    cam_cfg = cfg.get("camera", {}) or {}
    return {
        "index": args.camera_index if args.camera_index is not None
                 else cam_cfg.get("index", DEFAULT_CAMERA["index"]),
        "width": args.width or cam_cfg.get("width", DEFAULT_CAMERA["width"]),
        "height": args.height or cam_cfg.get("height", DEFAULT_CAMERA["height"]),
        "fps": args.fps or cam_cfg.get("fps", DEFAULT_CAMERA["fps"]),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["apriltag", "image"], default=None,
                        help="detection mode; overrides config.mode")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="camera index; overrides config.camera.index")
    parser.add_argument("--width", type=int, default=None,
                        help="frame width; overrides config.camera.width")
    parser.add_argument("--height", type=int, default=None,
                        help="frame height; overrides config.camera.height")
    parser.add_argument("--fps", type=int, default=None,
                        help="capture fps; overrides config.camera.fps")
    args = parser.parse_args()

    cfg = config.load()
    mode = args.mode or cfg.get("mode", "apriltag")
    cam = camera_settings(cfg, args)
    print(f"[yfips] mode={mode} camera={cam}")

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
    clicker = CalibClicker(cfg)
    detector = build_detector(mode, cfg)

    undistort_maps = None
    if cfg.get("undistort", True) and cfg.get("camera_matrix") and cfg.get("dist_coeffs"):
        K = np.array(cfg["camera_matrix"], dtype=np.float32)
        D = np.array(cfg["dist_coeffs"], dtype=np.float32)
        size = (cam["width"], cam["height"])
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=0.0, newImgSize=size)
        undistort_maps = cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_16SC2)
        print("[yfips] live undistortion enabled")

    cap = open_camera(cam)
    grabber = FrameGrabber(cap).start()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, clicker)

    fps_meter = FpsMeter(window=30)
    last_counter = 0
    try:
        while True:
            frame, counter, misses = grabber.get_new(last_counter, timeout=0.5)
            if misses >= CAPTURE_FAILURE_LIMIT:
                print(f"[yfips] camera returned no frame for "
                      f"{CAPTURE_FAILURE_LIMIT} consecutive reads — exiting")
                break
            if frame is None:
                continue
            last_counter = counter
            now = time.time()
            fps = fps_meter.tick(now)
            if undistort_maps is not None:
                frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1],
                                  cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for p in clicker.points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, (255, 0, 0), -1)

            detections = detector.detect(gray)
            detected_ids = set()
            for det in detections:
                cx, cy = det["center"]
                for cn in det["corners"]:
                    cv2.circle(frame, (int(cn[0]), int(cn[1])), 3, (0, 0, 255), -1)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                fx_, fy_ = det["forward"]
                cv2.arrowedLine(frame, (int(cx), int(cy)), (int(fx_), int(fy_)),
                                (0, 255, 255), 1, tipLength=0.3)

                if clicker.H is not None:
                    x_w, y_w = image_to_world(clicker.H, det["center"])
                    yaw = yaw_from_forward(clicker.H, det["center"], det["forward"])
                    if tracker is not None:
                        x_w, y_w, yaw = tracker.update(det["id"], x_w, y_w, yaw, now)
                    detected_ids.add(det["id"])
                    label = f"id={det['id']} x={x_w:.2f} y={y_w:.2f} yaw={math.degrees(yaw):.0f}"
                    cv2.putText(frame, label, (int(cx) + 6, int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    payload = {"id": det["id"], "x": x_w, "y": y_w,
                               "yaw": yaw, "t": now}
                    pub.send(payload)
                    ros_pub.send(payload)
                else:
                    cv2.putText(frame, f"id={det['id']}", (int(cx) + 6, int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            if clicker.H is not None:
                for rid, x_w, y_w, yaw in emit_predictions(tracker, detected_ids, now):
                    payload = {"id": rid, "x": x_w, "y": y_w,
                               "yaw": yaw, "t": now, "predicted": True}
                    pub.send(payload)
                    ros_pub.send(payload)

            cv2.putText(frame, f"{mode} | fps: {fps:.1f}",
                        (0, cam["height"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
            if clicker.H is None:
                cv2.putText(frame, "double-click 4 corners in world_corners_m order",
                            (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255))

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        grabber.stop()
        cv2.destroyAllWindows()
        ros_pub.shutdown()


if __name__ == "__main__":
    main()
