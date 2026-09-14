"""Per-id constant-velocity Kalman tracker over (x, y, yaw).

State:        [x, y, yaw, vx, vy, w]^T
Measurement:  [x, y, yaw]^T (yaw residual is wrapped to [-pi, pi])
Process model: constant velocity, Gaussian acceleration noise.
"""

from __future__ import annotations

import math

import numpy as np


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class _Filter:
    def __init__(self, x: float, y: float, yaw: float, t: float,
                 q_accel: float, r_pos: float, r_yaw: float) -> None:
        self.x = np.array([x, y, yaw, 0.0, 0.0, 0.0])
        self.P = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
        self.t = t
        self.last_measurement_t = t
        self.q_accel = q_accel
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.R = np.diag([r_pos ** 2, r_pos ** 2, r_yaw ** 2])

    def _F_Q(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        q = self.q_accel ** 2
        # discrete-time white-noise acceleration covariance, decoupled per axis
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        block = np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]]) * q
        Q = np.zeros((6, 6))
        for i in (0, 1, 2):
            Q[np.ix_([i, i + 3], [i, i + 3])] = block
        return F, Q

    def predict(self, t: float) -> None:
        dt = max(t - self.t, 1e-3)
        F, Q = self._F_Q(dt)
        self.x = F @ self.x
        self.x[2] = _wrap(self.x[2])
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, zx: float, zy: float, zyaw: float) -> None:
        z = np.array([zx, zy, zyaw])
        y = z - self.H @ self.x
        y[2] = _wrap(y[2])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = _wrap(self.x[2])
        self.P = (np.eye(6) - K @ self.H) @ self.P


class KalmanTracker:
    def __init__(self, q_accel: float = 1.0, r_pos: float = 0.05,
                 r_yaw: float = 0.1, timeout_s: float = 1.0) -> None:
        self.q_accel = q_accel
        self.r_pos = r_pos
        self.r_yaw = r_yaw
        self.timeout_s = timeout_s
        self.filters: dict[int, _Filter] = {}

    def update(self, rid: int, x: float, y: float, yaw: float, t: float
               ) -> tuple[float, float, float]:
        f = self.filters.get(rid)
        if f is None or (t - f.t) > self.timeout_s:
            f = _Filter(x, y, yaw, t, self.q_accel, self.r_pos, self.r_yaw)
            self.filters[rid] = f
        else:
            f.predict(t)
            f.update(x, y, yaw)
            f.last_measurement_t = t
        return float(f.x[0]), float(f.x[1]), float(f.x[2])

    def ids(self) -> list[int]:
        return list(self.filters.keys())

    def last_measurement_t(self, rid: int) -> float | None:
        f = self.filters.get(rid)
        return None if f is None else f.last_measurement_t

    def predict_only(self, rid: int, t: float) -> tuple[float, float, float] | None:
        """Advance a tracked id's state to time t without ingesting a measurement.

        Returns (x, y, yaw) or None if the id is unknown or has timed out.
        Each call advances the filter's internal time, so subsequent
        measurements use a small dt (correct under the constant-velocity model)."""
        f = self.filters.get(rid)
        if f is None or (t - f.t) > self.timeout_s:
            return None
        f.predict(t)
        return float(f.x[0]), float(f.x[1]), float(f.x[2])
