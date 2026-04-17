"""Per-id exponential moving average smoother for (x, y, yaw).
Yaw is smoothed via its unit vector to avoid wrap-around issues."""

from __future__ import annotations

import math


class EMATracker:
    def __init__(self, alpha: float = 0.4, timeout_s: float = 1.0) -> None:
        self.alpha = alpha
        self.timeout_s = timeout_s
        self.state: dict[int, tuple[float, float, float, float, float]] = {}

    def update(self, rid: int, x: float, y: float, yaw: float, t: float
               ) -> tuple[float, float, float]:
        a = self.alpha
        prev = self.state.get(rid)
        cy, sy = math.cos(yaw), math.sin(yaw)
        if prev is None or (t - prev[4]) > self.timeout_s:
            self.state[rid] = (x, y, cy, sy, t)
        else:
            px, py, pcy, psy, _ = prev
            self.state[rid] = (
                a * x + (1 - a) * px,
                a * y + (1 - a) * py,
                a * cy + (1 - a) * pcy,
                a * sy + (1 - a) * psy,
                t,
            )
        sx, sy_, scy, ssy, _ = self.state[rid]
        return sx, sy_, math.atan2(ssy, scy)

    def ids(self) -> list[int]:
        return list(self.state.keys())

    def predict_only(self, rid: int, t: float) -> None:
        # EMA has no velocity model; nothing to extrapolate.
        return None

    def last_measurement_t(self, rid: int) -> float | None:
        row = self.state.get(rid)
        return None if row is None else row[4]
