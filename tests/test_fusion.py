"""Cross-camera fusion sanity: two detections for the same id from two
cameras, fed through a shared tracker, produce a sensibly blended state.
Also: the prediction timer emits one payload per id per tick regardless
of how many cameras saw it."""

from yfips.detection import emit_predictions
from yfips.kalman_tracker import KalmanTracker
from yfips.tracker import EMATracker


def test_ema_fuses_two_camera_updates_on_same_id():
    ema = EMATracker(alpha=0.5, timeout_s=10.0)
    ema.update(7, 1.0, 2.0, 0.0, t=0.0)
    x, y, yaw = ema.update(7, 1.4, 2.2, 0.0, t=0.01)
    # Expected EMA with alpha=0.5: prev = (1.0, 2.0); new = 0.5*new + 0.5*prev
    assert abs(x - 1.2) < 1e-6
    assert abs(y - 2.1) < 1e-6
    assert abs(yaw) < 1e-6


def test_kalman_fuses_two_camera_updates_on_same_id():
    k = KalmanTracker(q_accel=0.05, r_pos=0.05, r_yaw=0.05, timeout_s=10.0)
    k.update(7, 1.0, 2.0, 0.0, t=0.0)
    x, y, _ = k.update(7, 1.4, 2.2, 0.0, t=0.01)
    assert 1.0 <= x <= 1.4
    assert 2.0 <= y <= 2.2


def test_emit_predictions_once_per_id_after_multiple_camera_updates():
    k = KalmanTracker(q_accel=0.05, r_pos=0.01, r_yaw=0.01, timeout_s=10.0)
    k.update(7, 1.0, 2.0, 0.0, t=0.0)
    k.update(7, 1.1, 2.1, 0.0, t=0.01)
    out = list(emit_predictions(k, detected_ids=set(), t=0.2, min_age_s=0.05))
    assert [row[0] for row in out] == [7]
