from __future__ import annotations

import math

import numpy as np


def positive_duration(value: float) -> float:
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("motion duration must be positive and finite")
    return duration


class JointTrajectory:
    def __init__(self, start: np.ndarray, target: np.ndarray, start_time: float, duration: float):
        self.start = start.copy()
        self.target = target.copy()
        self.start_time = start_time
        self.duration = positive_duration(duration)

    def at(self, current_time: float) -> np.ndarray:
        progress = min(max((current_time - self.start_time) / self.duration, 0.0), 1.0)
        blend = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
        return self.start + blend * (self.target - self.start)

    def finished(self, current_time: float) -> bool:
        return current_time - self.start_time >= self.duration
