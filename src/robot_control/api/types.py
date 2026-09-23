from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np


@dataclass(frozen=True)
class Pose:
    """Cartesian pose. Position is metres; quaternion is w, x, y, z."""

    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.quaternion) != 4:
            raise ValueError("position must have 3 and quaternion must have 4 values")
        norm = sum(x * x for x in self.quaternion) ** 0.5
        if norm < 1e-9:
            raise ValueError("quaternion must be non-zero")


@dataclass
class JointState:
    positions: np.ndarray
    velocities: np.ndarray | None = None
    gripper: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1)
        if self.positions.size not in (6, 7):
            raise ValueError("arm state requires six Piper or seven FR3 joint positions")
        self.velocities = (np.zeros_like(self.positions) if self.velocities is None
                           else np.asarray(self.velocities, dtype=float).reshape(-1))
        if self.velocities.size != self.positions.size:
            raise ValueError("arm joint velocities must match the position count")


@dataclass
class RobotState:
    connected: bool
    moving: bool
    joints: JointState
    pose: Optional[Pose] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
