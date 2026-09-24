from dataclasses import dataclass
import math


PIPER_INITIAL_JOINTS_DEG = (0.0, 30.0, -45.0, 0.0, 60.0, 0.0)
PIPER_INITIAL_JOINTS_RAD = tuple(math.radians(value) for value in PIPER_INITIAL_JOINTS_DEG)


@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    frame_id: str = "camera_color_optical_frame"


@dataclass
class RobotConfig:
    backend: str = "mujoco"
    can_name: str = "can0"
