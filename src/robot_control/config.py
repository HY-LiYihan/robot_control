from dataclasses import dataclass


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
