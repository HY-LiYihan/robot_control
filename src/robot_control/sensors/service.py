from __future__ import annotations

from functools import lru_cache

from ..api.protocols import RobotBackend
from ..errors import BackendUnavailableError
from ..scene import SceneClient
from ..selection import validate_backend_robot
from .extrinsics import (camera_extrinsics, fr3_link7_to_color_optical,
                         piper_link6_to_color_optical, pose_matrix)
from .frame import RGBDFrame
from .mujoco_rgbd import MujocoRGBDCamera
from .realsense import RealSenseCamera


def camera_name(robot: str) -> str:
    return "d435i_check" if robot == "franka_fr3" else "d435i_color_optical_camera"


def mujoco_camera(arm: RobotBackend, robot: str, width: int, height: int) -> MujocoRGBDCamera:
    return MujocoRGBDCamera(arm.model, arm.data, camera=camera_name(robot), width=width, height=height)


@lru_cache(maxsize=1)
def _fr3_kinematics():
    from ..fr3.ik import PinocchioIK
    from ..fr3.mujoco import DEFAULT_URDF

    return PinocchioIK(DEFAULT_URDF)


class CameraService:
    def __init__(self, backend: str, robot: str = "piper", arm: RobotBackend | None = None,
                 *, include_extrinsics: bool = True, width: int = 1280, height: int = 720):
        self.robot = validate_backend_robot(backend, robot)
        self.backend = backend
        self.arm = arm
        self.include_extrinsics = include_extrinsics
        self.width = width
        self.height = height

    def capture(self) -> RGBDFrame:
        if self.arm is None and (self.backend == "mujoco" or self.include_extrinsics):
            raise ValueError("camera extrinsics or simulation require a connected robot")
        if self.backend == "mujoco" and isinstance(self.arm, SceneClient):
            return self.arm.read(width=self.width, height=self.height)
        if self.backend == "mujoco":
            camera = mujoco_camera(self.arm, self.robot, self.width, self.height)
        else:
            camera = RealSenseCamera(width=self.width, height=self.height,
                                    frame_id="d435i_color_optical_frame")
        camera.connect()
        try:
            frame = camera.read()
        finally:
            camera.disconnect()
        if self.backend in ("real", "twin") and self.include_extrinsics:
            state = self.arm.state()
            if abs(state.timestamp - frame.timestamp) > 1.0:
                raise BackendUnavailableError("Robot joint feedback and camera capture are not synchronized")
            if self.robot == "franka_fr3":
                link_pose = _fr3_kinematics().forward(state.joints.positions)
                transform = pose_matrix(link_pose) @ fr3_link7_to_color_optical()
                reference_frame = "fr3_link0"
            else:
                transform = pose_matrix(state.pose) @ piper_link6_to_color_optical()
                reference_frame = "base_link"
            frame.extrinsics = camera_extrinsics(
                transform, reference_frame, frame.frame_id, state.timestamp)
        return frame
