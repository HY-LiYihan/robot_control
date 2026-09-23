from __future__ import annotations

from typing import Any, Sequence
from .protocols import RobotBackend
from .types import Pose, RobotState
from ..backends.mujoco import MujocoBackend
from ..backends.real import RealBackend
from ..errors import BackendUnavailableError
from ..scene import SceneClient, twin_socket_path


class Robot:
    def __init__(self, backend: RobotBackend):
        self._backend = backend

    @classmethod
    def connect(cls, backend: str = "mujoco", config: dict[str, Any] | None = None,
                robot: str = "piper") -> "Robot":
        config = dict(config or {})
        robot = "piper" if robot == "pepper" else robot
        if "robot" in config:
            selected = "piper" if config["robot"] == "pepper" else config["robot"]
            if robot != "piper" and robot != selected:
                raise ValueError("conflicting robot selections")
            config.pop("robot")
            robot = selected
        if robot not in ("piper", "franka_fr3"):
            raise ValueError(f"unknown robot: {robot}; choose piper or franka_fr3")
        if robot == "franka_fr3" and backend in ("real", "twin"):
            raise BackendUnavailableError("FR3 real-robot control is not implemented; use --backend mujoco")
        if config.get("scene") is not None and backend != "mujoco":
            raise ValueError("scene is only supported by the MuJoCo backend")
        if backend in ("real", "twin"):
            socket_path = config.pop("twin_socket_path", None)
            twin = SceneClient(socket_path=socket_path or twin_socket_path(), robot="piper")
            try:
                twin.connect()
            except BackendUnavailableError:
                twin.disconnect()
                if backend == "twin":
                    raise BackendUnavailableError("Piper twin is not running; start `robot_control --backend twin` first")
            else:
                try:
                    info = twin.scene_info()
                    if info.get("mode") != "twin" or info.get("robot") != "piper":
                        raise BackendUnavailableError("Twin socket does not belong to a Piper real-robot twin")
                    if info.get("can_name") != config.get("can_name", "can0"):
                        raise ValueError("Twin is connected to a different CAN interface")
                except Exception:
                    twin.disconnect()
                    raise
                return cls(twin)
        requested_scene = None
        if config.get("scene") is not None:
            if backend != "mujoco":
                raise ValueError("scene is only supported by the MuJoCo backend")
            if robot == "franka_fr3":
                from ..fr3.scene_builder import validate_scene
            else:
                from ..backends.scene_builder import validate_scene
            requested_scene = validate_scene(config["scene"])
            config["scene"] = requested_scene
        if backend == "mujoco":
            socket_path = config.pop("socket_path", None)
            scene = SceneClient(socket_path=socket_path, robot=robot)
            try:
                scene.connect()
            except BackendUnavailableError:
                # No shared scene is running: fall back to a private in-process
                # simulation so scripts and tests keep working standalone.
                scene.disconnect()
                if robot == "franka_fr3":
                    from ..fr3.mujoco import MujocoBackend as FR3MujocoBackend
                    impl = FR3MujocoBackend(**config)
                else:
                    impl = MujocoBackend(**config)
                impl.connect()
                return cls(impl)
            try:
                info = scene.scene_info()
                if info.get("robot", "piper") != robot:
                    raise ValueError(f"Running MuJoCo robot is {info.get('robot', 'piper')}, requested {robot}; restart the GUI with --robot")
            except Exception:
                scene.disconnect()
                raise
            if requested_scene is not None:
                try:
                    current_scene = info["scene_path"]
                    if current_scene != str(requested_scene):
                        raise ValueError(f"Running MuJoCo scene is {current_scene}, requested {requested_scene}; restart the GUI with --scene")
                except Exception:
                    scene.disconnect()
                    raise
            return cls(scene)
        elif backend == "real":
            impl = RealBackend(**config)
        else:
            raise ValueError(f"unknown backend: {backend}")
        impl.connect()
        return cls(impl)

    def disconnect(self) -> None:
        self._backend.disconnect()

    def state(self) -> RobotState:
        return self._backend.state()

    def move_joints(self, joints: Sequence[float]) -> None:
        self._backend.move_joints(joints)

    def move_p(self, pose: Pose) -> None:
        self._backend.move_p(pose)

    def gripper(self, width: float, effort: float | None = None) -> None:
        self._backend.gripper(width, effort)

    def stop(self) -> None:
        self._backend.stop()

    def step(self, steps: int = 1) -> None:
        """Advance a standalone MuJoCo simulation explicitly."""
        method = getattr(self._backend, "step", None)
        if method is None:
            raise NotImplementedError("Explicit stepping is only available on a standalone MuJoCo backend")
        method(steps)

    def wait_until_idle(self, timeout: float = 10.0) -> None:
        """Wait for measured simulation motion to settle, not just command acceptance."""
        method = getattr(self._backend, "wait_until_idle", None)
        if method is None:
            raise NotImplementedError("Waiting for motion is not implemented by this backend")
        method(timeout)


PiperRobot = Robot
