from __future__ import annotations

from typing import Any, Sequence
from .protocols import RobotBackend
from .types import Pose, RobotState
from ..backends.mujoco import MujocoBackend
from ..backends.real import RealBackend
from ..errors import BackendUnavailableError
from ..scene import SceneClient, twin_socket_path
from ..selection import normalize_robot, validate_backend_robot
from ..sensors.frame import RGBDFrame


class Robot:
    def __init__(self, backend: RobotBackend):
        self._backend = backend
        self._selected_backend = "mujoco"
        self._selected_robot = "piper"

    @classmethod
    def _attached(cls, impl: RobotBackend, backend: str, robot: str) -> "Robot":
        instance = cls(impl)
        instance._selected_backend = backend
        instance._selected_robot = robot
        return instance

    @classmethod
    def connect(cls, backend: str = "mujoco", config: dict[str, Any] | None = None,
                robot: str = "piper") -> "Robot":
        config = dict(config or {})
        robot = normalize_robot(robot)
        if "robot" in config:
            selected = normalize_robot(config["robot"])
            if robot != "piper" and robot != selected:
                raise ValueError("conflicting robot selections")
            config.pop("robot")
            robot = selected
        validate_backend_robot(backend, robot)
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
                return cls._attached(twin, backend, robot)
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
                return cls._attached(impl, backend, robot)
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
            return cls._attached(scene, backend, robot)
        elif backend == "real":
            impl = RealBackend(**config)
        else:
            raise ValueError(f"unknown backend: {backend}")
        impl.connect()
        return cls._attached(impl, backend, robot)

    def disconnect(self) -> None:
        self._backend.disconnect()

    def state(self) -> RobotState:
        return self._backend.state()

    def camera(self, width: int = 1280, height: int = 720) -> RGBDFrame:
        from ..sensors.service import CameraService
        return CameraService(self._selected_backend, self._selected_robot, self._backend,
                             width=width, height=height).capture()

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
