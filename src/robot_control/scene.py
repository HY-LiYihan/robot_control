from __future__ import annotations

"""Shared MuJoCo scene over a Unix domain socket.

One process (``robot_control --backend mujoco``) owns the simulation and runs a
:class:`SceneServer`.  Every other CLI invocation acts as a
:class:`SceneClient`, so all commands mutate the same ``MjModel``/``MjData``
instance and the viewer reflects the result.
"""

import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .api.types import JointState, Pose, RobotState
from .errors import BackendUnavailableError, IKError, NotConnectedError
from .sensors.frame import CameraExtrinsics, CameraIntrinsics, RGBDFrame


def default_socket_path(robot: str = "piper") -> Path:
    if robot == "franka_fr3":
        override = __import__("os").environ.get("FR3_SCENE_SOCKET")
        return Path(override) if override else Path(tempfile.gettempdir()) / "fr3_scene.sock"
    override = __import__("os").environ.get("PIPER_SCENE_SOCKET")
    return Path(override) if override else Path(tempfile.gettempdir()) / "piper_scene.sock"


def twin_socket_path() -> Path:
    override = __import__("os").environ.get("PIPER_TWIN_SOCKET")
    return Path(override) if override else Path(tempfile.gettempdir()) / "piper_twin.sock"


def _state_to_dict(state: RobotState) -> dict:
    return {
        "connected": state.connected,
        "moving": state.moving,
        "joints": {
            "positions": [float(v) for v in state.joints.positions],
            "velocities": [float(v) for v in state.joints.velocities],
            "gripper": float(state.joints.gripper),
            "timestamp": state.joints.timestamp,
        },
        "pose": None if state.pose is None else {
            "position": [float(v) for v in state.pose.position],
            "quaternion": [float(v) for v in state.pose.quaternion],
        },
        "error": state.error,
        "timestamp": state.timestamp,
    }


def _state_from_dict(data: dict) -> RobotState:
    joints = JointState(
        np.asarray(data["joints"]["positions"], dtype=float),
        np.asarray(data["joints"]["velocities"], dtype=float),
        float(data["joints"]["gripper"]),
        float(data["joints"]["timestamp"]),
    )
    pose = None
    if data["pose"] is not None:
        pose = Pose(tuple(data["pose"]["position"]), tuple(data["pose"]["quaternion"]))
    return RobotState(
        bool(data["connected"]), bool(data["moving"]), joints, pose,
        data.get("error"), float(data.get("timestamp", time.time())),
    )


class SceneServer:
    """Host the shared MuJoCo scene and serve CLI requests over a socket."""

    def __init__(self, backend, socket_path=None, robot: str = "piper", mode: str = "mujoco"):
        self.backend = backend
        self.robot = robot
        self.mode = mode
        self.socket_path = Path(socket_path) if socket_path else default_socket_path(robot)
        self.lock = threading.Lock()
        self._server = None
        self._thread = None
        self._owns_socket = False
        self._camera = None
        self._camera_size = None

    def start(self) -> None:
        if self._server is not None:
            return
        existing = SceneClient(socket_path=self.socket_path, robot=self.robot)
        try:
            existing.connect()
        except BackendUnavailableError:
            pass
        else:
            existing.disconnect()
            raise BackendUnavailableError(f"MuJoCo scene is already running at {self.socket_path}")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.socket_path))
            if self.mode == "twin":
                self.socket_path.chmod(0o600)
        except OSError as exc:
            server.close()
            if self.mode == "twin":
                self.socket_path.unlink(missing_ok=True)
            raise BackendUnavailableError(
                f"cannot bind scene socket {self.socket_path}: {exc}"
            ) from exc
        self._owns_socket = True
        server.listen(16)
        self._server = server
        self._thread = threading.Thread(
            target=self._accept_loop, name="piper-scene-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        self._thread = None
        if self._camera is not None:
            self._camera.disconnect()
            self._camera = None
        if self._owns_socket:
            self.socket_path.unlink(missing_ok=True)
            self._owns_socket = False

    def _accept_loop(self) -> None:
        while self._server is not None:
            try:
                conn, _ = self._server.accept()
            except OSError:
                break
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()

    def _serve_conn(self, conn) -> None:
        with conn:
            reader = conn.makefile("rb")
            while True:
                line = reader.readline()
                if not line or (self.mode == "twin" and self._server is None):
                    break
                try:
                    payload = json.loads(line.decode("utf-8"))
                except ValueError as exc:
                    self._respond_error(conn, exc)
                    continue
                try:
                    self._dispatch(conn, payload)
                except Exception as exc:  # surface any backend failure to the client
                    self._respond_error(conn, exc)

    def _dispatch(self, conn, payload: dict) -> None:
        command = payload.get("cmd")
        if command == "scene_info":
            path = getattr(self.backend, "scene_path", None)
            self._respond(conn, {"ok": True, "robot": self.robot, "mode": self.mode,
                                 "can_name": getattr(self.backend, "can_name", None),
                                 "scene_path": str(path) if path is not None else None})
        elif command == "state":
            with self.lock:
                state = self.backend.state()
            self._respond(conn, {"ok": True, "state": _state_to_dict(state)})
        elif command == "move_p":
            with self.lock:
                self.backend.move_p(Pose(tuple(payload["position"]), tuple(payload["quaternion"])))
            self._respond(conn, {"ok": True})
        elif command == "move_joints":
            with self.lock:
                self.backend.move_joints(payload["joints"])
            self._respond(conn, {"ok": True})
        elif command == "gripper":
            with self.lock:
                self.backend.gripper(float(payload["width"]), payload.get("effort"))
            self._respond(conn, {"ok": True})
        elif command == "stop":
            with self.lock:
                self.backend.stop()
            self._respond(conn, {"ok": True})
        elif command == "camera":
            if self.mode == "twin":
                raise BackendUnavailableError("Twin camera capture uses the real RealSense; use `robot_control --backend twin --robot piper camera`")
            self._serve_camera(conn, payload)
        else:
            raise ValueError(f"unknown scene command: {command}")

    def _camera_or_create(self, width: int, height: int):
        if self._camera is None or self._camera_size != (width, height):
            if self._camera is not None:
                self._camera.disconnect()
            from .sensors.service import mujoco_camera
            self._camera = mujoco_camera(self.backend, self.robot, width, height)
            self._camera.connect()
            self._camera_size = (width, height)
        return self._camera

    def _serve_camera(self, conn, payload: dict) -> None:
        width = int(payload.get("width", 1280))
        height = int(payload.get("height", 720))
        with self.lock:
            frame = self._camera_or_create(width, height).read()
        intr = frame.intrinsics
        header = {
            "ok": True,
            "frame_id": frame.frame_id,
            "depth_scale": float(frame.depth_scale),
            "timestamp": frame.timestamp,
            "color_shape": list(frame.color.shape),
            "depth_shape": list(frame.depth.shape),
            "intrinsics": {"width": intr.width, "height": intr.height, "fx": intr.fx,
                           "fy": intr.fy, "cx": intr.cx, "cy": intr.cy},
            "extrinsics": None if frame.extrinsics is None else {
                "rotation": list(frame.extrinsics.rotation),
                "translation": list(frame.extrinsics.translation),
                "reference_frame": frame.extrinsics.reference_frame,
                "camera_frame": frame.extrinsics.camera_frame,
                "timestamp": frame.extrinsics.timestamp,
            },
        }
        try:
            conn.sendall((json.dumps(header) + "\n").encode("utf-8"))
            conn.sendall(frame.color.tobytes())
            conn.sendall(frame.depth.tobytes())
        except OSError:
            # A client that disconnects mid-frame is not a server error.
            pass

    def _respond(self, conn, payload: dict) -> None:
        try:
            conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except OSError:
            pass

    def _respond_error(self, conn, exc: Exception) -> None:
        self._respond(conn, {"ok": False, "error_type": type(exc).__name__, "error": str(exc)})


class SceneClient:
    """A :class:`RobotBackend` that forwards every call to the shared scene."""

    def __init__(self, socket_path=None, robot: str = "piper"):
        self.robot = robot
        self.socket_path = Path(socket_path) if socket_path else default_socket_path(robot)
        self._socket = None
        self._reader = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(60.0)
        try:
            sock.connect(str(self.socket_path))
        except OSError as exc:
            sock.close()
            raise BackendUnavailableError(
                f"{self.robot} scene server is not running at {self.socket_path}; "
                f"start it first with `robot_control --backend mujoco --robot {self.robot}`"
            ) from exc
        self._socket = sock
        self._reader = sock.makefile("rb")

    def disconnect(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _send(self, payload: dict) -> None:
        if self._socket is None:
            raise NotConnectedError("scene client is not connected")
        self._socket.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    def _receive(self) -> dict:
        line = self._reader.readline()
        if not line:
            raise RuntimeError("piper scene server closed the connection")
        response = json.loads(line.decode("utf-8"))
        if not response.get("ok"):
            self._raise(response)
        return response

    def _raise(self, response: dict) -> None:
        error_type = response.get("error_type", "RuntimeError")
        message = response.get("error", "scene command failed")
        if error_type == "IKError":
            raise IKError(message)
        if error_type == "ValueError":
            raise ValueError(message)
        if error_type == "BackendUnavailableError":
            raise BackendUnavailableError(message)
        raise RuntimeError(message)

    def scene_info(self) -> dict:
        self._send({"cmd": "scene_info"})
        return self._receive()

    def state(self) -> RobotState:
        self._send({"cmd": "state"})
        return _state_from_dict(self._receive()["state"])

    def move_joints(self, joints) -> None:
        self._send({"cmd": "move_joints", "joints": [float(v) for v in joints]})
        self._receive()

    def move_p(self, pose: Pose) -> None:
        self._send({"cmd": "move_p", "position": list(pose.position),
                    "quaternion": list(pose.quaternion)})
        self._receive()

    def gripper(self, width: float, effort: float | None = None) -> None:
        self._send({"cmd": "gripper", "width": float(width), "effort": effort})
        self._receive()

    def stop(self) -> None:
        self._send({"cmd": "stop"})
        self._receive()

    def wait_until_idle(self, timeout: float = 10.0) -> None:
        """The GUI owns stepping; poll measured motion without holding its lock."""
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        deadline = time.monotonic() + timeout
        while self.state().moving:
            if time.monotonic() >= deadline:
                raise TimeoutError("Shared-scene motion did not settle before the timeout")
            time.sleep(0.01)

    def read(self, width: int = 1280, height: int = 720) -> RGBDFrame:
        """Capture one aligned RGB-D frame from the shared scene."""
        self._send({"cmd": "camera", "width": width, "height": height})
        header = self._receive()
        color = np.frombuffer(self._reader.read(header["color_shape"][0] * header["color_shape"][1] * 3),
                              dtype=np.uint8).reshape(header["color_shape"])
        depth = np.frombuffer(self._reader.read(header["depth_shape"][0] * header["depth_shape"][1] * 4),
                              dtype=np.float32).reshape(header["depth_shape"])
        intr = CameraIntrinsics(header["intrinsics"]["width"], header["intrinsics"]["height"],
                                header["intrinsics"]["fx"], header["intrinsics"]["fy"],
                                header["intrinsics"]["cx"], header["intrinsics"]["cy"])
        info = header.get("extrinsics")
        extrinsics = (CameraExtrinsics(tuple(info["rotation"]), tuple(info["translation"]),
                                       info["reference_frame"], info["camera_frame"], info["timestamp"])
                      if info is not None else None)
        return RGBDFrame(color, depth, header["timestamp"], header["frame_id"], intr,
                         header["depth_scale"], extrinsics)
