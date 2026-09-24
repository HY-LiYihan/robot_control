"""Local Franka FCI control through pylibfranka (no HTTP service)."""

from __future__ import annotations

import ctypes
import importlib
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np

from ..api.types import JointState, Pose, RobotState
from ..errors import BackendUnavailableError, NotConnectedError


def _bindings():
    libraries = (
        Path(__file__).resolve().parents[3] / "vendor/libfranka/libfranka.so.0.13.5",
        Path("/usr/local/lib/libfranka.so.0.13"),
        Path("/usr/lib/libfranka.so.0.13"),
    )
    for library in libraries:
        if library.exists():
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue
    try:
        return importlib.import_module("pylibfranka")
    except (ImportError, OSError) as exc:
        raise BackendUnavailableError("Install matching pylibfranka and libfranka on the Linux control PC") from exc


def _ready(state: object) -> None:
    mode = str(getattr(state, "robot_mode", "unknown"))
    errors = getattr(state, "current_errors", None)
    last_errors = getattr(state, "last_motion_errors", None)
    if not mode.endswith("Idle"):
        raise BackendUnavailableError(f"robot must be Idle; current mode is {mode}")
    if bool(errors) or bool(last_errors):
        raise BackendUnavailableError(f"robot errors present: current={errors}; last_motion={last_errors}")


def _matrix(state: object, commanded: bool = False) -> list[float]:
    if commanded:
        candidate = getattr(state, "O_T_EE_c", None)
        if candidate is not None:
            values = [float(value) for value in candidate]
            if len(values) == 16 and all(math.isfinite(value) for value in values):
                return values
    return [float(value) for value in state.O_T_EE]


def _xyz(matrix: Sequence[float]) -> tuple[float, float, float]:
    return float(matrix[12]), float(matrix[13]), float(matrix[14])


def _normalize(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    if len(quaternion) != 4 or not all(math.isfinite(float(value)) for value in quaternion):
        raise ValueError("quaternion must contain four finite values")
    length = math.sqrt(sum(float(value) ** 2 for value in quaternion))
    if length < 1e-12:
        raise ValueError("quaternion norm must be nonzero")
    return tuple(float(value) / length for value in quaternion)


def _quaternion(matrix: Sequence[float]) -> tuple[float, float, float, float]:
    r00, r01, r02 = matrix[0], matrix[4], matrix[8]
    r10, r11, r12 = matrix[1], matrix[5], matrix[9]
    r20, r21, r22 = matrix[2], matrix[6], matrix[10]
    trace = r00 + r11 + r22
    if trace > 0:
        scale = math.sqrt(trace + 1) * 2
        result = ((r21 - r12) / scale, (r02 - r20) / scale, (r10 - r01) / scale, scale / 4)
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1 + r00 - r11 - r22) * 2
        result = (scale / 4, (r01 + r10) / scale, (r02 + r20) / scale, (r21 - r12) / scale)
    elif r11 > r22:
        scale = math.sqrt(1 + r11 - r00 - r22) * 2
        result = ((r01 + r10) / scale, scale / 4, (r12 + r21) / scale, (r02 - r20) / scale)
    else:
        scale = math.sqrt(1 + r22 - r00 - r11) * 2
        result = ((r02 + r20) / scale, (r12 + r21) / scale, scale / 4, (r10 - r01) / scale)
    return _normalize(result)


def _angle(first: Sequence[float], second: Sequence[float]) -> float:
    dot = abs(sum(left * right for left, right in zip(first, second)))
    return 2 * math.acos(min(1.0, max(-1.0, dot)))


def _slerp(first: Sequence[float], second: Sequence[float], fraction: float) -> tuple[float, ...]:
    start, end = _normalize(first), _normalize(second)
    dot = sum(left * right for left, right in zip(start, end))
    if dot < 0:
        end = tuple(-value for value in end)
        dot = -dot
    if dot > 0.9995:
        return _normalize(tuple(left + fraction * (right - left) for left, right in zip(start, end)))
    angle = math.acos(min(1.0, dot))
    scale = math.sin(angle)
    return tuple((math.sin((1 - fraction) * angle) * left + math.sin(fraction * angle) * right) / scale
                 for left, right in zip(start, end))


def _pose_command(bindings, position: Sequence[float], quaternion: Sequence[float]):
    x, y, z, w = _normalize(quaternion)
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return bindings.CartesianPose([
        rotation[0][0], rotation[1][0], rotation[2][0], 0.0,
        rotation[0][1], rotation[1][1], rotation[2][1], 0.0,
        rotation[0][2], rotation[1][2], rotation[2][2], 0.0,
        *position, 1.0,
    ])


@contextmanager
def _realtime(priority: int):
    if priority <= 0 or not hasattr(os, "sched_setscheduler"):
        yield
        return
    policy = os.sched_getscheduler(0)
    previous = os.sched_getparam(0).sched_priority
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except OSError as exc:
        raise BackendUnavailableError("Cannot enable FIFO scheduling; configure rtprio or set FRANKA_RT_PRIORITY=0 for diagnostics") from exc
    try:
        yield
    finally:
        os.sched_setscheduler(0, policy, os.sched_param(previous))


class FrankaDirectBackend:
    def __init__(self, robot_ip: str | None = None, motion_duration_s: float | None = None,
                 gripper_speed_m_s: float = 0.05, rt_priority: int | None = None):
        self.robot_ip = robot_ip or os.environ.get("FRANKA_ROBOT_IP", "192.168.1.6")
        self.motion_duration_s = float(motion_duration_s if motion_duration_s is not None
                                       else os.environ.get("FRANKA_MOVE_DURATION_S", "4.0"))
        self.gripper_speed_m_s = float(gripper_speed_m_s)
        self.rt_priority = int(rt_priority if rt_priority is not None
                               else os.environ.get("FRANKA_RT_PRIORITY", "80"))
        if not math.isfinite(self.motion_duration_s) or self.motion_duration_s <= 0:
            raise ValueError("motion duration must be positive and finite")
        if not math.isfinite(self.gripper_speed_m_s) or not 0 < self.gripper_speed_m_s <= 0.2:
            raise ValueError("gripper speed must be in (0, 0.2] m/s")
        self._bindings = None
        self._robot = None
        self._gripper = None

    def connect(self) -> None:
        self._bindings = _bindings()
        self._robot = self._bindings.Robot(self.robot_ip)

    def disconnect(self) -> None:
        self._robot = None
        self._gripper = None
        self._bindings = None

    def _require(self):
        if self._robot is None:
            raise NotConnectedError("Franka direct backend is not connected")
        return self._robot

    def _hand(self):
        self._require()
        if self._gripper is None:
            self._gripper = self._bindings.Gripper(self.robot_ip)
        return self._gripper

    def state(self) -> RobotState:
        measured = self._require().read_once()
        matrix = _matrix(measured)
        x, y, z, w = _quaternion(matrix)
        hand = self._hand().read_once()
        mode = str(measured.robot_mode)
        errors = [str(error) for error in (getattr(measured, "current_errors", None),
                                           getattr(measured, "last_motion_errors", None)) if bool(error)]
        stamp = time.time()
        return RobotState(True, not mode.endswith("Idle"),
                          JointState(np.asarray(measured.q, dtype=float), np.asarray(measured.dq, dtype=float),
                                     gripper=float(hand.width), timestamp=stamp),
                          Pose(_xyz(matrix), (w, x, y, z)), "; ".join(errors) or None, stamp)

    def _run(self, control, initialize, command_at):
        elapsed = 0.0
        first = True
        with _realtime(self.rt_priority):
            while True:
                active_state, period = control.readOnce()
                if first:
                    initialize(active_state)
                    progress = 0.0
                    first = False
                else:
                    elapsed += period.to_sec()
                    progress = min(elapsed / self.motion_duration_s, 1.0)
                blend = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
                command = command_at(blend)
                command.motion_finished = progress >= 1.0
                control.writeOnce(command)
                if command.motion_finished:
                    return

    def move_joints(self, joints: Sequence[float]) -> None:
        target = np.asarray(joints, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("move_joints requires seven finite joint values in radians")
        robot = self._require()
        before = robot.read_once()
        _ready(before)
        current = robot.read_once()
        _ready(current)
        if max(abs(float(left) - float(right)) for left, right in zip(before.q, current.q)) > 0.02:
            raise BackendUnavailableError("joint state changed by more than 0.02 rad; re-read before moving")
        control = robot.start_joint_position_control(self._bindings.ControllerMode.CartesianImpedance)
        start = []

        def initialize(active_state):
            start[:] = [float(value) for value in getattr(active_state, "q_d", active_state.q)]
            if max(abs(left - float(right)) for left, right in zip(start, current.q)) > 0.02:
                raise BackendUnavailableError("joint state changed by more than 0.02 rad after control started")

        def command_at(blend):
            return self._bindings.JointPositions([left + blend * (right - left)
                                                   for left, right in zip(start, target)])

        self._run(control, initialize, command_at)

    def move_p(self, pose: Pose) -> None:
        target_position = tuple(float(value) for value in pose.position)
        if not all(math.isfinite(value) for value in target_position):
            raise ValueError("target position must be finite")
        target_quaternion = _normalize((pose.quaternion[1], pose.quaternion[2],
                                        pose.quaternion[3], pose.quaternion[0]))
        robot = self._require()
        before = robot.read_once()
        _ready(before)
        before_actual = _matrix(before)
        before_command = _matrix(before, commanded=True)
        if _angle(_quaternion(before_command), target_quaternion) > math.radians(10):
            raise ValueError("target orientation differs by more than 10 degrees")
        current = robot.read_once()
        _ready(current)
        current_actual = _matrix(current)
        if math.dist(_xyz(before_actual), _xyz(current_actual)) > 0.005:
            raise BackendUnavailableError("pose changed by more than 5 mm; re-read before moving")
        if _angle(_quaternion(before_actual), _quaternion(current_actual)) > math.radians(2):
            raise BackendUnavailableError("orientation changed by more than 2 degrees; re-read before moving")
        control = robot.start_cartesian_pose_control(self._bindings.ControllerMode.JointImpedance)
        start = {}

        def initialize(active_state):
            matrix = _matrix(active_state, commanded=True)
            quaternion = _quaternion(matrix)
            if _angle(quaternion, target_quaternion) > math.radians(10):
                raise BackendUnavailableError("active control start orientation differs from target by more than 10 degrees")
            start.update(matrix=matrix, position=_xyz(matrix), quaternion=quaternion)

        def command_at(blend):
            if blend == 0.0:
                return self._bindings.CartesianPose(start["matrix"])
            position = tuple(left + blend * (right - left)
                             for left, right in zip(start["position"], target_position))
            quaternion = _slerp(start["quaternion"], target_quaternion, blend)
            return _pose_command(self._bindings, position, quaternion)

        self._run(control, initialize, command_at)

    def gripper(self, width: float, effort: float | None = None) -> None:
        if effort is not None:
            raise ValueError("Franka gripper move does not accept effort")
        hand = self._hand()
        state = hand.read_once()
        if not math.isfinite(width) or not 0 <= width <= float(state.max_width):
            raise ValueError(f"gripper width must be between 0 and {state.max_width} m")
        if width > float(hand.read_once().max_width):
            raise ValueError("target width exceeds latest gripper limit")
        if not hand.move(float(width), self.gripper_speed_m_s):
            raise BackendUnavailableError("Franka gripper move did not succeed")

    def stop(self) -> None:
        robot = self._require()
        method = getattr(robot, "stop", None)
        if method is None:
            raise NotImplementedError("This pylibfranka build does not expose Robot.stop(); use the physical stop if needed")
        method()
