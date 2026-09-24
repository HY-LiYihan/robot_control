from __future__ import annotations

import time
import sys
from pathlib import Path
import numpy as np
from ..api.types import JointState, Pose, RobotState
from ..config import PIPER_INITIAL_JOINTS_RAD
from ..errors import BackendUnavailableError, NotConnectedError
from ..kinematics.ik import PinocchioIK
from .piper_model import ASSET_ROOT


PIPER_COMMAND_HEARTBEAT_S = 0.02


class RealBackend:
    """Thin adapter around the pinned piper_sdk; imports the SDK only on connect."""

    def __init__(self, can_name: str = "can0", judge_flag: bool = False,
                 initial_motion_speed: int = 30, initial_motion_timeout_s: float = 15.0,
                 auto_initialize: bool = True, feedback_timeout_s: float = 5.0,
                 enable_timeout_s: float = 5.0, **kwargs):
        self.can_name, self.judge_flag, self.kwargs = can_name, judge_flag, kwargs
        self.initial_motion_speed = int(initial_motion_speed)
        self.initial_motion_timeout_s = float(initial_motion_timeout_s)
        self.auto_initialize = bool(auto_initialize)
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.enable_timeout_s = float(enable_timeout_s)
        if not 0 <= self.initial_motion_speed <= 100:
            raise ValueError("initial_motion_speed must be between 0 and 100")
        if not np.isfinite(self.initial_motion_timeout_s) or self.initial_motion_timeout_s <= 0:
            raise ValueError("initial_motion_timeout_s must be positive and finite")
        if not np.isfinite(self.feedback_timeout_s) or self.feedback_timeout_s <= 0:
            raise ValueError("feedback_timeout_s must be positive and finite")
        if not np.isfinite(self.enable_timeout_s) or self.enable_timeout_s <= 0:
            raise ValueError("enable_timeout_s must be positive and finite")
        self._sdk = None
        self._fk = None
        self._connected = False
        self._target_joints = None
        self._target_pose = None
        self._last_error = None

    def connect(self, *, piper_init: bool = True) -> None:
        try:
            from piper_sdk import C_PiperInterface
        except ImportError as exc:
            vendor = Path(__file__).parents[3] / "vendor" / "piper_sdk"
            if vendor.exists():
                sys.path.insert(0, str(vendor))
                try:
                    from piper_sdk import C_PiperInterface
                except ImportError:
                    raise BackendUnavailableError("Install the pinned vendor/piper_sdk package") from exc
            else:
                raise BackendUnavailableError("Install the pinned vendor/piper_sdk package") from exc
        self._fk = PinocchioIK(ASSET_ROOT / "piper/urdf/piper_description.urdf")
        self._sdk = C_PiperInterface(can_name=self.can_name, judge_flag=self.judge_flag,
                                     can_auto_init=True, **self.kwargs)
        self._sdk.ConnectPort(piper_init=piper_init)
        self._connected = True
        if self.auto_initialize and piper_init:
            try:
                if self._reset_faults():
                    self._wait_for_feedback()
                    self._power_cycle()
                    self._move_to_initial_position()
            except Exception:
                self.disconnect()
                raise

    def _wait_for_feedback(self) -> RobotState:
        deadline = time.monotonic() + self.feedback_timeout_s
        while True:
            try:
                return self.state()
            except BackendUnavailableError as exc:
                if time.monotonic() >= deadline:
                    raise BackendUnavailableError(
                        f"No fresh Piper joint feedback on {self.can_name} within {self.feedback_timeout_s:g} seconds"
                    ) from exc
                time.sleep(0.05)

    def _arm_enable_status(self) -> tuple[bool, ...] | None:
        status = getattr(self._sdk, "GetArmEnableStatus", None)
        low_speed = getattr(self._sdk, "GetArmLowSpdInfoMsgs", None)
        if status is None or low_speed is None:
            return None
        timestamp = float(low_speed().time_stamp)
        if timestamp <= 0 or time.time() - timestamp > 1.0:
            return None
        enabled = tuple(bool(value) for value in status())
        if len(enabled) != 6:
            return None
        return enabled

    def _arm_enabled(self) -> bool | None:
        status = self._arm_enable_status()
        return None if status is None else all(status)

    def _arm_status(self):
        method = getattr(self._sdk, "GetArmStatus", None)
        if method is None:
            return None
        status = method()
        timestamp = float(getattr(status, "time_stamp", 0.0))
        if timestamp <= 0 or time.time() - timestamp > 1.0:
            return None
        return status

    def _arm_status_error(self) -> str | None:
        status = self._arm_status()
        if status is None:
            return None
        arm_status_value = getattr(status, "arm_status", 0)
        arm_status = int(getattr(arm_status_value, "arm_status", arm_status_value))
        if arm_status == 0:
            return None
        return f"Piper arm_status={arm_status}; command was rejected or stopped"

    def _reset_faults(self) -> bool:
        reset = getattr(self._sdk, "MotionCtrl_1", None)
        if reset is None:
            raise BackendUnavailableError("piper_sdk does not provide the official Piper reset command")
        reset(0x02, 0x00, 0x00)
        deadline = time.monotonic() + self.enable_timeout_s
        while time.monotonic() < deadline:
            error = self._arm_status_error()
            if error is None and self._arm_status() is not None:
                return True
            time.sleep(0.05)
        self._last_error = "Piper reset did not clear the arm fault; home command was not sent"
        return False

    def _wait_for_enable(self, enabled: bool) -> bool:
        deadline = time.monotonic() + self.enable_timeout_s
        while time.monotonic() < deadline:
            status = self._arm_enable_status()
            if status is not None and (all(status) if enabled else not any(status)):
                return True
            time.sleep(0.05)
        return False

    def _power_cycle(self) -> None:
        disable = getattr(self._sdk, "DisableArm", None)
        enable = getattr(self._sdk, "EnableArm", None)
        if disable is None or enable is None:
            raise BackendUnavailableError("Piper motor enable feedback or enable/disable commands are unavailable")
        deadline = time.monotonic() + self.enable_timeout_s
        while self._arm_enable_status() is None:
            if time.monotonic() >= deadline:
                raise BackendUnavailableError("No fresh Piper motor enable feedback; home command was not sent")
            time.sleep(0.05)
        disable(7)
        if not self._wait_for_enable(False):
            self._last_error = "Piper did not confirm motor disable; home command was not sent"
            return
        enable(7)
        if not self._wait_for_enable(True):
            self._last_error = "Piper did not confirm motor enable; home command was not sent"

    def _move_to_initial_position(self) -> None:
        if self._last_error is not None:
            return
        target = np.asarray(PIPER_INITIAL_JOINTS_RAD, dtype=float)
        target_sdk = np.rint(np.rad2deg(target) * 1000).astype(int)
        mode = getattr(self._sdk, "MotionCtrl_2", None)
        command = getattr(self._sdk, "JointCtrl", None)
        if mode is None or command is None:
            raise BackendUnavailableError("piper_sdk does not provide joint initialization commands")
        self._target_joints = target
        self._wait_for_target((0x01, 0x01, self.initial_motion_speed, 0x00),
                              mode, command, target_sdk.tolist())

    def _wait_for_target(self, mode_values, mode, command, command_values) -> None:
        deadline = time.monotonic() + self.initial_motion_timeout_s
        next_heartbeat = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_heartbeat:
                mode(*mode_values)
                command(*command_values)
                next_heartbeat = now + PIPER_COMMAND_HEARTBEAT_S
            if self._arm_status_error() is not None:
                self._last_error = self._arm_status_error()
                return
            if self._arm_enabled() is False:
                self._last_error = "Piper motors are not enabled; target was not reached"
                return
            try:
                if not self.state().moving:
                    self._last_error = None
                    return
            except BackendUnavailableError:
                pass
            time.sleep(min(PIPER_COMMAND_HEARTBEAT_S, max(0.0, deadline - time.monotonic())))
        self._last_error = "Piper target was not reached before the motion timeout"

    def gripper_width(self) -> float | None:
        self._require()
        feedback = self._sdk.GetArmGripperMsgs()
        timestamp = float(feedback.time_stamp)
        if timestamp <= 0 or time.time() - timestamp > 1.0:
            return None
        width = float(feedback.gripper_state.grippers_angle) / 1_000_000
        if not np.isfinite(width):
            raise BackendUnavailableError("Piper gripper feedback is not finite")
        return width

    def _require(self):
        if not self._connected or self._sdk is None:
            raise NotConnectedError("real backend is not connected")

    def disconnect(self):
        if self._sdk is not None:
            self._sdk.DisconnectPort()
        self._connected = False
        self._fk = None
        self._target_joints = None
        self._target_pose = None
        self._last_error = None

    def state(self) -> RobotState:
        self._require()
        msg = self._sdk.GetArmJointMsgs()
        timestamp = float(msg.time_stamp)
        if timestamp <= 0 or time.time() - timestamp > 1.0:
            raise BackendUnavailableError("No fresh Piper joint feedback is available")
        names = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
        values = [getattr(msg.joint_state, name) for name in names]
        joints = np.deg2rad(np.asarray(values, dtype=float) * 0.001)
        if not np.isfinite(joints).all():
            raise BackendUnavailableError("Piper joint feedback is not finite")
        pose = self._fk.forward(joints)
        status_error = self._arm_status_error()
        if status_error is not None:
            self._last_error = status_error
        enabled = self._arm_enabled()
        joint_pending = (self._target_joints is not None
                         and np.max(np.abs(joints - self._target_joints)) > 0.02)
        pose_pending = False
        if self._target_pose is not None:
            orientation = abs(float(np.dot(pose.quaternion, self._target_pose.quaternion)))
            pose_pending = (np.linalg.norm(np.asarray(pose.position) - self._target_pose.position) > 0.01
                            or orientation < np.cos(0.05 / 2))
        moving = bool((joint_pending or pose_pending) and enabled is not False and status_error is None)
        if (not moving and enabled is True and status_error is None
                and (self._target_joints is not None or self._target_pose is not None)):
            self._last_error = None
        return RobotState(True, moving, JointState(joints, timestamp=timestamp),
                          pose, self._last_error, timestamp)

    def move_joints(self, joints):
        self._require()
        values = np.asarray(joints, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("move_joints requires six finite joint values")
        status_error = self._arm_status_error()
        if status_error is not None:
            self._last_error = status_error
            return
        if self._arm_enabled() is not True:
            self._last_error = "Piper motors are not enabled; joint command was not sent"
            return
        method = getattr(self._sdk, "JointCtrl", None)
        if method is None:
            raise NotImplementedError("piper_sdk joint control method is not available")
        # Piper SDK JointCtrl uses 0.001 degrees, while the public API uses radians.
        self._target_pose = None
        self._target_joints = values
        self._last_error = None
        mode = getattr(self._sdk, "MotionCtrl_2", None)
        if mode is None:
            raise NotImplementedError("piper_sdk motion mode control is not available")
        self._wait_for_target((0x01, 0x01, self.initial_motion_speed, 0x00),
                              mode, method,
                              np.rint(np.rad2deg(values) * 1000).astype(int).tolist())

    def move_p(self, pose: Pose):
        self._require()
        status_error = self._arm_status_error()
        if status_error is not None:
            self._last_error = status_error
            return
        if self._arm_enabled() is not True:
            self._last_error = "Piper motors are not enabled; pose command was not sent"
            return
        method = getattr(self._sdk, "EndPoseCtrl", None)
        if method is None:
            raise NotImplementedError("piper_sdk EndPoseCtrl is not available in this SDK build")
        w, x, y, z = np.asarray(pose.quaternion, dtype=float)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        position = np.rint(np.asarray(pose.position) * 1_000_000).astype(int)
        angles = np.rint(np.rad2deg((roll, pitch, yaw)) * 1000).astype(int)
        mode = getattr(self._sdk, "MotionCtrl_2", None)
        if mode is None:
            raise NotImplementedError("piper_sdk motion mode control is not available")
        self._target_joints = None
        self._target_pose = pose
        self._last_error = None
        self._wait_for_target((0x01, 0x00, self.initial_motion_speed, 0x00),
                              mode, method, [*position.tolist(), *angles.tolist()])

    def gripper(self, width: float, effort: float | None = None):
        self._require()
        method = getattr(self._sdk, "GripperCtrl", None)
        if method is None:
            raise NotImplementedError("piper_sdk gripper control method is not available")
        if not 0.0 <= width <= 0.07:
            raise ValueError("gripper width must be between 0 and 0.07 metres")
        method(int(round(width * 1_000_000)), int(round((effort or 0.0) * 1000)), 0x01, 0)

    def stop(self):
        self._require()
        method = getattr(self._sdk, "MotionCtrl_1", None)
        if method is not None:
            method(0x01, 0x00, 0x00)
