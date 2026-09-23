from __future__ import annotations

import time
import sys
from pathlib import Path
import numpy as np
from ..api.types import JointState, Pose, RobotState
from ..errors import BackendUnavailableError, NotConnectedError
from ..kinematics.ik import PinocchioIK
from .piper_model import ASSET_ROOT


class RealBackend:
    """Thin adapter around the pinned piper_sdk; imports the SDK only on connect."""

    def __init__(self, can_name: str = "can0", judge_flag: bool = False, **kwargs):
        self.can_name, self.judge_flag, self.kwargs = can_name, judge_flag, kwargs
        self._sdk = None
        self._fk = None
        self._connected = False

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
        return RobotState(True, False, JointState(joints, timestamp=timestamp),
                          self._fk.forward(joints), timestamp=timestamp)

    def move_joints(self, joints):
        self._require()
        values = np.asarray(joints, dtype=float)
        if values.shape != (6,):
            raise ValueError("move_joints requires six joint values")
        method = getattr(self._sdk, "JointCtrl", None)
        if method is None:
            raise NotImplementedError("piper_sdk joint control method is not available")
        # Piper SDK JointCtrl uses 0.001 degrees, while the public API uses radians.
        method(*np.rint(np.rad2deg(values) * 1000).astype(int).tolist())

    def move_p(self, pose: Pose):
        self._require()
        method = getattr(self._sdk, "EndPoseCtrl", None)
        if method is None:
            raise NotImplementedError("piper_sdk EndPoseCtrl is not available in this SDK build")
        w, x, y, z = np.asarray(pose.quaternion, dtype=float)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        position = np.rint(np.asarray(pose.position) * 1_000_000).astype(int)
        angles = np.rint(np.rad2deg((roll, pitch, yaw)) * 1000).astype(int)
        method(*position.tolist(), *angles.tolist())

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
