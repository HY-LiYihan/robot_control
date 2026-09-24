from unittest.mock import patch
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot_control.backends.piper_model import ASSET_ROOT
from robot_control.backends.real import RealBackend
from robot_control.config import PIPER_INITIAL_JOINTS_RAD
from robot_control.errors import BackendUnavailableError
from robot_control.kinematics.ik import PinocchioIK


def test_real_backend_import_is_lazy():
    backend = RealBackend()
    with patch.dict("sys.modules", {"piper_sdk": None}):
        try:
            backend.connect()
        except Exception as exc:
            assert "piper_sdk" in str(exc)


def test_real_backend_state_includes_feedback_pose():
    class JointFeedback:
        joint_1 = 1000
        joint_2 = 0
        joint_3 = 0
        joint_4 = 0
        joint_5 = 0
        joint_6 = 0

    class Message:
        time_stamp = time.time()
        joint_state = JointFeedback()

    backend = RealBackend()
    backend._sdk = type("Sdk", (), {"GetArmJointMsgs": lambda self: Message()})()
    backend._fk = PinocchioIK(ASSET_ROOT / "piper/urdf/piper_description.urdf")
    backend._connected = True
    state = backend.state()
    assert state.pose is not None
    np.testing.assert_allclose(state.joints.positions[0], np.deg2rad(1.0))
    np.testing.assert_allclose(state.pose.position, backend._fk.forward(state.joints.positions).position)


def test_real_backend_rejects_missing_feedback():
    backend = RealBackend()
    backend._sdk = type("Sdk", (), {"GetArmJointMsgs": lambda self: type("Message", (), {"time_stamp": 0})()})()
    backend._connected = True
    with pytest.raises(BackendUnavailableError, match="fresh"):
        backend.state()


def test_passive_real_connection_and_gripper_feedback(monkeypatch):
    calls = []

    class FakeSdk:
        def __init__(self, **kwargs):
            calls.append(("create", kwargs["can_name"]))

        def ConnectPort(self, *, piper_init=True):
            calls.append(("connect", piper_init))

        def DisconnectPort(self):
            calls.append(("disconnect",))

        def MotionCtrl_2(self, *values):
            calls.append(("motion_ctrl_2", values))

        def JointCtrl(self, *values):
            calls.append(("joint_ctrl", values))

        def GetArmJointMsgs(self):
            values = np.rint(np.rad2deg(PIPER_INITIAL_JOINTS_RAD) * 1000).astype(int)
            feedback = SimpleNamespace(**{
                f"joint_{index}": int(value) for index, value in enumerate(values, start=1)
            })
            return SimpleNamespace(time_stamp=time.time(), joint_state=feedback)

        def GetArmGripperMsgs(self):
            return SimpleNamespace(time_stamp=time.time(),
                                   gripper_state=SimpleNamespace(grippers_angle=40000))

    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface=FakeSdk))
    backend = RealBackend(can_name="can1")
    backend.connect(piper_init=False)
    try:
        assert calls[:2] == [("create", "can1"), ("connect", False)]
        assert len(calls) == 2
        assert backend.gripper_width() == pytest.approx(0.04)
    finally:
        backend.disconnect()
    assert calls[-1] == ("disconnect",)


def test_real_startup_power_cycles_and_reports_unreached_target(monkeypatch):
    calls = []

    class FakeSdk:
        def __init__(self, **kwargs):
            self.enabled = [False] * 6
            self.joints = np.zeros(6, dtype=int)
            self.feedback_reads = 0
            self.move_allowed = True

        def ConnectPort(self, *, piper_init=True):
            calls.append(("connect", piper_init))

        def DisconnectPort(self):
            calls.append(("disconnect",))

        def MotionCtrl_1(self, *values):
            calls.append(("reset", values))

        def GetArmJointMsgs(self):
            self.feedback_reads += 1
            feedback = SimpleNamespace(**{
                f"joint_{index}": int(value) for index, value in enumerate(self.joints, start=1)
            })
            timestamp = 0 if self.feedback_reads <= 2 else time.time()
            return SimpleNamespace(time_stamp=timestamp, joint_state=feedback)

        def GetArmLowSpdInfoMsgs(self):
            return SimpleNamespace(time_stamp=time.time())

        def GetArmStatus(self):
            return SimpleNamespace(time_stamp=time.time(), arm_status=0)

        def GetArmEnableStatus(self):
            return self.enabled.copy()

        def DisableArm(self, motor_num):
            calls.append(("disable", motor_num))
            self.enabled = [False] * 6

        def EnableArm(self, motor_num):
            calls.append(("enable", motor_num))
            self.enabled = [True] * 6

        def MotionCtrl_2(self, *values):
            calls.append(("mode", values))

        def JointCtrl(self, *values):
            calls.append(("joints", values))
            if self.move_allowed:
                self.joints = np.asarray(values)

        def EndPoseCtrl(self, *values):
            calls.append(("pose", values))

    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface=FakeSdk))
    backend = RealBackend(initial_motion_timeout_s=0.15)
    backend.connect()
    try:
        assert calls[:5] == [("connect", True), ("reset", (0x02, 0x00, 0x00)),
                             ("disable", 7), ("enable", 7),
                             ("mode", (0x01, 0x01, 30, 0x00))]
        np.testing.assert_allclose(backend.state().joints.positions, PIPER_INITIAL_JOINTS_RAD, atol=1e-5)
        assert backend.state().error is None
        backend._sdk.move_allowed = False
        backend.move_joints(np.zeros(6))
        state = backend.state()
        np.testing.assert_allclose(state.joints.positions, PIPER_INITIAL_JOINTS_RAD, atol=1e-5)
        assert state.moving and "not reached" in state.error
        backend._sdk.enabled = [False] * 6
        commands_before = len(calls)
        backend.move_joints(np.zeros(6))
        assert len(calls) == commands_before
        assert not backend.state().moving
        assert "not enabled" in backend.state().error
    finally:
        backend.disconnect()


def test_real_failed_enable_skips_home_but_keeps_measured_state(monkeypatch):
    calls = []

    class FakeSdk:
        def __init__(self, **kwargs):
            self.enabled = [False] * 6

        def ConnectPort(self, *, piper_init=True):
            pass

        def DisconnectPort(self):
            pass

        def GetArmJointMsgs(self):
            feedback = SimpleNamespace(**{f"joint_{index}": 0 for index in range(1, 7)})
            return SimpleNamespace(time_stamp=time.time(), joint_state=feedback)

        def GetArmLowSpdInfoMsgs(self):
            return SimpleNamespace(time_stamp=time.time())

        def GetArmStatus(self):
            return SimpleNamespace(time_stamp=time.time(), arm_status=0)

        def GetArmEnableStatus(self):
            return self.enabled.copy()

        def MotionCtrl_1(self, *values):
            calls.append(("reset", values))

        def DisableArm(self, motor_num):
            calls.append("disable")

        def EnableArm(self, motor_num):
            calls.append("enable")

        def JointCtrl(self, *values):
            calls.append("joints")

    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface=FakeSdk))
    backend = RealBackend(enable_timeout_s=0.12)
    backend.connect()
    try:
        assert calls == [("reset", (0x02, 0x00, 0x00)), "disable", "enable"]
        state = backend.state()
        np.testing.assert_array_equal(state.joints.positions, np.zeros(6))
        assert "not confirm motor enable" in state.error
        backend.move_joints(PIPER_INITIAL_JOINTS_RAD)
        assert calls == [("reset", (0x02, 0x00, 0x00)), "disable", "enable"]
        assert "not enabled" in backend.state().error
    finally:
        backend.disconnect()


def test_real_joint_control_repeats_mode_and_command_heartbeat(monkeypatch):
    calls = []

    class FakeSdk:
        def __init__(self, **kwargs):
            self.joints = np.zeros(6, dtype=int)
            self.enabled = [True] * 6

        def ConnectPort(self, *, piper_init=True):
            pass

        def DisconnectPort(self):
            pass

        def GetArmJointMsgs(self):
            feedback = SimpleNamespace(**{
                f"joint_{index}": int(value) for index, value in enumerate(self.joints, start=1)
            })
            return SimpleNamespace(time_stamp=time.time(), joint_state=feedback)

        def GetArmLowSpdInfoMsgs(self):
            return SimpleNamespace(time_stamp=time.time())

        def GetArmEnableStatus(self):
            return self.enabled.copy()

        def GetArmStatus(self):
            return SimpleNamespace(time_stamp=time.time(), arm_status=0)

        def MotionCtrl_2(self, *values):
            calls.append(("mode", values))

        def JointCtrl(self, *values):
            calls.append(("joints", values))

    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface=FakeSdk))
    backend = RealBackend(initial_motion_timeout_s=0.07)
    backend.connect(piper_init=False)
    try:
        backend.move_joints(np.ones(6))
        modes = [entry for entry in calls if entry[0] == "mode"]
        joints = [entry for entry in calls if entry[0] == "joints"]
        assert len(modes) >= 3
        assert len(modes) == len(joints)
        assert all(entry[1] == (0x01, 0x01, 30, 0x00) for entry in modes)
    finally:
        backend.disconnect()
