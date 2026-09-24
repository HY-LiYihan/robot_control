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
        assert calls[2][0] == "motion_ctrl_2"
        assert calls[3][0] == "joint_ctrl"
        assert backend.gripper_width() == pytest.approx(0.04)
    finally:
        backend.disconnect()
    assert calls[-1] == ("disconnect",)
