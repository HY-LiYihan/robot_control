import time

import numpy as np
import pytest

from robot_control.api.types import JointState, Pose, RobotState
from robot_control.errors import BackendUnavailableError
from robot_control.sensors import service
from robot_control.sensors.frame import CameraIntrinsics, RGBDFrame
from robot_control.sensors.extrinsics import fr3_link7_to_color_optical, pose_matrix


def test_real_camera_without_arm(monkeypatch):
    calls = []

    class Camera:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def connect(self):
            calls.append("connect")

        def read(self):
            calls.append("read")
            return RGBDFrame(np.zeros((2, 2, 3), dtype=np.uint8),
                             np.ones((2, 2), dtype=np.uint16), time.time(),
                             "d435i_color_optical_frame", CameraIntrinsics(2, 2, 1, 1, 1, 1), 0.001)

        def disconnect(self):
            calls.append("disconnect")

    monkeypatch.setattr(service, "RealSenseCamera", Camera)
    frame = service.CameraService("real", include_extrinsics=False, width=2, height=2).capture()
    assert frame.extrinsics is None
    assert calls == [("init", {"width": 2, "height": 2, "frame_id": "d435i_color_optical_frame"}),
                     "connect", "read", "disconnect"]
    with pytest.raises(ValueError, match="connected robot"):
        service.CameraService("real").capture()
    with pytest.raises(ValueError, match="connected robot"):
        service.CameraService("mujoco", include_extrinsics=False).capture()


def test_real_camera_disconnects_when_feedback_is_stale(monkeypatch):
    calls = []

    class Camera:
        def __init__(self, **kwargs):
            pass

        def connect(self):
            calls.append("connect")

        def read(self):
            calls.append("read")
            return RGBDFrame(np.zeros((2, 2, 3), dtype=np.uint8),
                             np.ones((2, 2), dtype=np.uint16), time.time() - 5,
                             "d435i_color_optical_frame", CameraIntrinsics(2, 2, 1, 1, 1, 1), 0.001)

        def disconnect(self):
            calls.append("disconnect")

    class Arm:
        def state(self):
            return type("State", (), {"pose": Pose((0, 0, 0)), "timestamp": time.time()})()

    monkeypatch.setattr(service, "RealSenseCamera", Camera)
    with pytest.raises(BackendUnavailableError, match="not synchronized"):
        service.CameraService("twin", arm=Arm()).capture()
    assert calls == ["connect", "read", "disconnect"]


@pytest.mark.parametrize("backend", ["real", "twin"])
def test_fr3_real_camera_extrinsics_follow_measured_joints(monkeypatch, backend):
    pytest.importorskip("pinocchio")
    joints = np.array([0.1, -0.2, 0.15, -1.4, 0.2, 1.5, -0.7])
    calls = []

    class Camera:
        def __init__(self, **kwargs):
            pass

        def connect(self):
            calls.append("connect")

        def read(self):
            return RGBDFrame(np.zeros((2, 2, 3), dtype=np.uint8),
                             np.zeros((2, 2), dtype=np.uint16), time.time(),
                             "d435i_color_optical_frame", CameraIntrinsics(2, 2, 1, 1, 1, 1), .001)

        def disconnect(self):
            calls.append("disconnect")

    class Arm:
        def state(self):
            calls.append("state")
            return RobotState(True, False, JointState(joints), Pose((0, 0, 0)))

    monkeypatch.setattr(service, "RealSenseCamera", Camera)
    frame = service.CameraService(backend, "franka_fr3", Arm(), width=2, height=2).capture()
    expected = pose_matrix(service._fr3_kinematics().forward(joints)) @ fr3_link7_to_color_optical()
    assert frame.extrinsics.reference_frame == "fr3_link0"
    assert frame.extrinsics.camera_frame == frame.frame_id
    np.testing.assert_allclose(frame.extrinsics.rotation, expected[:3, :3].flat)
    np.testing.assert_allclose(frame.extrinsics.translation, expected[:3, 3])
    assert calls == ["connect", "disconnect", "state"]
