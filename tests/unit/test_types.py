import numpy as np
import pytest
from robot_control.api.types import JointState, Pose
from robot_control.sensors.frame import CameraIntrinsics, RGBDFrame


def test_pose_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        Pose((0, 0, 0), (0, 0, 0, 0))


def test_joint_state_has_six_joints():
    state = JointState(np.zeros(6))
    assert state.positions.shape == (6,)
    with pytest.raises(ValueError):
        JointState(np.zeros(5))


def test_rgbd_contract():
    color = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = np.zeros((720, 1280), dtype=np.uint16)
    frame = RGBDFrame(color, depth, 0.0, "camera", CameraIntrinsics(1280, 720, 1, 1, 640, 360), 0.001)
    assert frame.color.shape == (720, 1280, 3)
