import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_control import Pose
from robot_control import Robot
from robot_control.backends.mujoco import MujocoBackend
from robot_control.errors import IKError
from robot_control.scene import SceneClient, SceneServer


@pytest.fixture
def scene():
    # AF_UNIX paths are length-limited on macOS, so keep the socket short.
    socket_path = Path(tempfile.gettempdir()) / f"piper_test_{id(object())}.sock"
    backend = MujocoBackend()
    backend.connect()
    server = SceneServer(backend, socket_path=socket_path)
    server.start()
    yield server, socket_path
    server.stop()
    backend.disconnect()


def _client(socket_path):
    client = SceneClient(socket_path=socket_path)
    client.connect()
    return client


def test_client_round_trip_joints_gripper_and_pose(scene):
    server, socket_path = scene
    client = _client(socket_path)
    try:
        target = [0.1, 0.5, -0.5, 0.0, 0.0, 0.0]
        client.move_joints(target)
        assert client.state().moving
        np.testing.assert_array_equal(server.backend.data.qpos[server.backend._arm_qpos], np.zeros(6))
        with server.lock:
            server.backend.wait_until_idle()
        assert np.max(np.abs(client.state().joints.positions - target)) < 0.02

        client.gripper(0.03)
        with server.lock:
            server.backend.wait_until_idle()
        assert client.state().joints.gripper == pytest.approx(0.03, abs=5e-4)

        # Generate a reachable pose from the active model, then solve from a
        # nearby seed. The old model's hard-coded home position is no longer valid.
        target_pose = server.backend.ik.forward([0.2, 0.8, -1.2, 0.2, -0.3, 0.4])
        before = server.backend.data.qpos.copy()
        client.move_p(target_pose)
        np.testing.assert_array_equal(server.backend.data.qpos, before)
        with server.lock:
            server.backend.wait_until_idle()
        pose = client.state().pose
        assert pose is not None
        assert np.linalg.norm(np.asarray(pose.position) - np.asarray(target_pose.position)) < 0.001
    finally:
        client.disconnect()


def test_client_propagates_ik_error(scene):
    _, socket_path = scene
    client = _client(socket_path)
    try:
        with pytest.raises(IKError):
            client.move_p(Pose((10.0, 0.0, 10.0), (1.0, 0.0, 0.0, 0.0)))
    finally:
        client.disconnect()


def test_client_camera_round_trip(scene):
    _, socket_path = scene
    client = _client(socket_path)
    try:
        frame = client.read()
        assert frame.extrinsics is not None
        assert frame.extrinsics.reference_frame == "base_link"
        assert frame.extrinsics.camera_frame == frame.frame_id
        assert frame.extrinsics.timestamp == frame.timestamp
        assert frame.color.shape == (720, 1280, 3)
        assert frame.color.dtype == np.uint8
        assert frame.depth.shape == (720, 1280)
        assert frame.depth.dtype == np.float32
        assert frame.depth_scale == 1.0
        assert np.isfinite(frame.depth).all()
    finally:
        client.disconnect()


def test_disconnect_lets_new_client_reconnect(scene):
    _, socket_path = scene
    first = _client(socket_path)
    first.move_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    first.disconnect()
    second = _client(socket_path)
    try:
        assert second.state().connected
    finally:
        second.disconnect()


def test_explicit_scene_must_match_running_server(scene, tmp_path):
    server, socket_path = scene
    matching = Robot.connect("mujoco", {"socket_path": socket_path, "scene": server.backend.scene_path})
    try:
        assert isinstance(matching._backend, SceneClient)
        assert matching.state().connected
    finally:
        matching.disconnect()
    different = tmp_path / "different.xml"
    different.write_text('<mujoco><worldbody><body name="piper_mount" pos="1 0 0"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match="restart the GUI with --scene"):
        Robot.connect("mujoco", {"socket_path": socket_path, "scene": different})
