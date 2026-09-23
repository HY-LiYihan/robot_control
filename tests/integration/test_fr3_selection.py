import numpy as np
import pytest
from typer.testing import CliRunner
from pathlib import Path
from uuid import uuid4

from robot_control import Robot
from robot_control.api.types import JointState
from robot_control import cli
from robot_control.cli import app
from robot_control.errors import BackendUnavailableError
from robot_control.scene import SceneClient, SceneServer, default_socket_path
from robot_control.fr3.scene_builder import validate_scene


runner = CliRunner()


def test_joint_count_and_real_backend_guard():
    assert JointState(np.zeros(6)).velocities.shape == (6,)
    assert JointState(np.zeros(7)).velocities.shape == (7,)
    with pytest.raises(ValueError, match="velocities"):
        JointState(np.zeros(7), np.zeros(6))
    with pytest.raises(BackendUnavailableError, match="not implemented"):
        Robot.connect("real", robot="franka_fr3")


def test_robot_alias_and_distinct_sockets(monkeypatch):
    monkeypatch.delenv("PIPER_SCENE_SOCKET", raising=False)
    monkeypatch.delenv("FR3_SCENE_SOCKET", raising=False)
    assert default_socket_path("piper") != default_socket_path("franka_fr3")
    with pytest.raises(ValueError, match="unknown robot"):
        Robot.connect(robot="other")
    robot = Robot.connect(robot="pepper", config={"socket_path": "/tmp/pepper_missing.sock"})
    try:
        assert robot.state().joints.positions.shape == (6,)
    finally:
        robot.disconnect()


def test_cli_fr3_joint_validation_before_connection():
    common = ["move-joints", "--j1", "0", "--j2", "0", "--j3", "0",
              "--j4", "-1.57", "--j5", "0", "--j6", "1.57"]
    missing = runner.invoke(app, common + ["--robot", "franka_fr3"])
    assert missing.exit_code == 2
    assert "--j7 is required" in missing.output
    extra = runner.invoke(app, common + ["--robot", "piper", "--j7", "0"])
    assert extra.exit_code == 2
    assert "--j7 is only valid" in extra.output


def test_fr3_gui_macos_reexec_and_scene(monkeypatch, tmp_path):
    scene = tmp_path / "fr3_scene.xml"
    scene.write_text('<mujoco><worldbody><body name="fr3_mount" pos="0 0 0"/></worldbody></mujoco>')
    calls = []
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.delenv("ROBOT_CONTROL_MUJOCO_GUI_REEXEC", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: calls.append(command))
    result = runner.invoke(app, ["run", "--robot", "franka_fr3", "--gui", "--scene", str(scene)])
    assert result.exit_code == 0, result.output
    assert calls[0][1:3] == ["-m", "robot_control.mujoco_gui"]
    assert calls[0][-4:] == ["--robot", "franka_fr3", "--scene", str(scene.resolve())]


def test_fr3_scene_mount_validation(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text('<mujoco><worldbody><body name="fr3_mount" pos="0 0 0"/></worldbody></mujoco>')
    assert validate_scene(path) == path.resolve()
    result = runner.invoke(app, ["run", "--robot", "franka_fr3", "--scene", str(path), "--steps", "1"])
    assert result.exit_code == 0, result.output
    path.write_text('<mujoco><worldbody><body name="piper_mount" pos="0 0 0"/></worldbody></mujoco>')
    invalid = runner.invoke(app, ["run", "--robot", "franka_fr3", "--scene", str(path)])
    assert invalid.exit_code == 2
    assert "fr3_mount" in invalid.output


def test_fr3_mujoco_and_shared_scene():
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    from robot_control.fr3.mujoco import MujocoBackend

    backend = MujocoBackend()
    backend.connect()
    expected_home = np.array([0, 0, 0, -1.57, 0, 1.57, .785])
    np.testing.assert_allclose(backend.state().joints.positions, expected_home, atol=1e-9)
    np.testing.assert_allclose(backend.data.ctrl[backend._arm_actuators], expected_home, atol=1e-9)
    assert backend.state().joints.gripper == pytest.approx(0.08)
    np.testing.assert_allclose(backend.data.qpos[backend._finger_qpos], [0.04, 0.04])
    np.testing.assert_allclose(backend.data.ctrl[backend._finger_actuators], [0.04, 0.04])
    socket_path = Path("/tmp") / f"fr3-{uuid4().hex[:12]}.sock"
    server = SceneServer(backend, socket_path=socket_path, robot="franka_fr3")
    server.start()
    try:
        with pytest.raises(ValueError, match="--robot"):
            Robot.connect(config={"socket_path": socket_path})
        robot = Robot.connect(robot="franka_fr3", config={"socket_path": socket_path})
        try:
            assert isinstance(robot._backend, SceneClient)
            initial = robot.state()
            assert initial.joints.positions.shape == (7,)
            robot.move_joints([.05, .02, 0, -1.5, 0, 1.5, -.78])
            backend.wait_until_idle(6)
            robot.move_p(initial.pose)
            backend.wait_until_idle(6)
            np.testing.assert_allclose(robot.state().pose.position, initial.pose.position, atol=.003)
            robot.gripper(.04)
            backend.wait_until_idle(12)
            assert robot.state().joints.gripper == pytest.approx(.04, abs=.001)
            frame = robot._backend.read(width=160, height=120)
            assert frame.extrinsics.reference_frame == "fr3_link0"
            assert frame.extrinsics.camera_frame == frame.frame_id == "d435i_color_optical_frame"
            site = backend.model.site("d435i_color_optical_frame").id
            camera = backend.model.camera("d435i_check").id
            np.testing.assert_allclose(backend.data.cam_xpos[camera], backend.data.site_xpos[site], atol=1e-8)
            np.testing.assert_allclose(backend.data.cam_xmat[camera].reshape(3, 3),
                                       backend.data.site_xmat[site].reshape(3, 3) @ np.diag([1, -1, -1]), atol=1e-8)
            assert frame.color.shape == (120, 160, 3)
            assert frame.depth.shape == (120, 160)
            assert frame.depth.dtype == np.float32
            assert np.isfinite(frame.depth).all()
        finally:
            robot.disconnect()
    finally:
        server.stop()
        backend.disconnect()
