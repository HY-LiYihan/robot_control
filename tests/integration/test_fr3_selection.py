import numpy as np
import pytest
import time
from typer.testing import CliRunner
from pathlib import Path
from uuid import uuid4

from robot_control import Robot
from robot_control.api.types import JointState, Pose, RobotState
from robot_control import cli
from robot_control.cli import app
from robot_control.errors import BackendUnavailableError
from robot_control.scene import SceneClient, SceneServer, default_socket_path
from robot_control.fr3.scene_builder import validate_scene
from robot_control.sensors.extrinsics import fr3_link7_to_color_optical, pose_matrix, rpy_rotation


runner = CliRunner()


def test_joint_count_and_real_backend_guard():
    assert JointState(np.zeros(6)).velocities.shape == (6,)
    assert JointState(np.zeros(7)).velocities.shape == (7,)
    with pytest.raises(ValueError, match="velocities"):
        JointState(np.zeros(7), np.zeros(6))


def test_fr3_simulation_duration_and_override():
    from robot_control.fr3.mujoco import MujocoBackend

    backend = MujocoBackend()
    backend.connect()
    try:
        assert backend.motion_duration_s == 4.0
        target = np.array([0.05, -0.6, 0, -2.0, 0, 1.6, 0.7])
        backend.move_joints(target)
        assert backend._motion.duration == 4.0
        initial = backend.data.ctrl[backend._arm_actuators].copy()
        backend.move_joints(target, duration_s=0.1)
        backend.step(int(0.05 / backend.model.opt.timestep))
        assert np.linalg.norm(backend.data.ctrl[backend._arm_actuators] - target) < np.linalg.norm(initial - target)
        assert backend.state().moving
        backend.step(int(0.06 / backend.model.opt.timestep) + 1)
        np.testing.assert_allclose(backend.data.ctrl[backend._arm_actuators], target)
    finally:
        backend.disconnect()


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
    piper_real_duration = runner.invoke(app, ["--backend", "real", "--robot", "piper"]
                                        + common + ["--duration", "4"])
    assert piper_real_duration.exit_code == 2
    assert "not supported for Piper real" in piper_real_duration.output
    with pytest.raises(ValueError, match="does not support motion_duration_s"):
        Robot.connect("real", {"motion_duration_s": 4}, robot="piper")


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
    expected_home = np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, .7854])
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
            frame = robot.camera(width=160, height=120)
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


def test_fr3_camera_mount_and_urdf_extrinsics_match(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    from robot_control.fr3.mujoco import MujocoBackend
    from robot_control.sensors import service
    from robot_control.sensors.frame import CameraIntrinsics, RGBDFrame

    class Camera:
        def __init__(self, **kwargs):
            pass

        def connect(self):
            pass

        def read(self):
            return RGBDFrame(np.zeros((2, 2, 3), dtype=np.uint8),
                             np.zeros((2, 2), dtype=np.uint16), time.time(),
                             "d435i_color_optical_frame", CameraIntrinsics(2, 2, 1, 1, 1, 1), .001)

        def disconnect(self):
            pass

    class Arm:
        def __init__(self, joints):
            self.joints = joints

        def state(self):
            return RobotState(True, False, JointState(self.joints), Pose((0, 0, 0)))

    monkeypatch.setattr(service, "RealSenseCamera", Camera)

    backend = MujocoBackend()
    backend.connect()
    try:
        camera_body = backend.model.body("d435i").id
        mount = backend.model.body("fr3_d435i_mount").id
        base = backend.model.body("fr3_link0").id
        optical = backend.model.site("d435i_color_optical_frame").id
        for joints in ([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854],
                       [.1, -.2, .15, -1.4, .2, 1.5, -.7]):
            backend.data.qpos[backend._arm_qpos] = joints
            mujoco.mj_forward(backend.model, backend.data)
            mount_rotation = backend.data.xmat[mount].reshape(3, 3)
            camera_rotation = backend.data.xmat[camera_body].reshape(3, 3)
            np.testing.assert_allclose(mount_rotation.T @ camera_rotation,
                                       rpy_rotation(0, -2.00712863979, 0) @ rpy_rotation(np.pi, 0, 0), atol=1e-6)
            physical_center = (backend.data.xpos[camera_body]
                               + camera_rotation @ [0, -.0175, 0])
            original_center = (backend.data.xpos[mount]
                               + mount_rotation @ [.0635832488467, .04912 - .0175, .0378225720135])
            np.testing.assert_allclose(physical_center, original_center, atol=1e-8)
            base_transform = np.eye(4)
            base_transform[:3, :3] = backend.data.xmat[base].reshape(3, 3)
            base_transform[:3, 3] = backend.data.xpos[base]
            optical_transform = np.eye(4)
            optical_transform[:3, :3] = backend.data.site_xmat[optical].reshape(3, 3)
            optical_transform[:3, 3] = backend.data.site_xpos[optical]
            from_urdf = pose_matrix(backend.ik.forward(joints)) @ fr3_link7_to_color_optical()
            np.testing.assert_allclose(np.linalg.inv(base_transform) @ optical_transform,
                                       from_urdf, atol=2e-6)
            for backend_name in ("real", "twin"):
                frame = service.CameraService(backend_name, "franka_fr3", Arm(joints),
                                              width=2, height=2).capture()
                calculated = np.eye(4)
                calculated[:3, :3] = np.asarray(frame.extrinsics.rotation).reshape(3, 3)
                calculated[:3, 3] = frame.extrinsics.translation
                assert frame.extrinsics.reference_frame == "fr3_link0"
                np.testing.assert_allclose(calculated, np.linalg.inv(base_transform) @ optical_transform,
                                           atol=2e-6)
    finally:
        backend.disconnect()
