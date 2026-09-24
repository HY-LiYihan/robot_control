import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from typer.testing import CliRunner

from robot_control import Robot, cli, twin
from robot_control.api.types import JointState, Pose, RobotState
from robot_control.errors import BackendUnavailableError
from robot_control.scene import SceneClient, twin_socket_path


def test_fr3_twin_mirrors_feedback_and_routes_real_commands(monkeypatch):
    socket_path = Path("/tmp") / f"fr3-twin-{uuid4().hex[:12]}.sock"
    monkeypatch.setenv("FR3_TWIN_SOCKET", str(socket_path))
    monkeypatch.setenv("FRANKA_ROBOT_IP", "192.168.1.6")
    assert twin_socket_path("piper") != twin_socket_path("franka_fr3")
    from robot_control.fr3.mujoco import MujocoBackend
    simulation = MujocoBackend()
    monkeypatch.setattr("robot_control.fr3.mujoco.MujocoBackend", lambda **kwargs: simulation)
    positions = np.array([0.1, -0.6, 0.2, -2.0, 0.2, 1.5, 0.4])
    commands = []

    class FakeFranka:
        robot_ip = "192.168.1.6"
        motion_duration_s = 4.0
        gripper_speed_m_s = 0.05
        rt_priority = 80

        def __init__(self, **kwargs):
            if kwargs:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def connect(self):
            commands.append("connect")

        def disconnect(self):
            commands.append("disconnect")

        def state(self):
            return RobotState(True, False, JointState(positions.copy(), gripper=0.06),
                              Pose((0.5, 0.0, 0.4)))

        def move_joints(self, joints):
            commands.append(("move_joints", joints))

        def move_p(self, pose):
            commands.append(("move_p", pose))

        def gripper(self, width, effort=None):
            commands.append(("gripper", width))

        def stop(self):
            commands.append("stop")

    monkeypatch.setattr("robot_control.backends.franka_direct.FrankaDirectBackend", FakeFranka)
    failures = []

    def host():
        try:
            twin.run_host(duration=8, gui=False, robot="franka_fr3")
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=host)
    thread.start()
    try:
        for attempt in range(120):
            if socket_path.exists():
                break
            if failures or not thread.is_alive():
                pytest.fail(f"FR3 twin host failed: {failures}")
            time.sleep(0.05)
        assert socket_path.exists()
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        client = SceneClient(socket_path=socket_path, robot="franka_fr3")
        client.connect()
        try:
            info = client.scene_info()
            assert (info["mode"], info["robot"], info["robot_ip"]) == ("twin", "franka_fr3", "192.168.1.6")
            with pytest.raises(BackendUnavailableError, match="real RealSense"):
                client.read(width=10, height=10)
        finally:
            client.disconnect()
        with pytest.raises(ValueError, match="different FR3 IP"):
            Robot.connect("real", {"robot_ip": "192.168.1.9"}, robot="franka_fr3")
        with pytest.raises(ValueError, match="different motion_duration_s"):
            Robot.connect("twin", {"motion_duration_s": 8}, robot="franka_fr3")
        arm = Robot.connect("twin", robot="franka_fr3")
        try:
            assert isinstance(arm._backend, SceneClient)
            np.testing.assert_allclose(arm.state().joints.positions, positions)
            arm.move_p(Pose((0.5, 0.0, 0.4)))
            arm.gripper(0.04)
        finally:
            arm.disconnect()
        assert commands.count("connect") == 1
        runner = CliRunner()
        state = runner.invoke(cli.app, ["--backend", "real", "--robot", "franka_fr3", "state"])
        assert state.exit_code == 0, state.output
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        separate = subprocess.run([sys.executable, "-m", "robot_control.cli", "--backend", "real",
                                   "--robot", "franka_fr3", "state"], env=environment,
                                  capture_output=True, text=True, timeout=10)
        assert separate.returncode == 0, separate.stderr
        motion = ["--backend", "twin", "--robot", "franka_fr3", "move-joints",
                  "--j1", "0", "--j2", "-0.6", "--j3", "0", "--j4", "-2",
                  "--j5", "0", "--j6", "1.5", "--j7", "0.4"]
        accepted = runner.invoke(cli.app, motion)
        assert accepted.exit_code == 0, accepted.output
        assert ("move_joints", [0.0, -0.6, 0.0, -2.0, 0.0, 1.5, 0.4]) in commands
        moved_pose = runner.invoke(cli.app, ["--backend", "real", "--robot", "franka_fr3",
                                              "move-p", "--x", "0.5", "--y", "0", "--z", "0.4"])
        assert moved_pose.exit_code == 0, moved_pose.output
        assert any(isinstance(command, tuple) and command[0] == "move_p" for command in commands)
        moved_gripper = runner.invoke(cli.app, ["--backend", "twin", "--robot", "franka_fr3",
                                                 "gripper", "0.04"])
        assert moved_gripper.exit_code == 0, moved_gripper.output
        assert ("gripper", 0.04) in commands
        assert commands.count("connect") == 1
        for attempt in range(60):
            if np.allclose(simulation.data.qpos[simulation._arm_qpos], positions):
                break
            time.sleep(0.05)
        np.testing.assert_allclose(simulation.data.qpos[simulation._arm_qpos], positions)
        np.testing.assert_allclose(simulation.data.qpos[simulation._finger_qpos], [0.03, 0.03])
        assert simulation.data.time == 0
    finally:
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert not failures
    assert commands[-1] == "disconnect"
    assert not socket_path.exists()
    with pytest.raises(BackendUnavailableError, match="twin is not running"):
        Robot.connect("twin", robot="franka_fr3")
