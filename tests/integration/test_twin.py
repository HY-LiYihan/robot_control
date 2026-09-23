import os
import stat
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from typer.testing import CliRunner

from robot_control import Robot
from robot_control import cli, twin
from robot_control.api.types import JointState, Pose, RobotState
from robot_control.backends.mujoco import MujocoBackend
from robot_control.errors import BackendUnavailableError
from robot_control.scene import SceneClient


def test_twin_mirrors_feedback_without_stepping_or_simulated_control(monkeypatch):
    socket_path = Path("/tmp") / f"piper-twin-{uuid4().hex[:12]}.sock"
    monkeypatch.setenv("PIPER_TWIN_SOCKET", str(socket_path))
    simulation = MujocoBackend()
    monkeypatch.setattr(twin, "MujocoBackend", lambda **kwargs: simulation)
    commands = []
    feedback = [[0.1, 0.2, -0.3, 0.1, 0.2, 0.3]]

    class FakeReal:
        can_name = "can0"

        def __init__(self, can_name):
            self.can_name = can_name

        def connect(self, *, piper_init=True):
            assert piper_init is False
            commands.append("connected")

        def disconnect(self):
            commands.append("disconnected")

        def state(self):
            return RobotState(True, False, JointState(feedback[0]), Pose((0, 0, 0)))

        def gripper_width(self):
            return 0.04

        def move_joints(self, joints):
            commands.append(("move_joints", joints))

        def move_p(self, pose):
            commands.append(("move_p", pose))

        def gripper(self, width, effort=None):
            commands.append(("gripper", width))

        def stop(self):
            commands.append("stop")

    monkeypatch.setattr(twin, "RealBackend", FakeReal)
    errors = []

    def host():
        try:
            twin.run_host(duration=5, gui=False)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=host)
    thread.start()
    try:
        for attempt in range(120):
            if socket_path.exists():
                break
            if errors or not thread.is_alive():
                pytest.fail(f"Twin host failed: {errors}")
            time.sleep(0.05)
        assert socket_path.exists()
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        client = SceneClient(socket_path=socket_path)
        client.connect()
        try:
            assert client.scene_info()["mode"] == "twin"
            with pytest.raises(BackendUnavailableError, match="real RealSense"):
                client.read(width=10, height=10)
        finally:
            client.disconnect()

        runner = CliRunner()
        assert runner.invoke(cli.app, ["--backend", "twin", "--robot", "piper", "state"]).exit_code == 0
        assert runner.invoke(cli.app, ["--backend", "real", "--robot", "piper", "state"]).exit_code == 0

        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2] / "src")
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        command = [sys.executable, "-m", "robot_control.cli", "--backend", "real", "--robot", "piper"]
        for args in (["state"], ["move-joints", "--j1", "0.05", "--j2", "0", "--j3", "0",
                                     "--j4", "0", "--j5", "0", "--j6", "0"]):
            result = subprocess.run(command + args, env=environment, capture_output=True,
                                    text=True, timeout=15)
            assert result.returncode == 0, result.stderr
        assert ("move_joints", [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]) in commands

        robot = Robot.connect("real", {"can_name": "can0"})
        try:
            assert isinstance(robot._backend, SceneClient)
            np.testing.assert_allclose(robot.state().joints.positions, [0.1, 0.2, -0.3, 0.1, 0.2, 0.3])
            robot.move_joints([0] * 6)
            robot.move_p(Pose((0, 0, 0)))
            robot.gripper(0.03)
            robot.stop()
        finally:
            robot.disconnect()

        with pytest.raises(ValueError, match="different CAN"):
            Robot.connect("twin", {"can_name": "can1"})
        np.testing.assert_allclose(simulation.data.qpos[simulation._arm_qpos], [0.1, 0.2, -0.3, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(simulation.data.qpos[simulation._finger_qpos], [0.02, -0.02])
        feedback[0] = [-0.1, 0.1, -0.2, 0, 0, 0]
        for attempt in range(40):
            if np.allclose(simulation.data.qpos[simulation._arm_qpos], feedback[0]):
                break
            time.sleep(0.05)
        np.testing.assert_allclose(simulation.data.qpos[simulation._arm_qpos], feedback[0])
        assert simulation.data.time == 0
        assert ("move_joints", [0.0] * 6) in commands
        assert "stop" in commands
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert commands[-1] == "disconnected"
    assert not socket_path.exists()


def test_twin_gui_only_syncs_measured_configuration(monkeypatch):
    import mujoco.viewer

    simulation = MujocoBackend()
    monkeypatch.setattr(twin, "MujocoBackend", lambda **kwargs: simulation)
    frames = []

    class FakeReal:
        can_name = "can0"

        def __init__(self, can_name):
            pass

        def connect(self, *, piper_init=True):
            assert not piper_init

        def disconnect(self):
            pass

        def state(self):
            return RobotState(True, False, JointState([0.1] * 6), Pose((0, 0, 0)))

        def gripper_width(self):
            return None

    class FakeViewer:
        def __init__(self, model, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def lock(self):
            return nullcontext()

        def is_running(self):
            return len(frames) < 2

        def sync(self):
            frames.append((self.data.qpos[simulation._arm_qpos].copy(), self.data.time))

    monkeypatch.setattr(twin, "RealBackend", FakeReal)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", FakeViewer)
    twin.run_host(gui=True)
    assert len(frames) == 2
    for joints, timestamp in frames:
        np.testing.assert_allclose(joints, [0.1] * 6)
        assert timestamp == 0


def test_twin_cli_requires_explicit_robot_for_commands_and_rejects_fr3(monkeypatch):
    runner = CliRunner()
    calls = []
    monkeypatch.setattr(cli, "_run_twin_host", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert runner.invoke(cli.app, ["--backend", "twin"]).exit_code == 0
    assert calls == [((0.0,), {"scene": None})]
    assert runner.invoke(cli.app, ["--backend", "twin", "state"]).exit_code == 2
    assert runner.invoke(cli.app, ["--backend", "twin", "--robot", "franka_fr3"]).exit_code == 2
    assert runner.invoke(cli.app, ["--backend", "twin", "--robot", "piper", "run", "--steps", "1"]).exit_code == 2
    result = runner.invoke(cli.app, ["--backend", "twin", "--robot", "piper", "run", "--gui",
                                     "--duration", "1", "--can-name", "can1"])
    assert result.exit_code == 0, result.output
    assert calls[-1] == ((1.0,), {"scene": None, "can_name": "can1"})


def test_twin_client_fails_closed_when_host_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPER_TWIN_SOCKET", str(tmp_path / "missing.sock"))
    with pytest.raises(BackendUnavailableError, match="twin is not running"):
        Robot.connect("twin")
