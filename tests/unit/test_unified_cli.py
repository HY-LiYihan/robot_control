import json
from pathlib import Path
import time

import numpy as np

from typer.testing import CliRunner
import pytest

from robot_control import cli
from robot_control.api.types import JointState, Pose, RobotState
from robot_control.cli import app
from robot_control.errors import BackendUnavailableError
from robot_control.scene import SceneClient, SceneServer
from robot_control.sensors.frame import CameraIntrinsics, RGBDFrame
from robot_control.sensors import service
from robot_control.sensors.extrinsics import piper_link6_to_color_optical


runner = CliRunner()


def test_real_camera_outputs_base_extrinsics_and_camera_only_mode(monkeypatch, tmp_path):
    class Camera:
        def __init__(self, **kwargs):
            pass

        def connect(self):
            pass

        def disconnect(self):
            pass

        def read(self):
            return RGBDFrame(np.zeros((2, 2, 3), dtype=np.uint8),
                             np.ones((2, 2), dtype=np.uint16), time.time(),
                             "d435i_color_optical_frame", CameraIntrinsics(2, 2, 1, 1, 1, 1), 0.001)

    class Robot:
        def __init__(self, backend):
            self.backend = backend

        def state(self):
            return type("State", (), {"pose": Pose((0, 0, 0)), "timestamp": time.time()})()

        def camera(self):
            return service.CameraService(self.backend, "piper", self).capture()

        def disconnect(self):
            pass

    monkeypatch.setattr(service, "RealSenseCamera", Camera)
    monkeypatch.setattr(cli, "_robot", lambda backend, *args: Robot(backend))
    output = ["--rgb-out", str(tmp_path / "rgb.png"), "--depth-out", str(tmp_path / "depth.npy")]
    result = runner.invoke(app, ["--backend", "real", "--robot", "piper", "camera", *output])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["extrinsics"]["reference_frame"] == "base_link"
    np.testing.assert_allclose(data["extrinsics"]["translation_m"], piper_link6_to_color_optical()[:3, 3])
    assert Path(data["rgb"]).is_file()
    assert Path(data["depth"]).is_file()

    twin_result = runner.invoke(app, ["--backend", "twin", "--robot", "piper", "camera", *output])
    assert twin_result.exit_code == 0, twin_result.output
    assert json.loads(twin_result.output)["extrinsics"]["reference_frame"] == "base_link"

    monkeypatch.setattr(cli, "_robot", lambda *args: pytest.fail("camera-only mode connected to arm"))
    only_camera = runner.invoke(app, ["--backend", "real", "camera", "--no-extrinsics", *output])
    assert only_camera.exit_code == 0, only_camera.output
    assert json.loads(only_camera.output)["extrinsics"] is None
    missing_arm = runner.invoke(app, ["--backend", "real", "camera", *output])
    assert missing_arm.exit_code == 2
    assert "--robot piper" in missing_arm.output


def test_fr3_real_camera_cli_outputs_extrinsics(monkeypatch, tmp_path):
    pytest.importorskip("pinocchio")

    class Camera:
        def __init__(self, **kwargs):
            pass

        def connect(self):
            pass

        def read(self):
            return RGBDFrame(np.zeros((2, 2, 3), dtype=np.uint8),
                             np.ones((2, 2), dtype=np.uint16), time.time(),
                             "d435i_color_optical_frame", CameraIntrinsics(2, 2, 1, 1, 1, 1), .001)

        def disconnect(self):
            pass

    class Robot:
        def camera(self):
            return service.CameraService("real", "franka_fr3", self).capture()

        def state(self):
            return RobotState(True, False, JointState([0, -.7854, 0, -2.3562, 0, 1.5708, .7854]),
                              Pose((0, 0, 0)))

        def disconnect(self):
            pass

    monkeypatch.setattr(service, "RealSenseCamera", Camera)
    monkeypatch.setattr(cli, "_robot", lambda *args: Robot())
    output = ["--rgb-out", str(tmp_path / "rgb.png"), "--depth-out", str(tmp_path / "depth.npy")]
    result = runner.invoke(app, ["--backend", "real", "--robot", "franka_fr3", "camera", *output])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["extrinsics"]["reference_frame"] == "fr3_link0"
    assert data["extrinsics"]["camera_frame"] == "d435i_color_optical_frame"


def test_root_launches_gui_for_selected_robot(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_run_scene_host", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert runner.invoke(app, ["--backend", "mujoco"]).exit_code == 0
    assert calls[-1] == ((0.0,), {"scene": None})
    assert runner.invoke(app, ["--backend", "mujoco", "--robot", "franka_fr3"]).exit_code == 0
    assert calls[-1] == ((0.0,), {"scene": None, "robot": "franka_fr3"})


def test_no_gui_hosts_scene_without_viewer(monkeypatch):
    calls = []
    monkeypatch.setattr("robot_control.mujoco_gui.run_host", lambda **kwargs: calls.append(kwargs))
    result = runner.invoke(app, ["--backend", "mujoco", "--robot", "franka_fr3", "--no-gui"])
    assert result.exit_code == 0, result.output
    assert calls == [{"scene": None, "robot": "franka_fr3", "gui": False}]


def test_real_backend_requires_explicit_piper_without_touching_hardware(monkeypatch):
    monkeypatch.setattr(cli, "_robot", lambda *args, **kwargs: 1 / 0)
    missing = runner.invoke(app, ["--backend", "real", "state"])
    assert missing.exit_code == 2
    assert "requires --robot piper" in missing.output
    assert runner.invoke(app, ["--backend", "twin", "state"]).exit_code == 2


def test_command_global_options_and_conflicts(monkeypatch):
    calls = []
    class FakeRobot:
        def state(self):
            return "fr3 state"

        def disconnect(self):
            pass

    monkeypatch.setattr(cli, "_robot", lambda backend, can_name, robot: calls.append((backend, robot)) or FakeRobot())
    selected = runner.invoke(app, ["--backend", "mujoco", "--robot", "franka_fr3", "state"])
    assert selected.exit_code == 0, selected.output
    assert calls == [("mujoco", "franka_fr3")]
    conflict = runner.invoke(app, ["--robot", "franka_fr3", "state", "--robot", "piper"])
    assert conflict.exit_code == 2
    assert "conflicting --robot" in conflict.output


def test_auto_selection_and_ambiguity(monkeypatch, tmp_path):
    class FakeBackend:
        scene_path = None

    fr3_socket = Path("/tmp") / f"fr3-select-{id(tmp_path)}.sock"
    piper_socket = Path("/tmp") / f"piper-select-{id(tmp_path)}.sock"
    monkeypatch.setenv("FR3_SCENE_SOCKET", str(fr3_socket))
    monkeypatch.setenv("PIPER_SCENE_SOCKET", str(piper_socket))
    fr3_server = SceneServer(FakeBackend(), robot="franka_fr3")
    fr3_server.start()
    calls = []
    class FakeRobot:
        def state(self):
            return "selected"

        def disconnect(self):
            pass

    monkeypatch.setattr(cli, "_robot", lambda backend, can_name, robot: calls.append(robot) or FakeRobot())
    try:
        result = runner.invoke(app, ["state"])
        assert result.exit_code == 0, result.output
        assert calls == ["franka_fr3"]
        piper_server = SceneServer(FakeBackend(), robot="piper")
        piper_server.start()
        try:
            ambiguous = runner.invoke(app, ["state"])
            assert ambiguous.exit_code == 2
            assert "Both Piper and FR3" in ambiguous.output
            explicit = runner.invoke(app, ["--robot", "franka_fr3", "state"])
            assert explicit.exit_code == 0, explicit.output
        finally:
            piper_server.stop()
    finally:
        fr3_server.stop()
    fallback = runner.invoke(app, ["state"])
    assert fallback.exit_code == 0, fallback.output
    assert calls[-1] == "piper"


def test_duplicate_host_cannot_unlink_running_scene(tmp_path):
    class FakeBackend:
        scene_path = None

    socket_path = Path("/tmp") / f"scene-duplicate-{id(tmp_path)}.sock"
    first = SceneServer(FakeBackend(), socket_path=socket_path)
    first.start()
    second = SceneServer(FakeBackend(), socket_path=socket_path)
    try:
        with pytest.raises(BackendUnavailableError, match="already running"):
            second.start()
        second.stop()
        client = SceneClient(socket_path=socket_path)
        client.connect()
        try:
            assert client.scene_info()["robot"] == "piper"
        finally:
            client.disconnect()
    finally:
        first.stop()
