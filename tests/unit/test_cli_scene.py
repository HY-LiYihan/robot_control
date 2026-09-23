from pathlib import Path

from typer.testing import CliRunner

from robot_control.cli import app
from robot_control import cli


runner = CliRunner()


def test_real_scene_option_rejected_before_backend_connection(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Must not connect to hardware")
    monkeypatch.setattr(cli, "_robot", unexpected)
    result = runner.invoke(app, ["run", "--backend", "real", "--scene", "missing.xml"])
    assert result.exit_code == 2
    assert "only supported by the MuJoCo" in result.output


def test_missing_mount_rejected_before_gui(monkeypatch, tmp_path):
    scene = tmp_path / "missing-mount.xml"
    scene.write_text("<mujoco><worldbody/></mujoco>")
    def unexpected(*args, **kwargs):
        raise AssertionError("Must not launch GUI")
    monkeypatch.setattr(cli, "_run_scene_host", unexpected)
    result = runner.invoke(app, ["run", "--gui", "--scene", str(scene)])
    assert result.exit_code == 2
    assert "piper_mount" in result.output


def test_scene_reaches_gui_and_macos_reexec(monkeypatch, tmp_path):
    scene = tmp_path / "scene with spaces.xml"
    scene.write_text('<mujoco><worldbody><body name="piper_mount" pos="1 2 3"/></worldbody></mujoco>')
    calls = []
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.delenv("ROBOT_CONTROL_MUJOCO_GUI_REEXEC", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: calls.append(command))
    result = runner.invoke(app, ["run", "--backend", "mujoco", "--gui", "--scene", str(scene)])
    assert result.exit_code == 0, result.output
    assert calls[0][1:3] == ["-m", "robot_control.mujoco_gui"]
    assert calls[0][-2:] == ["--scene", str(scene.resolve())]
