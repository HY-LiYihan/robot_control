from importlib import import_module

from robot_control import PiperRobot, Pose, Robot


def test_legacy_imports_share_canonical_modules():
    legacy = import_module("piper_control")
    assert Robot is PiperRobot is legacy.Robot is legacy.PiperRobot
    assert legacy.Pose is Pose
    for name in ("api.robot", "api.types", "backends.mujoco", "cli", "fr3.mujoco", "scene"):
        assert import_module(f"piper_control.{name}") is import_module(f"robot_control.{name}")
