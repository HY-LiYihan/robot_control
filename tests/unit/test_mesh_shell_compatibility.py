import sys
from types import SimpleNamespace

import pytest

from robot_control.backends.piper_model import build_piper_scene
from robot_control.backends.scene_builder import _set_visual_mesh_shell_inertia as set_piper_shell
from robot_control.fr3.scene_builder import _set_visual_mesh_shell_inertia as set_fr3_shell


@pytest.mark.parametrize("supported", [False, True])
def test_piper_visual_mesh_declares_shell_when_supported(monkeypatch, supported):
    enum = SimpleNamespace()
    if supported:
        enum.mjMESH_INERTIA_SHELL = object()
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mjtMeshInertia=enum))

    root = build_piper_scene().getroot()
    visual = root.find("asset/mesh[@name='gripper_link1_visual0_2']")
    collision = root.find("asset/mesh[@name='gripper_link1_mesh0']")
    assert visual is not None and collision is not None
    assert visual.get("inertia") == ("shell" if supported else None)
    assert collision.get("inertia") is None


@pytest.mark.parametrize("set_shell", [set_piper_shell, set_fr3_shell])
def test_scene_builder_sets_only_visual_mesh_shell_inertia(set_shell):
    shell = object()
    visual = SimpleNamespace(name="visual", inertia=None)
    collision = SimpleNamespace(name="collision", inertia=None)
    spec = SimpleNamespace(
        geoms=[
            SimpleNamespace(meshname="visual", contype=0, conaffinity=0),
            SimpleNamespace(meshname="collision", contype=1, conaffinity=1),
        ],
        meshes=[visual, collision],
    )
    mujoco = SimpleNamespace(mjtMeshInertia=SimpleNamespace(mjMESH_INERTIA_SHELL=shell))
    set_shell(spec, mujoco)
    assert visual.inertia is shell
    assert collision.inertia is None
