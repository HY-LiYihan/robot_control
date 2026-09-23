from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_control import Pose
from robot_control.backends.mujoco import MujocoBackend
from robot_control.backends.scene_builder import DEFAULT_SCENE, validate_scene


@pytest.mark.parametrize("mount", [
    "", '<body name="piper_mount"/>',
    '<body name="piper_mount" pos="0 0"/>',
    '<body name="piper_mount" pos="0 nan 0"/>',
    '<body name="piper_mount" pos="0 0 0" quat="0 0 0 0"/>',
    '<body name="piper_mount" pos="0 0 0" quat="1 0 0 inf"/>',
    '<body name="piper_mount" pos="0 0 0"><freejoint/></body>',
    '<body><body name="piper_mount" pos="0 0 0"/></body>',
    '<body name="piper_mount" pos="0 0 0"/><body name="piper_mount" pos="1 0 0"/>',
])
def test_invalid_mount_rejected_before_robot_is_built(tmp_path, monkeypatch, mount):
    scene = tmp_path / "invalid.xml"
    scene.write_text(f"<mujoco><worldbody>{mount}</worldbody></mujoco>")
    def unexpected_build(*args, **kwargs):
        pytest.fail("The robot/assets must not load before scene validation")
    monkeypatch.setattr("robot_control.backends.mujoco.build_piper_scene", unexpected_build)
    with pytest.raises(ValueError, match="piper_mount"):
        backend = MujocoBackend(scene=scene)
        backend.connect()


def test_default_scene_has_blue_sky_and_ground():
    backend = MujocoBackend(wrist_camera=False)
    backend.connect()
    try:
        assert backend.scene_path == DEFAULT_SCENE.resolve()
        assert backend.model.ntex == 2
        assert backend.model.geom("floor").type == mujoco.mjtGeom.mjGEOM_PLANE
        assert backend.model.nq == 8
        np.testing.assert_allclose(backend.data.xpos[backend.model.body("base_link").id], [0, 0, 0])
        assert validate_scene() == backend.scene_path
    finally:
        backend.disconnect()


def test_moved_robot_ik_camera_defaults_and_relative_assets(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tetra.obj").write_text(
        "v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\n"
        "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    (tmp_path / "objects.xml").write_text('''<mujoco>
      <asset><mesh name="environment_mesh" file="tetra.obj"/></asset>
      <worldbody>
        <geom name="environment_geom" type="mesh" mesh="environment_mesh" pos="3 0 0"/>
        <body name="falling" pos="3 0 2"><freejoint/><geom type="sphere" size=".02" mass=".1"/></body>
        <body name="hinged" pos="4 0 2"><joint name="environment_joint"/>
          <geom type="sphere" size=".02" mass=".1"/></body>
      </worldbody>
      <actuator><motor name="environment_motor" joint="environment_joint"/></actuator>
    </mujoco>''')
    # Rotate by 30 degrees about X followed by 90 degrees about Z.
    half_roll = np.pi / 12
    quat = np.array([np.cos(half_roll), np.sin(half_roll), np.sin(half_roll), np.cos(half_roll)]) / np.sqrt(2)
    rotation = np.array([[0, -np.sqrt(3)/2, .5], [1, 0, 0], [0, .5, np.sqrt(3)/2]])
    translation = np.array([1.2, -.7, .8])
    scene = tmp_path / "scene.xml"
    scene.write_text(f'''<mujoco>
      <compiler meshdir="assets" angle="degree"/>
      <default><geom friction="2 .2 .2" rgba="1 0 0 1"/><joint damping="99"/></default>
      <include file="objects.xml"/>
      <worldbody><body name="piper_mount" pos="1.2 -.7 .8" quat="{' '.join(map(str, quat))}"/></worldbody>
    </mujoco>''')
    reference = MujocoBackend()
    placed = MujocoBackend(scene=scene)
    reference.connect()
    placed.connect()
    try:
        assert placed.model.nq == 16  # A free object, one hinge, plus the Piper.
        assert placed.model.nu == 9
        assert placed.model.mesh("environment_mesh").id >= 0
        for name in ("base_link", "link6", "gripper_link1", "d435i_color_optical_frame"):
            ref_id, body_id = reference.model.body(name).id, placed.model.body(name).id
            np.testing.assert_allclose(placed.data.xpos[body_id], rotation @ reference.data.xpos[ref_id] + translation, atol=1e-10)
            np.testing.assert_allclose(placed.data.xmat[body_id].reshape(3, 3), rotation @ reference.data.xmat[ref_id].reshape(3, 3), atol=1e-10)
            np.testing.assert_allclose(placed.model.body_mass[body_id], reference.model.body_mass[ref_id])
            np.testing.assert_allclose(placed.model.body_inertia[body_id], reference.model.body_inertia[ref_id])
        np.testing.assert_allclose(placed.model.geom("link2_geom0").friction, reference.model.geom("link2_geom0").friction)
        np.testing.assert_allclose(placed.model.dof_damping[placed._arm_dofs], reference.model.dof_damping[reference._arm_dofs])
        np.testing.assert_allclose(placed.model.joint("joint2").range, reference.model.joint("joint2").range)
        # Independently transform a known reachable target into world coordinates.
        local = reference.ik.forward([.2, .8, -1.2, .2, -.3, .4])
        target_quat = np.empty(4)
        mujoco.mju_mulQuat(target_quat, quat, np.array(local.quaternion))
        target = Pose(tuple(rotation @ local.position + translation), tuple(target_quat))
        q, v = placed.data.qpos.copy(), placed.data.qvel.copy()
        placed.move_p(target)
        np.testing.assert_array_equal(placed.data.qpos, q)
        np.testing.assert_array_equal(placed.data.qvel, v)
        assert placed.data.ctrl[placed.model.actuator("environment_motor").id] == 0
        placed.wait_until_idle()
        actual = placed.state().pose
        np.testing.assert_allclose(actual.position, target.position, atol=1e-3)
        assert abs(np.dot(actual.quaternion, target.quaternion)) > 1 - 1e-5
    finally:
        reference.disconnect()
        placed.disconnect()
