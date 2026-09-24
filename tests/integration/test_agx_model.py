"""Cross-check the generated scene against MuJoCo's independent URDF importer."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_control.backends.mujoco import MujocoBackend
from robot_control.backends.piper_model import ASSET_ROOT, ARM_JOINTS, build_piper_scene


@pytest.fixture
def backend():
    robot = MujocoBackend(wrist_camera=False)
    robot.connect()
    yield robot
    robot.disconnect()


def reference_arm():
    root = ET.parse(ASSET_ROOT / "piper/urdf/piper_description.urdf").getroot()
    for link in root.findall("link"):
        # FK/inertia reference needs no geometry and thus no asset conversion.
        for child in list(link):
            if child.tag in ("visual", "collision"):
                link.remove(child)
    compiler = ET.SubElement(root, "mujoco")
    ET.SubElement(compiler, "compiler", fusestatic="false")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def test_fk_and_arm_parameters_match_official_urdf(backend):
    reference = reference_arm()
    data = mujoco.MjData(reference)
    scratch = mujoco.MjData(backend.model)
    rng = np.random.default_rng(1729)
    configurations = [np.zeros(6), np.array([0.2, 0.8, -1.2, 0.2, -0.3, 0.4])]
    configurations.extend(rng.uniform(reference.jnt_range[:, 0], reference.jnt_range[:, 1], (12, 6)))
    for q in configurations:
        # This is a static FK comparison on scratch data, not a motion command.
        scratch.qpos[backend._arm_qpos] = q
        mujoco.mj_forward(backend.model, scratch)
        data.qpos[:] = q
        mujoco.mj_forward(reference, data)
        pin_pose = backend.ik.forward(q)
        np.testing.assert_allclose(pin_pose.position, data.xpos[reference.body("link6").id], atol=1e-10)
        assert abs(np.dot(pin_pose.quaternion, data.xquat[reference.body("link6").id])) > 1 - 1e-10
        for name in ("base_link", *(f"link{i}" for i in range(1, 7))):
            expected, actual = reference.body(name).id, backend.model.body(name).id
            np.testing.assert_allclose(scratch.xpos[actual], data.xpos[expected], atol=1e-10)
            np.testing.assert_allclose(scratch.xmat[actual], data.xmat[expected], atol=1e-10)
            np.testing.assert_allclose(backend.model.body_mass[actual], reference.body_mass[expected])
            np.testing.assert_allclose(backend.model.body_ipos[actual], reference.body_ipos[expected])
            np.testing.assert_allclose(backend.model.body_inertia[actual], reference.body_inertia[expected])
    for name in ARM_JOINTS:
        joint = backend.model.joint(name).id
        expected_range = reference.jnt_range[reference.joint(name).id]
        np.testing.assert_allclose(backend.model.jnt_range[joint], expected_range)
        np.testing.assert_allclose(backend.model.actuator_ctrlrange[backend.model.actuator(name).id], expected_range)
    # Specifically catch the old asymmetric joint1 limit.
    backend.move_joints([2.5, 0.5, -0.5, 0, 0, 0])
    assert backend.state().joints.positions[0] == pytest.approx(0)
    backend.step(int(backend.motion_duration_s / backend.model.opt.timestep) + 1)
    assert backend.data.ctrl[backend._arm_actuators[0]] == pytest.approx(2.5)


def test_arm_meshes_only_reference_new_repository():
    tree = build_piper_scene()
    meshes = tree.findall("asset/mesh[@file]")
    assert len(meshes) == 11
    for mesh in meshes:
        path = Path(mesh.attrib["file"])
        assert path.is_relative_to(ASSET_ROOT.resolve())
        assert path.is_file()


def test_visuals_preserve_physics_and_use_official_colors():
    tree = build_piper_scene()
    root = tree.getroot()
    visuals = root.findall(".//geom[@group='1']")
    assert visuals
    colors = {tuple(float(v) for v in mat.attrib["rgba"].split())
              for mat in root.findall("asset/material")}
    assert (0.854902, 0.121569, 0.121569, 1.0) in colors
    assert (0.113725, 0.113725, 0.113725, 1.0) in colors
    for geom in visuals:
        assert geom.attrib["contype"] == geom.attrib["conaffinity"] == "0"
        assert geom.attrib["mass"] == "0"
    for geom in root.findall(".//geom[@group='3']"):
        assert "contype" not in geom.attrib  # Keep the original contact defaults.
        assert geom.attrib["rgba"] == "0 0 0 0"
    colored = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # Reconstruct the prior STL-only scene, including its original display group.
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom in visuals:
                body.remove(geom)
            else:
                geom.set("group", "0")
    asset = root.find("asset")
    for item in list(asset):
        if item.tag == "material" or (item.tag == "mesh" and "file" not in item.attrib):
            asset.remove(item)
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos",
                  "body_quat", "body_gravcomp", "jnt_range", "dof_damping",
                  "actuator_gainprm", "actuator_ctrlrange", "actuator_forcerange"):
        np.testing.assert_allclose(getattr(colored, field), getattr(original, field), atol=1e-12)
    for i in range(original.ngeom):
        geom = colored.geom(original.geom(i).name).id
        for field in ("geom_pos", "geom_quat", "geom_size", "geom_friction",
                      "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp"):
            np.testing.assert_allclose(getattr(colored, field)[geom], getattr(original, field)[i])
    old_data, new_data = mujoco.MjData(original), mujoco.MjData(colored)
    for data in (old_data, new_data):
        data.ctrl[:] = [0.2, 0.8, -1.2, 0.2, -0.3, 0.4, 0.04, -0.04]
    for _ in range(500):
        mujoco.mj_step(original, old_data)
        mujoco.mj_step(colored, new_data)
        np.testing.assert_allclose(new_data.qpos, old_data.qpos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(new_data.qvel, old_data.qvel, atol=1e-12, rtol=0)
        assert new_data.ncon == old_data.ncon


def test_new_flange_and_gripper_geometry_and_opening(backend):
    model, data = backend.model, backend.data
    assert model.body("flange_link").parentid[0] == model.body("link6").id
    np.testing.assert_allclose(model.body("gripper_base").pos, [0, 0, 0.0045])
    for name in ("gripper_link1", "gripper_link2"):
        np.testing.assert_allclose(model.body(name).pos, [0, 0, 0.138])
    for width in (0.0, 0.02, 0.07, 0.1):
        backend.gripper(width)
        for name, sign in (("gripper_joint1", 1), ("gripper_joint2", -1)):
            assert data.ctrl[model.actuator(name).id] == pytest.approx(sign * width / 2)
        backend.wait_until_idle()
        assert backend.state().joints.gripper == pytest.approx(width, abs=5e-4)
    for invalid in (-0.001, 0.101, float("nan")):
        with pytest.raises(ValueError):
            backend.gripper(invalid)


def test_dynamics_keep_fingers_synchronized_and_state_finite(backend):
    backend.move_joints([0.2, 0.8, -1.2, 0.2, -0.3, 0.4])
    backend.gripper(0.08)
    backend._step(500)
    assert np.isfinite(backend.data.qpos).all()
    assert not np.any(backend.data.warning.number)
    finger_positions = backend.data.qpos[backend._finger_qpos]
    assert sum(finger_positions) == pytest.approx(0.0, abs=1e-5)
    assert backend.state().joints.gripper == pytest.approx(0.08, abs=1e-3)
