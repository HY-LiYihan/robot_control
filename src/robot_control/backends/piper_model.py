"""Build MuJoCo's scene from the pinned AgileX Piper description.

This adapter supports the concrete, include-only Piper gripper Xacro. It is
not a general Xacro interpreter. Collision uses upstream STLs; display uses
the official DAE geometry, normals and material colors without a ROS runtime.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

from ..errors import BackendUnavailableError
from .collada_visual import visual_meshes


ASSET_ROOT = Path(__file__).parents[3] / "vendor/agx_arm_urdf"
DEFAULT_MODEL = ASSET_ROOT / "piper/urdf/piper_with_gripper_description.xacro"
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGER_JOINTS = ("gripper_joint1", "gripper_joint2")
XACRO = "{http://www.ros.org/wiki/xacro}"


def _numbers(values) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _origin(element: ET.Element | None) -> dict[str, str]:
    origin = None if element is None else element.find("origin")
    if origin is None:
        return {"pos": "0 0 0", "quat": "1 0 0 0"}
    roll, pitch, yaw = (float(v) for v in origin.get("rpy", "0 0 0").split())
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return {"pos": origin.get("xyz", "0 0 0"), "quat": _numbers((
        cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
    ))}


def _resource(filename: str) -> Path:
    prefixes = ("package://agx_arm_description/agx_arm_urdf/",
                "$(find agx_arm_description)/agx_arm_urdf/")
    for prefix in prefixes:
        if filename.startswith(prefix):
            path = ASSET_ROOT / filename[len(prefix):]
            if not path.is_file():
                raise BackendUnavailableError(f"AgileX model resource not found: {path}")
            return path.resolve()
    raise ValueError(f"Unsupported Piper resource path: {filename}")


def _description(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected a Piper URDF/Xacro robot: {path}")
    expanded = ET.Element("robot", name=root.get("name", "piper"))
    for child in root:
        if child.tag == XACRO + "include":
            expanded.extend(_description(_resource(child.attrib["filename"])))
        elif child.tag.startswith(XACRO) or "${" in ET.tostring(child, encoding="unicode"):
            raise ValueError("Only the pinned include-only Piper Xacro is supported")
        else:
            expanded.append(deepcopy(child))
    return expanded


def build_piper_scene(path: Path = DEFAULT_MODEL) -> ET.ElementTree:
    try:
        import mujoco
    except ImportError:
        shell_inertia_supported = False
    else:
        shell_inertia_supported = hasattr(mujoco.mjtMeshInertia, "mjMESH_INERTIA_SHELL")
    robot = _description(path)
    links = {link.attrib["name"]: link for link in robot.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in robot.findall("joint")}
    for name in (*ARM_JOINTS, *FINGER_JOINTS, "gripper"):
        if name not in joints:
            raise ValueError(f"Piper gripper description is missing joint {name}")

    # The massless gripper_link only exposes the total opening to ROS. The two
    # physical fingers carry the simulation DOFs; eliminate that virtual DOF.
    master = joints.pop("gripper")
    virtual_link = master.find("child").attrib["link"]
    if list(links[virtual_link]):
        raise ValueError("Expected gripper_link to be an empty virtual link")
    links.pop(virtual_link)
    multipliers = []
    for name in FINGER_JOINTS:
        mimic = joints[name].find("mimic")
        if mimic is None or mimic.get("joint") != "gripper" or float(mimic.get("offset", "0")) != 0:
            raise ValueError(f"Unsupported Piper mimic relation for {name}")
        multipliers.append(float(mimic.attrib["multiplier"]))
    if multipliers != [0.5, -0.5]:
        raise ValueError("Expected symmetric Piper finger multipliers +0.5 and -0.5")
    master_limit = master.find("limit")
    for name, multiplier in zip(FINGER_JOINTS, multipliers):
        expected = sorted(float(master_limit.attrib[k]) * multiplier for k in ("lower", "upper"))
        limit = joints[name].find("limit")
        if not np.allclose(expected, [float(limit.attrib[k]) for k in ("lower", "upper")]):
            raise ValueError("Piper finger limits disagree with the total opening limits")

    root = ET.Element("mujoco", model="agx_piper")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", fusestatic="false")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    visual_settings = ET.SubElement(root, "visual")
    # Keep the official dark material values visible under the default viewer
    # light, which was previously illuminating a uniformly pale STL model.
    ET.SubElement(visual_settings, "headlight", ambient="0.3 0.3 0.3",
                  diffuse="0.8 0.8 0.8", specular="0.5 0.5 0.5")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    children: dict[str, list[ET.Element]] = {}
    for joint in joints.values():
        children.setdefault(joint.find("parent").attrib["link"], []).append(joint)

    def add_link(parent: ET.Element, joint: ET.Element) -> None:
        name = joint.find("child").attrib["link"]
        link = links[name]
        body = ET.SubElement(parent, "body", name=name, gravcomp="1", **_origin(joint))
        joint_type = joint.attrib["type"]
        if joint_type != "fixed":
            if joint_type not in ("revolute", "prismatic"):
                raise ValueError(f"Unsupported Piper joint type: {joint_type}")
            limit = joint.find("limit")
            ET.SubElement(body, "joint", name=joint.attrib["name"],
                          type="hinge" if joint_type == "revolute" else "slide",
                          actuatorgravcomp="true",
                          axis=joint.find("axis").attrib["xyz"],
                          range=f"{limit.attrib['lower']} {limit.attrib['upper']}")
        inertial = link.find("inertial")
        if inertial is not None:
            origin = _origin(inertial)
            # URDF expresses its tensor in the inertial frame; MJCF fullinertia
            # is in the body frame and must not also carry an inertial quat.
            w, x, y, z = (float(v) for v in origin["quat"].split())
            rotation = np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
            ])
            inertia = inertial.find("inertia").attrib
            tensor = np.array([[float(inertia["i" + a + b if a <= b else "i" + b + a])
                                for b in "xyz"] for a in "xyz"])
            tensor = rotation @ tensor @ rotation.T
            ET.SubElement(body, "inertial", pos=origin["pos"],
                          mass=inertial.find("mass").attrib["value"],
                          fullinertia=_numbers(tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))
        for index, collision in enumerate(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Expected an upstream collision mesh for {name}")
            mesh_name = f"{name}_mesh{index}"
            attributes = {"name": mesh_name, "file": str(_resource(mesh.attrib["filename"]))}
            if mesh.get("scale"):
                attributes["scale"] = mesh.attrib["scale"]
            ET.SubElement(asset, "mesh", **attributes)
            ET.SubElement(body, "geom", name=f"{name}_geom{index}", type="mesh", mesh=mesh_name,
                          group="3", rgba="0 0 0 0", **_origin(collision))
        for index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Expected an upstream visual mesh for {name}")
            for part, converted in enumerate(visual_meshes(_resource(mesh.attrib["filename"]))):
                mesh_name = f"{name}_visual{index}_{part}"
                attributes = {key: value for key, value in converted.items() if key != "rgba"}
                if mesh.get("scale"):
                    attributes["scale"] = mesh.attrib["scale"]
                if shell_inertia_supported:
                    attributes["inertia"] = "shell"
                ET.SubElement(asset, "mesh", name=mesh_name, **attributes)
                # Lambert diffuse colors have no specular lobe. Do not map
                # COLLADA reflectivity onto MuJoCo's unrelated mirror setting.
                ET.SubElement(asset, "material", name=mesh_name, rgba=converted["rgba"],
                              specular="0", shininess="0")
                ET.SubElement(body, "geom", name=mesh_name, type="mesh", mesh=mesh_name,
                              material=mesh_name, group="1", contype="0", conaffinity="0",
                              mass="0", **_origin(visual))
        for child in children.get(name, []):
            add_link(body, child)

    for joint in children.get("world", []):
        add_link(world, joint)
    if not world.findall("body"):
        raise ValueError("Expected the Piper world_to_base_link fixed joint")

    # MuJoCo's automatic parent-child collision filter exempts world-welded
    # parents. The fixed base and link1 mounting meshes overlap by 6 mm, so
    # explicitly exclude this adjacent pair rather than generating joint1 drag.
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="base_link", body2="link1")

    equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "joint", name="gripper_mimic", joint1=FINGER_JOINTS[1],
                  joint2=FINGER_JOINTS[0], polycoef="0 -1 0 0 0",
                  solref="0.004 1", solimp="0.99 0.999 0.001")
    actuator = ET.SubElement(root, "actuator")
    # Controller tuning belongs to this simulator, not the upstream URDF.
    gains = (400, 400, 300, 100, 80, 50, 200, 200)
    damping = (20, 20, 10, 2, 2, 1, 2, 2)
    for name, kp, damp in zip((*ARM_JOINTS, *FINGER_JOINTS), gains, damping):
        limit = joints[name].find("limit")
        effort = float(limit.attrib["effort"])
        joint = root.find(f".//body/joint[@name='{name}']")
        joint.set("damping", str(damp))
        # Joint-level clamping also includes gravity compensation, which is
        # added after the individual position actuator's force calculation.
        joint.set("actuatorfrcrange", _numbers((-effort, effort)))
        joint.set("actuatorfrclimited", "true")
        ET.SubElement(actuator, "position", name=name, joint=name, kp=str(kp),
                      ctrlrange=f"{limit.attrib['lower']} {limit.attrib['upper']}",
                      forcerange=_numbers((-effort, effort)))
    return ET.ElementTree(root)
