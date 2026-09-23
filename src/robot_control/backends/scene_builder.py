"""Attach the complete Piper assembly to a validated native MuJoCo scene."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_SCENE = Path(__file__).parents[1] / "assets/scenes/default.xml"
MOUNT_NAME = "piper_mount"


def _set_visual_mesh_shell_inertia(spec, mujoco) -> None:
    shell = getattr(mujoco.mjtMeshInertia, "mjMESH_INERTIA_SHELL", None)
    if shell is None:
        return
    visual_meshes = {
        geom.meshname for geom in spec.geoms
        if geom.meshname and geom.contype == 0 and geom.conaffinity == 0
    }
    for mesh in spec.meshes:
        if mesh.name in visual_meshes:
            mesh.inertia = shell


def _find_body(spec, name: str):
    body = getattr(spec, "body", None)
    if callable(body):
        return body(name)
    return spec.find_body(name)


def validate_scene(path: str | Path | None = None) -> Path:
    """Require an explicit, fixed world-space mounting pose before loading assets.

    The empty mount must be in the main XML, directly under worldbody. Other
    environment content may use native MJCF includes and relative asset paths.
    """
    path = (DEFAULT_SCENE if path is None else Path(path)).expanduser().resolve()
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Cannot read scene {path}: {exc}") from exc
    if root.tag != "mujoco":
        raise ValueError(f"Scene must be a MuJoCo MJCF XML file: {path}")
    mounts = root.findall(f".//body[@name='{MOUNT_NAME}']")
    if len(mounts) != 1 or mounts[0] not in root.findall("worldbody/body"):
        raise ValueError(
            f'Scene must define exactly one <body name="{MOUNT_NAME}" pos="x y z" '
            'quat="w x y z"/> directly under <worldbody> in the main XML'
        )
    mount = mounts[0]
    if len(mount) or set(mount.attrib) - {"name", "pos", "quat"}:
        raise ValueError("piper_mount must be an empty fixed body with only name, pos and optional quat")
    for key, count, default in (("pos", 3, ""), ("quat", 4, "1 0 0 0")):
        try:
            values = np.array([float(v) for v in mount.get(key, default).split()])
        except ValueError as exc:
            raise ValueError(f"piper_mount {key} must contain {count} finite numbers") from exc
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"piper_mount {key} must contain {count} finite numbers")
        if key == "quat" and (not np.isfinite(np.linalg.norm(values)) or np.linalg.norm(values) < 1e-9):
            raise ValueError("piper_mount quat must be a nonzero finite quaternion (w x y z)")
    return path


def compile_scene(robot: ET.ElementTree, path: Path):
    """Use MuJoCo's attachment API to preserve assets and isolate robot defaults."""
    import mujoco

    environment = mujoco.MjSpec.from_file(str(path))
    component = mujoco.MjSpec.from_string(ET.tostring(robot.getroot(), encoding="unicode"))
    _set_visual_mesh_shell_inertia(component, mujoco)
    mount = _find_body(environment, MOUNT_NAME)
    robot_body = _find_body(component, "base_link")
    if mount is None:
        raise ValueError(f'Scene does not contain body "{MOUNT_NAME}"')
    if robot_body is None:
        raise ValueError('Piper model does not contain body "base_link"')
    # Keep the robot's controller timestep/integrator and explicit inertias.
    # Environment gravity, lights, textures, objects and defaults remain local
    # scene choices; attachment copies robot defaults rather than inheriting them.
    environment.option.timestep = component.option.timestep
    environment.option.integrator = component.option.integrator
    environment.compiler.fusestatic = False
    # Every dynamic Piper body has an explicit inertial. Avoid carrying the
    # component's geometry-inertia setting into newer MuJoCo versions, where
    # zero-volume visual meshes can fail compilation.
    component.compiler.inertiafromgeom = 0
    environment.compiler.autolimits = True
    mount.add_frame().attach_body(robot_body, prefix="", suffix="")
    return environment.compile()
