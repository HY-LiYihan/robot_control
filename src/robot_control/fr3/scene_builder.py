"""Mount the FR3 component inside a standalone MJCF environment."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

DEFAULT_SCENE = Path(__file__).parent / "assets/scenes/default.xml"
MOUNT_NAME = "fr3_mount"


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
    path = (DEFAULT_SCENE if path is None else Path(path)).expanduser().resolve()
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Cannot read scene {path}: {exc}") from exc
    mounts = root.findall(f".//body[@name='{MOUNT_NAME}']")
    if root.tag != "mujoco" or len(mounts) != 1 or mounts[0] not in root.findall("worldbody/body"):
        raise ValueError(f"Scene must contain exactly one direct worldbody/{MOUNT_NAME} body")
    mount = mounts[0]
    if len(mount) or set(mount.attrib) - {"name", "pos", "quat"}:
        raise ValueError("fr3_mount must be an empty fixed body")
    for key, size, default in (("pos", 3, ""), ("quat", 4, "1 0 0 0")):
        try:
            values = np.array([float(value) for value in mount.get(key, default).split()])
        except ValueError as exc:
            raise ValueError(f"fr3_mount {key} must contain {size} finite numbers") from exc
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"fr3_mount {key} must contain {size} finite numbers")
        if key == "quat" and np.linalg.norm(values) < 1e-9:
            raise ValueError("fr3_mount quaternion must be nonzero")
    return path


def compile_scene(model_path: Path, scene_path: Path):
    import mujoco

    environment = mujoco.MjSpec.from_file(str(scene_path))
    robot = mujoco.MjSpec.from_file(str(model_path))
    _set_visual_mesh_shell_inertia(robot, mujoco)
    environment.option.timestep = robot.option.timestep
    environment.option.integrator = robot.option.integrator
    environment.compiler.fusestatic = False
    environment.compiler.autolimits = True
    robot.compiler.inertiafromgeom = 0
    mount = _find_body(environment, MOUNT_NAME)
    robot_body = _find_body(robot, "fr3_link0")
    if mount is None:
        raise ValueError(f'Scene does not contain body "{MOUNT_NAME}"')
    if robot_body is None:
        raise ValueError('FR3 model does not contain body "fr3_link0"')
    mount.add_frame().attach_body(robot_body, prefix="", suffix="")
    return environment.compile()
