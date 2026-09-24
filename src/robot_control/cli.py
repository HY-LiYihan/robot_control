from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Annotated
import typer
from .api.robot import Robot
from .api.types import Pose
from .errors import BackendUnavailableError
from .scene import SceneClient
from .selection import BACKENDS, ROBOT_BACKENDS, normalize_robot, validate_backend_robot
from .sensors.service import CameraService

app = typer.Typer(help="Piper (default) and Franka FR3 control and D435i RGB-D CLI")


def _robot_name(robot: str) -> str:
    try:
        return normalize_robot(robot)
    except ValueError as exc:
        raise typer.BadParameter("choose piper or franka_fr3", param_hint="--robot") from exc


def _robot(backend: str, can_name: str, robot: str = "piper", **config):
    options = {"can_name": can_name} if backend in ("real", "twin") and robot == "piper" else {}
    return Robot.connect(backend, {**options, **config}, robot=_robot_name(robot))


def _active_robot() -> str | None:
    active = []
    for name in ROBOT_BACKENDS:
        client = SceneClient(robot=name)
        try:
            client.connect()
        except BackendUnavailableError:
            continue
        try:
            info = client.scene_info()
            if info.get("robot", "piper") != name:
                raise RuntimeError(f"Scene socket for {name} belongs to {info.get('robot')}")
            active.append(name)
        finally:
            client.disconnect()
    if len(active) > 1:
        raise typer.BadParameter("Both Piper and FR3 scenes are running; specify --robot", param_hint="--robot")
    return active[0] if active else None


def _selection(ctx: typer.Context, backend: str | None = None, robot: str | None = None,
               *, launch: bool = False, camera: bool = False) -> tuple[str, str]:
    global_backend = ctx.obj.get("backend")
    global_robot = ctx.obj.get("robot")
    if backend is not None and global_backend is not None and backend != global_backend:
        raise typer.BadParameter("conflicting --backend options", param_hint="--backend")
    if robot is not None and global_robot is not None and _robot_name(robot) != _robot_name(global_robot):
        raise typer.BadParameter("conflicting --robot options", param_hint="--robot")
    selected_backend = backend or global_backend or "mujoco"
    if selected_backend not in BACKENDS:
        raise typer.BadParameter("choose mujoco, real or twin", param_hint="--backend")
    explicit_robot = robot or global_robot
    if (selected_backend in ("real", "twin") and not camera and explicit_robot is None
            and not (selected_backend == "twin" and launch)):
        raise typer.BadParameter("real-robot control requires --robot piper", param_hint="--robot")
    selected_robot = (_robot_name(explicit_robot) if explicit_robot is not None else
                      "piper" if launch or selected_backend in ("real", "twin") else _active_robot() or "piper")
    try:
        validate_backend_robot(selected_backend, selected_robot)
    except BackendUnavailableError as exc:
        raise typer.BadParameter(str(exc), param_hint="--robot") from exc
    return selected_backend, selected_robot


def _scene_for(robot: str, path: Path | None) -> Path | None:
    if path is None:
        return None
    if robot == "franka_fr3":
        from .fr3.scene_builder import validate_scene
    else:
        from .backends.scene_builder import validate_scene
    try:
        return validate_scene(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--scene") from exc


def _run_scene_host(duration: float, scene: Path | None = None, robot: str = "piper") -> None:
    """Host the shared MuJoCo scene; on macOS re-exec through mjpython for the viewer."""
    if sys.platform == "darwin" and not os.environ.get("ROBOT_CONTROL_MUJOCO_GUI_REEXEC"):
        mjpython = Path(sys.executable).with_name("mjpython")
        if not mjpython.is_file():
            raise RuntimeError("Install robot-control[mujoco] in this Python environment to provide mjpython")
        environment = os.environ.copy()
        environment["ROBOT_CONTROL_MUJOCO_GUI_REEXEC"] = "1"
        source_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        command = [str(mjpython), "-m", "robot_control.mujoco_gui", "--duration", str(duration)]
        if robot != "piper":
            command.extend(["--robot", robot])
        if scene is not None:
            command.extend(["--scene", str(scene)])
        subprocess.run(command, env=environment, check=True)
        return
    from .mujoco_gui import run_host
    run_host(duration, scene=scene, robot=robot)


def _run_twin_host(duration: float, scene: Path | None = None, can_name: str = "can0",
                   robot: str = "piper") -> None:
    if sys.platform == "darwin" and not os.environ.get("ROBOT_CONTROL_MUJOCO_GUI_REEXEC"):
        mjpython = Path(sys.executable).with_name("mjpython")
        if not mjpython.is_file():
            raise RuntimeError("Install robot-control[mujoco] in this Python environment to provide mjpython")
        environment = os.environ.copy()
        environment["ROBOT_CONTROL_MUJOCO_GUI_REEXEC"] = "1"
        source_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        command = [str(mjpython), "-m", "robot_control.twin", "--duration", str(duration),
                   "--can-name", can_name]
        if robot != "piper":
            command.extend(["--robot", robot])
        if scene is not None:
            command.extend(["--scene", str(scene)])
        subprocess.run(command, env=environment, check=True)
        return
    from .twin import run_host
    if robot == "piper":
        run_host(duration, scene=scene, can_name=can_name)
    else:
        run_host(duration, scene=scene, robot=robot)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, backend: str | None = typer.Option(None, "--backend"),
         robot: str | None = typer.Option(None, "--robot"),
         scene: Path | None = typer.Option(None, "--scene"),
         no_gui: bool = typer.Option(False, "--no-gui")):
    """Start a MuJoCo scene with GUI by default; control commands reuse it."""
    ctx.obj = {"backend": backend, "robot": robot, "scene": scene, "no_gui": no_gui}
    if ctx.invoked_subcommand is not None:
        if scene is not None and ctx.invoked_subcommand != "run":
            raise typer.BadParameter("--scene is only valid when starting a scene", param_hint="--scene")
        if no_gui:
            raise typer.BadParameter("--no-gui is only valid when starting a scene", param_hint="--no-gui")
        return
    selected_backend, selected_robot = _selection(ctx, launch=True)
    if selected_backend == "twin":
        validated = _scene_for(selected_robot, scene)
        if no_gui:
            from .twin import run_host
            run_host(scene=validated, gui=False, robot=selected_robot)
        else:
            if selected_robot == "piper":
                _run_twin_host(0.0, scene=validated)
            else:
                _run_twin_host(0.0, scene=validated, robot=selected_robot)
        return
    if selected_backend == "real":
        if scene is not None or no_gui:
            raise typer.BadParameter("--scene and --no-gui require the MuJoCo backend")
        instance = _robot(selected_backend, "can0", selected_robot)
        try:
            typer.echo(instance.state())
        finally:
            instance.disconnect()
        return
    validated = _scene_for(selected_robot, scene)
    if no_gui:
        from .mujoco_gui import run_host
        run_host(scene=validated, robot=selected_robot, gui=False)
    elif selected_robot == "piper":
        _run_scene_host(0.0, scene=validated)
    else:
        _run_scene_host(0.0, scene=validated, robot=selected_robot)


def _pose_dict(pose: Pose) -> dict[str, list[float]]:
    return {
        "position_m": [float(value) for value in pose.position],
        "quaternion_wxyz": [float(value) for value in pose.quaternion],
    }


@app.command()
def doctor():
    """Report optional runtime dependencies and pinned asset paths."""
    checks = {}
    for name in ("numpy", "typer", "mujoco", "pyrealsense2", "can"):
        try:
            __import__(name)
            checks[name] = "available"
        except ImportError:
            checks[name] = "missing"
    typer.echo(json.dumps(checks, indent=2))


@app.command()
def state(ctx: typer.Context, backend: str | None = None, can_name: str = "can0", robot: str | None = None):
    backend, robot = _selection(ctx, backend, robot)
    instance = _robot(backend, can_name, robot)
    try:
        typer.echo(instance.state())
    finally:
        instance.disconnect()


@app.command()
def pose(ctx: typer.Context, backend: str | None = None, can_name: str = "can0", robot: str | None = None):
    """Print the current end-effector pose as JSON."""
    backend, robot = _selection(ctx, backend, robot)
    instance = _robot(backend, can_name, robot)
    try:
        current = instance.state().pose
        if current is None:
            raise typer.BadParameter("backend did not return an end-effector pose")
        typer.echo(json.dumps(_pose_dict(current), indent=2))
    finally:
        instance.disconnect()


@app.command("move-joints")
def move_joints(
    ctx: typer.Context,
    j1: float = typer.Option(..., "--j1", help="Joint 1 in radians"),
    j2: float = typer.Option(..., "--j2", help="Joint 2 in radians"),
    j3: float = typer.Option(..., "--j3", help="Joint 3 in radians"),
    j4: float = typer.Option(..., "--j4", help="Joint 4 in radians"),
    j5: float = typer.Option(..., "--j5", help="Joint 5 in radians"),
    j6: float = typer.Option(..., "--j6", help="Joint 6 in radians"),
    j7: float | None = typer.Option(None, "--j7", help="Joint 7, required only for franka_fr3"),
    backend: str | None = None, can_name: str = "can0", robot: str | None = None,
    duration: float | None = typer.Option(None, "--duration", help="FR3 real motion duration in seconds"),
    degrees: bool = typer.Option(False, "--degrees", help="Interpret joint positions as degrees")):
    backend, selected = _selection(ctx, backend, robot)
    if selected == "franka_fr3" and j7 is None:
        raise typer.BadParameter("--j7 is required for franka_fr3", param_hint="--j7")
    if selected == "piper" and j7 is not None:
        raise typer.BadParameter("--j7 is only valid for franka_fr3", param_hint="--j7")
    if duration is not None and not (backend in ("real", "twin") and selected == "franka_fr3"):
        raise typer.BadParameter("--duration is only supported for FR3 real motion")
    joints = [j1, j2, j3, j4, j5, j6]
    if j7 is not None:
        joints.append(j7)
    if degrees:
        import math
        joints = [math.radians(value) for value in joints]
    options = {"motion_duration_s": duration} if duration is not None else {}
    instance = _robot(backend, can_name, selected, **options)
    try:
        instance.move_joints(joints)
        if backend == "mujoco":
            instance.wait_until_idle()
        typer.echo(json.dumps({"joints_rad": instance.state().joints.positions.tolist()}, indent=2))
    finally:
        instance.disconnect()


@app.command("move-p")
def move_p(
           ctx: typer.Context,
           x: float = typer.Option(..., "--x", help="X position in metres"),
           y: float = typer.Option(..., "--y", help="Y position in metres"),
           z: float = typer.Option(..., "--z", help="Z position in metres"),
           qw: float | None = typer.Option(None, "--qw"), qx: float | None = typer.Option(None, "--qx"),
           qy: float | None = typer.Option(None, "--qy"), qz: float | None = typer.Option(None, "--qz"),
           backend: str | None = None, can_name: str = "can0", robot: str | None = None,
           duration: float | None = typer.Option(None, "--duration", help="FR3 real motion duration in seconds")):
    backend, robot = _selection(ctx, backend, robot)
    if duration is not None and not (backend in ("real", "twin") and robot == "franka_fr3"):
        raise typer.BadParameter("--duration is only supported for FR3 real motion")
    options = {"motion_duration_s": duration} if duration is not None else {}
    instance = _robot(backend, can_name, robot, **options)
    try:
        quaternion = (qw, qx, qy, qz)
        if all(value is None for value in quaternion):
            current = instance.state().pose
            if current is None:
                raise typer.BadParameter("backend did not return a pose for orientation hold")
            quaternion = current.quaternion
        elif any(value is None for value in quaternion):
            raise typer.BadParameter("provide all four quaternion options or none")
        target = Pose((x, y, z), tuple(float(value) for value in quaternion))
        instance.move_p(target)
        if backend == "mujoco":
            instance.wait_until_idle()
        current = instance.state().pose
        if current is not None:
            typer.echo(json.dumps(_pose_dict(current), indent=2))
    finally:
        instance.disconnect()


@app.command()
def gripper(ctx: typer.Context, width: float, effort: float | None = None, backend: str | None = None,
            can_name: str = "can0", robot: str | None = None,
            speed: float | None = typer.Option(None, "--speed", help="FR3 real gripper speed in m/s")):
    backend, robot = _selection(ctx, backend, robot)
    if speed is not None and not (backend in ("real", "twin") and robot == "franka_fr3"):
        raise typer.BadParameter("--speed is only supported for FR3 real gripper")
    options = {"gripper_speed_m_s": speed} if speed is not None else {}
    instance = _robot(backend, can_name, robot, **options)
    try:
        instance.gripper(width, effort)
        if backend == "mujoco":
            instance.wait_until_idle()
        typer.echo(json.dumps({"gripper_width_m": instance.state().joints.gripper}, indent=2))
    finally:
        instance.disconnect()


@app.command()
def stop(ctx: typer.Context, backend: str | None = None, can_name: str = "can0", robot: str | None = None):
    backend, robot = _selection(ctx, backend, robot)
    instance = _robot(backend, can_name, robot)
    try:
        instance.stop()
    finally:
        instance.disconnect()


@app.command()
def run(ctx: typer.Context, backend: str | None = None, can_name: str = "can0", steps: int = 0,
        gui: bool = False, duration: float = 0.0,
        scene: Annotated[Path | None, typer.Option("--scene", help="MuJoCo scene XML with the selected robot's mount pose")] = None,
        robot: str | None = None):
    """Start a MuJoCo or real-robot twin GUI (--gui), or report the current state."""
    scene = scene or ctx.obj["scene"]
    if scene is not None and (backend or ctx.obj["backend"] or "mujoco") == "real":
        raise typer.BadParameter("--scene is only supported by the MuJoCo or twin backend", param_hint="--scene")
    backend, selected = _selection(ctx, backend, robot, launch=gui or steps > 0 or scene is not None)
    if scene is not None:
        scene = _scene_for(selected, scene)
    if gui:
        if backend == "twin":
            if selected == "piper":
                _run_twin_host(duration, scene=scene, can_name=can_name)
            else:
                _run_twin_host(duration, scene=scene, robot=selected)
            return
        if backend != "mujoco":
            raise typer.BadParameter("--gui is only supported by MuJoCo or twin")
        if selected == "piper":
            _run_scene_host(duration, scene=scene)
        else:
            _run_scene_host(duration, scene=scene, robot=selected)
        return
    if backend == "twin" and (steps > 0 or scene is not None):
        raise typer.BadParameter("twin is a real-robot mirror; use --gui to start the viewer")
    if backend == "mujoco" and (steps > 0 or scene is not None):
        # Standalone stepping keeps the documented --steps mode working
        # without requiring a running GUI process.
        if selected == "franka_fr3":
            from .fr3.mujoco import MujocoBackend
        else:
            from .backends.mujoco import MujocoBackend
        impl = MujocoBackend(realtime=True, scene=scene)
        impl.connect()
        try:
            impl._step(steps)
            typer.echo(impl.state())
        finally:
            impl.disconnect()
        return
    instance = _robot(backend, can_name, selected)
    try:
        typer.echo(instance.state())
    finally:
        instance.disconnect()


@app.command()
def camera(ctx: typer.Context, backend: str | None = None, can_name: str = "can0",
           rgb_out: Path = Path("wrist_rgb.png"), depth_out: Path = Path("wrist_depth.npy"),
           robot: str | None = None, no_extrinsics: bool = False):
    """Capture aligned wrist RGB-D and its base-to-color-optical extrinsics."""
    backend, selected = _selection(ctx, backend, robot, camera=True)
    instance = None
    try:
        if backend in ("real", "twin") and no_extrinsics:
            frame = CameraService(backend, selected, include_extrinsics=False).capture()
        else:
            if backend in ("real", "twin") and robot is None and ctx.obj.get("robot") is None:
                raise typer.BadParameter("real camera extrinsics require --robot piper", param_hint="--robot")
            instance = _robot(backend, can_name, selected)
            frame = instance.camera()
        import numpy as np
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install robot-control[camera] to save PNG images") from exc
        rgb_out.parent.mkdir(parents=True, exist_ok=True)
        depth_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame.color).save(rgb_out)
        np.save(depth_out, frame.depth)
        typer.echo(json.dumps({
            "rgb": str(rgb_out.resolve()),
            "depth": str(depth_out.resolve()),
            "color_shape": list(frame.color.shape),
            "depth_shape": list(frame.depth.shape),
            "frame_id": frame.frame_id,
            "depth_scale": frame.depth_scale,
            "extrinsics": None if frame.extrinsics is None else {
                "reference_frame": frame.extrinsics.reference_frame,
                "camera_frame": frame.extrinsics.camera_frame,
                "rotation_row_major": list(frame.extrinsics.rotation),
                "translation_m": list(frame.extrinsics.translation),
                "timestamp": frame.extrinsics.timestamp,
            },
        }, indent=2))
    finally:
        if instance is not None:
            instance.disconnect()


if __name__ == "__main__":
    app()
