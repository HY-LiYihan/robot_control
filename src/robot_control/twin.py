from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np

from .backends.mujoco import MujocoBackend
from .backends.real import RealBackend
from .errors import BackendUnavailableError
from .scene import SceneClient, SceneServer, twin_socket_path


def mirror_feedback(real, simulation, robot: str = "piper") -> None:
    import mujoco

    state = real.state()
    joints = state.joints.positions
    count = 7 if robot == "franka_fr3" else 6
    if joints.shape != (count,) or not np.isfinite(joints).all():
        raise BackendUnavailableError(f"{robot} twin requires {count} finite measured joint angles")
    simulation.data.qpos[simulation._arm_qpos] = joints
    width = state.joints.gripper if robot == "franka_fr3" else real.gripper_width()
    if width is not None and np.isfinite(width):
        opening = np.clip(width, 0.0, simulation._gripper_max_width) / 2
        simulation.data.qpos[simulation._finger_qpos] = ((opening, opening) if robot == "franka_fr3"
                                                       else (opening, -opening))
    simulation.data.qvel[:] = 0
    mujoco.mj_forward(simulation.model, simulation.data)


def run_host(duration: float = 0.0, *, scene: Path | None = None,
             can_name: str = "can0", gui: bool = True, robot: str = "piper") -> None:
    """Mirror measured real joints in MuJoCo; serve control using the real backend."""
    if robot not in ("piper", "franka_fr3"):
        raise ValueError(f"unknown robot: {robot}")
    socket_path = twin_socket_path(robot)
    existing = SceneClient(socket_path=socket_path, robot=robot)
    try:
        existing.connect()
    except BackendUnavailableError:
        pass
    else:
        raise BackendUnavailableError(f"{robot} twin is already running")
    finally:
        existing.disconnect()

    if robot == "franka_fr3":
        from .fr3.mujoco import MujocoBackend as FR3MujocoBackend
        from .backends.franka_direct import FrankaDirectBackend
        simulation = FR3MujocoBackend(scene=scene)
        real = FrankaDirectBackend()
    else:
        simulation = MujocoBackend(scene=scene)
        real = RealBackend(can_name=can_name)
    simulation.connect()
    server = SceneServer(real, socket_path=socket_path, robot=robot, mode="twin")
    try:
        if robot == "piper":
            real.connect(piper_init=False)
        else:
            real.connect()
        deadline = time.monotonic() + 5
        while True:
            try:
                mirror_feedback(real, simulation, robot=robot)
                break
            except BackendUnavailableError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        server.start()
        if gui:
            import mujoco.viewer
            viewer_context = mujoco.viewer.launch_passive(simulation.model, simulation.data)
        else:
            viewer_context = nullcontext(None)
        with viewer_context as viewer:
            started_at = time.monotonic()
            while viewer is None or viewer.is_running():
                tick = time.monotonic()
                if duration > 0 and tick - started_at >= duration:
                    break
                if viewer is not None:
                    with viewer.lock():
                        with server.lock:
                            mirror_feedback(real, simulation, robot=robot)
                    viewer.sync()
                else:
                    with server.lock:
                        mirror_feedback(real, simulation, robot=robot)
                time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        with server.lock:
            real.disconnect()
        simulation.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror real-robot feedback in a MuJoCo viewer")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--robot", choices=("piper", "franka_fr3"), default="piper")
    args = parser.parse_args()
    run_host(args.duration, scene=args.scene, can_name=args.can_name, robot=args.robot)


if __name__ == "__main__":
    main()
