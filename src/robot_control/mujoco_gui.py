from __future__ import annotations

import argparse
from pathlib import Path
import time

from .backends.mujoco import MujocoBackend
from .scene import SceneServer


def run_host(duration: float = 0.0, scene: Path | None = None, robot: str = "piper",
             gui: bool = True) -> None:
    """Own the shared MuJoCo scene: host the viewer and serve CLI commands."""
    if robot == "franka_fr3":
        from .fr3.mujoco import MujocoBackend as FR3MujocoBackend
        backend = FR3MujocoBackend(realtime=True, scene=scene)
    elif robot == "piper":
        backend = MujocoBackend(realtime=True, scene=scene)
    else:
        raise ValueError(f"unknown robot: {robot}")
    backend.connect()
    server = SceneServer(backend, robot=robot)
    try:
        server.start()
        if gui:
            backend.run_gui(duration, lock=server.lock)
        else:
            started = previous = time.monotonic()
            remainder = 0.0
            while duration <= 0 or time.monotonic() - started < duration:
                frame_start = time.monotonic()
                remainder += min(frame_start - previous, 0.1)
                previous = frame_start
                steps = int(remainder / backend.model.opt.timestep)
                remainder -= steps * backend.model.opt.timestep
                with server.lock:
                    backend._step(steps, pace=False)
                time.sleep(max(0, 1 / 60 - (time.monotonic() - frame_start)))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        backend.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the selected MuJoCo viewer and serve the shared scene")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 keeps the window open")
    parser.add_argument("--scene", type=Path, help="MuJoCo scene XML containing the robot mount")
    parser.add_argument("--robot", choices=("piper", "franka_fr3"), default="piper")
    args = parser.parse_args()
    run_host(args.duration, scene=args.scene, robot=args.robot)


if __name__ == "__main__":
    main()
