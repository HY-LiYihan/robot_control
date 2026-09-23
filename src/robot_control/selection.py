from __future__ import annotations

from .errors import BackendUnavailableError


ROBOT_BACKENDS = {
    "piper": frozenset(("mujoco", "real", "twin")),
    "franka_fr3": frozenset(("mujoco",)),
}
BACKENDS = frozenset(("mujoco", "real", "twin"))


def normalize_robot(robot: str) -> str:
    selected = "piper" if robot == "pepper" else robot
    if selected not in ROBOT_BACKENDS:
        raise ValueError(f"unknown robot: {robot}; choose piper or franka_fr3")
    return selected


def validate_backend_robot(backend: str, robot: str) -> str:
    selected = normalize_robot(robot)
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    if backend not in ROBOT_BACKENDS[selected]:
        raise BackendUnavailableError("FR3 real-robot control is not implemented; use --backend mujoco")
    return selected
