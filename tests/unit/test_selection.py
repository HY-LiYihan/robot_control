import pytest

from robot_control.errors import BackendUnavailableError
from robot_control.selection import normalize_robot, validate_backend_robot


@pytest.mark.parametrize("backend,robot", [
    ("mujoco", "piper"), ("mujoco", "franka_fr3"),
    ("real", "piper"), ("twin", "piper"),
])
def test_supported_combinations(backend, robot):
    assert validate_backend_robot(backend, robot) == robot


def test_alias_and_unsupported_combinations():
    assert normalize_robot("pepper") == "piper"
    assert validate_backend_robot("mujoco", "pepper") == "piper"
    assert validate_backend_robot("real", "franka_fr3") == "franka_fr3"
    assert validate_backend_robot("twin", "franka_fr3") == "franka_fr3"
    with pytest.raises(ValueError, match="unknown robot"):
        validate_backend_robot("mujoco", "other")
    with pytest.raises(ValueError, match="unknown backend"):
        validate_backend_robot("other", "piper")
