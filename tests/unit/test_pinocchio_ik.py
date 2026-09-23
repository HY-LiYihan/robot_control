import numpy as np
import pytest

pytest.importorskip("pinocchio")

from robot_control import Pose
from robot_control.backends.piper_model import ASSET_ROOT
from robot_control.kinematics import PinocchioIK


@pytest.fixture
def ik():
    return PinocchioIK(ASSET_ROOT / "piper/urdf/piper_description.urdf")


@pytest.mark.parametrize("q", [
    [.2, .8, -1.2, .2, -.3, .4],
    [-.6, 1.1, -1.6, -.4, .5, -.8],
    [2.4, 1., -1.4, .4, -.6, 1.],
    [0., .05, -.1, 0., .1, 0.],
])
def test_solves_reachable_pose_within_limits(ik, q):
    target = ik.forward(q)
    seed = np.clip(np.array(q) + .08, ik.lower, ik.upper)
    result = ik.solve(target, seed=seed)
    assert result.success, result
    assert result.position_error <= 1e-4
    assert result.orientation_error <= 1e-3
    assert np.all(result.joints >= ik.lower)
    assert np.all(result.joints <= ik.upper)
    actual = ik.forward(result.joints)
    np.testing.assert_allclose(actual.position, target.position, atol=1e-4)
    assert abs(np.dot(actual.quaternion, target.quaternion)) > 1 - 1e-6


def test_half_turn_is_not_misclassified_as_zero_orientation_error(ik):
    seed = np.array([.2, .8, -1.2, .2, -.3, 1.5])
    target_q = seed.copy()
    target_q[-1] -= np.pi
    target = ik.forward(target_q)
    np.testing.assert_allclose(target.position, ik.forward(seed).position, atol=1e-12)
    result = ik.solve(target, seed=seed)
    assert result.success, result
    assert result.iterations > 1
    assert result.orientation_error <= 1e-3
    assert abs(np.dot(ik.forward(result.joints).quaternion, target.quaternion)) > 1 - 1e-6


def test_unreachable_target_reports_failure_and_finite_error(ik):
    result = ik.solve(Pose((10., 0., 10.)), max_iterations=40)
    assert not result.success
    assert result.position_error > 1
    assert np.isfinite(result.joints).all()
    assert np.all(result.joints >= ik.lower)
    assert np.all(result.joints <= ik.upper)


def test_invalid_input_is_rejected(ik):
    with pytest.raises(ValueError):
        ik.solve(Pose((np.nan, 0., 0.)))
    with pytest.raises(ValueError):
        ik.solve(Pose((0., 0., 0.)), seed=np.full(6, np.inf))
