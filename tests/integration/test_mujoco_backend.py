import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_control.backends.mujoco import MujocoBackend
from robot_control import Pose
from robot_control.errors import IKError


def test_mujoco_model_loads_and_moves():
    backend = MujocoBackend()
    backend.connect()
    try:
        assert backend.model.nq == 8
        assert backend.state().joints.gripper == pytest.approx(0.07)
        np.testing.assert_allclose(backend.data.qpos[backend._finger_qpos], [0.035, -0.035])
        np.testing.assert_allclose(backend.data.ctrl[backend._finger_actuators], [0.035, -0.035])
        target = [0, 0.5, -0.5, 0, 0, 0]
        backend.move_joints(target)
        assert backend.state().moving
        np.testing.assert_array_equal(backend.data.qpos[backend._arm_qpos], np.zeros(6))
        np.testing.assert_allclose(backend.data.ctrl[backend._arm_actuators], target)
        backend.step(1)
        assert np.linalg.norm(backend.state().joints.positions) > 0
        assert np.max(np.abs(backend.state().joints.positions - target)) > 0.1
        backend.wait_until_idle()
        assert np.max(np.abs(backend.state().joints.positions - target)) < 0.02
        assert np.isfinite(backend.state().joints.positions).all()
        backend.gripper(0.02)
        backend.wait_until_idle()
        assert backend.state().joints.gripper == pytest.approx(0.02, abs=5e-4)
    finally:
        backend.disconnect()


def test_ik_and_stop_only_update_controls_and_failure_is_atomic():
    backend = MujocoBackend(wrist_camera=False)
    backend.connect()
    try:
        target = backend.ik.forward([.2, .8, -1.2, .2, -.3, .4])
        initial_q = backend.data.qpos.copy()
        initial_v = backend.data.qvel.copy()
        backend.move_p(target)
        np.testing.assert_array_equal(backend.data.qpos, initial_q)
        np.testing.assert_array_equal(backend.data.qvel, initial_v)
        assert backend.state().moving
        controls = backend.data.ctrl.copy()
        with pytest.raises(IKError):
            backend.move_p(Pose((10., 0., 10.)))
        np.testing.assert_array_equal(backend.data.ctrl, controls)
        np.testing.assert_array_equal(backend.data.qpos, initial_q)
        np.testing.assert_array_equal(backend.data.qvel, initial_v)
        backend.wait_until_idle()
        assert np.linalg.norm(np.array(backend.state().pose.position) - target.position) < 1e-3

        backend.move_joints([.4, .9, -1.3, .3, -.4, .5])
        backend.step(20)
        q, v = backend.data.qpos.copy(), backend.data.qvel.copy()
        assert np.linalg.norm(v) > 0
        backend.stop()
        np.testing.assert_array_equal(backend.data.qpos, q)
        np.testing.assert_array_equal(backend.data.qvel, v)
        np.testing.assert_allclose(backend.data.ctrl[backend._arm_actuators], q[backend._arm_qpos])
    finally:
        backend.disconnect()


def test_wait_timeout_does_not_force_position():
    backend = MujocoBackend(wrist_camera=False)
    backend.connect()
    try:
        backend.move_joints([1., 1., -1., .4, .4, .4])
        with pytest.raises(TimeoutError):
            backend.wait_until_idle(timeout=.002)
        assert np.max(np.abs(backend.state().joints.positions - backend.data.ctrl[:6])) > .5
    finally:
        backend.disconnect()


def test_continuous_physics_tracks_target_without_polling_or_mount_collision():
    backend = MujocoBackend(wrist_camera=False)
    backend.connect()
    try:
        target = np.array([.2, .8, -1.2, .2, -.3, .4])
        backend.move_joints(target)
        for _ in range(1000):
            backend.step()
            # These are the applied joint actuator forces, including gravity
            # compensation, not just the position servo's unclamped request.
            force = backend.data.qfrc_actuator[backend._arm_dofs]
            assert np.all(np.abs(force) <= 100 + 1e-8)
        np.testing.assert_allclose(backend.data.qpos[backend._arm_qpos], target, atol=1e-3)
        assert not backend.state().moving
        for contact in backend.data.contact[:backend.data.ncon]:
            names = {backend.model.geom(int(g)).name for g in (contact.geom1, contact.geom2)}
            assert names != {"base_link_geom0", "link1_geom0"}
    finally:
        backend.disconnect()
