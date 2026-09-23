import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_control.backends.mujoco import MujocoBackend
from robot_control.sensors.mujoco_rgbd import MujocoRGBDCamera
from robot_control.sensors.extrinsics import piper_link6_to_color_optical, pose_matrix


FRAME_NAMES = (
    "d435i_link", "d435i_depth_frame", "d435i_depth_optical_frame",
    "d435i_infra1_frame", "d435i_infra1_optical_frame", "d435i_infra2_frame",
    "d435i_infra2_optical_frame", "d435i_color_frame", "d435i_color_optical_frame",
    "d435i_accel_frame", "d435i_accel_optical_frame", "d435i_gyro_frame",
    "d435i_gyro_optical_frame",
)


def test_mujoco_rgbd_contract():
    backend = MujocoBackend(settle_steps=1)
    backend.connect()
    try:
        assert backend.model.nq == 8
        assert backend.model.nu == 8
        assert backend.model.ncam == 2
        assert backend.model.camera("d435i_color_optical_camera").id >= 0
        assert backend.model.camera("d435i_depth_optical_camera").id >= 0
        assert all(backend.model.geom(i).name != "d435i_collision" for i in range(backend.model.ngeom))
        color_cam = backend.model.camera("d435i_color_optical_camera").id
        depth_cam = backend.model.camera("d435i_depth_optical_camera").id
        np.testing.assert_allclose(backend.data.cam_xpos[color_cam], backend.data.cam_xpos[depth_cam], atol=1e-12)
        np.testing.assert_allclose(backend.data.cam_xmat[color_cam], backend.data.cam_xmat[depth_cam], atol=1e-12)
        for name in FRAME_NAMES:
            assert backend.model.body(name).id >= 0

        link6_id = backend.model.body("link6").id
        stand_id = backend.model.body("camera_stand_link").id
        camera_id = backend.model.body("d435i_link").id
        stand_offset = backend.data.xpos[stand_id] - backend.data.xpos[link6_id]
        camera_offset = backend.data.xpos[camera_id] - backend.data.xpos[link6_id]
        assert np.linalg.norm(stand_offset) < 0.05
        assert np.linalg.norm(camera_offset) < 0.08
        assert abs(float(stand_offset[1])) < 0.05
        assert abs(float(stand_offset[2])) < 0.05

        stand_mesh = backend.model.mesh("d435i_printed_stand").id
        start = backend.model.mesh_vertadr[stand_mesh]
        count = backend.model.mesh_vertnum[stand_mesh]
        vertices = np.asarray(backend.model.mesh_vert[start:start + count])
        size = np.ptp(vertices, axis=0)
        assert np.all(size > 0.02)
        assert np.max(size) < 0.15

        camera = MujocoRGBDCamera(backend.model, backend.data)
        camera.connect()
        try:
            frame = camera.read()
            assert frame.extrinsics.reference_frame == "base_link"
            assert frame.extrinsics.camera_frame == frame.frame_id
            actual = np.eye(4)
            actual[:3, :3] = np.asarray(frame.extrinsics.rotation).reshape(3, 3)
            actual[:3, 3] = frame.extrinsics.translation
            expected = pose_matrix(backend.state().pose) @ piper_link6_to_color_optical()
            base = backend.model.body("base_link").id
            base_pose = np.eye(4)
            base_pose[:3, :3] = backend.data.xmat[base].reshape(3, 3)
            base_pose[:3, 3] = backend.data.xpos[base]
            np.testing.assert_allclose(actual, np.linalg.inv(base_pose) @ expected, atol=2e-7)
            assert frame.color.shape == (720, 1280, 3)
            assert frame.color.dtype == np.uint8
            assert frame.depth.shape == (720, 1280)
            assert frame.depth.dtype == np.float32
            assert frame.depth_scale == 1.0
            assert np.isfinite(frame.depth).all()
            assert np.any(frame.depth > 0.0)
        finally:
            camera.disconnect()
    finally:
        backend.disconnect()


def test_camera_attachment_does_not_change_piper_fk():
    q = np.array([0.1, 0.4, -1.1, 0.2, -0.3, 0.7])
    plain = MujocoBackend(wrist_camera=False, settle_steps=1)
    with_camera = MujocoBackend(wrist_camera=True, settle_steps=1)
    plain.connect()
    with_camera.connect()
    try:
        # Static camera geometry test: seed test states directly, outside the
        # command API whose targets now require physical stepping.
        import mujoco
        for backend in (plain, with_camera):
            backend.data.qpos[backend._arm_qpos] = q
            mujoco.mj_forward(backend.model, backend.data)
        plain_link6 = plain.model.body("link6").id
        camera_link6 = with_camera.model.body("link6").id
        np.testing.assert_allclose(plain.data.xpos[plain_link6], with_camera.data.xpos[camera_link6], atol=1e-12)
        np.testing.assert_allclose(plain.data.xmat[plain_link6], with_camera.data.xmat[camera_link6], atol=1e-12)
    finally:
        plain.disconnect()
        with_camera.disconnect()


def test_camera_mount_is_fixed_to_link6_across_joint_angles():
    backend = MujocoBackend(settle_steps=1)
    backend.connect()
    try:
        link6 = backend.model.body("link6").id
        camera = backend.model.body("d435i_link").id
        stand = backend.model.body("camera_stand_link").id
        relative = []
        for q in (np.array([0.0, 0.2, -0.8, 0.0, 0.0, 0.0]), np.array([0.4, 0.8, -1.5, 0.2, -0.4, 1.0])):
            import mujoco
            backend.data.qpos[backend._arm_qpos] = q
            mujoco.mj_forward(backend.model, backend.data)
            link_rotation = backend.data.xmat[link6].reshape(3, 3)
            relative.append((
                link_rotation.T @ (backend.data.xpos[camera] - backend.data.xpos[link6]),
                link_rotation.T @ (backend.data.xpos[stand] - backend.data.xpos[link6]),
            ))
        np.testing.assert_allclose(relative[0][0], relative[1][0], atol=1e-10)
        np.testing.assert_allclose(relative[0][1], relative[1][1], atol=1e-10)
    finally:
        backend.disconnect()
