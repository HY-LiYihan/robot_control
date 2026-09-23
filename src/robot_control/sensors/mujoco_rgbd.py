from __future__ import annotations

import time
import numpy as np
from .frame import CameraIntrinsics, RGBDFrame
from .extrinsics import camera_extrinsics
from ..errors import BackendUnavailableError


class MujocoRGBDCamera:
    def __init__(self, model, data, camera="d435i_color_optical_camera", width=1280, height=720,
                 frame_id="d435i_color_optical_frame"):
        self.model, self.data, self.camera = model, data, camera
        self.width, self.height, self.frame_id = width, height, frame_id
        self._renderer = None

    def connect(self):
        try:
            import mujoco
            # The upstream XML uses MuJoCo's 640x480 default offscreen buffer.
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.height)
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera) < 0:
                raise ValueError(f"MuJoCo D435i camera not found: {self.camera}")
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        except ImportError as exc:
            raise BackendUnavailableError("Install robot-control[mujoco] for MuJoCo RGB-D") from exc

    def disconnect(self):
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None

    def read(self) -> RGBDFrame:
        if self._renderer is None:
            raise RuntimeError("camera is not connected")
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera=self.camera)
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        # MuJoCo returns metric distance for depth rendering.
        depth = np.asarray(depth, dtype=np.float32)
        self._renderer.update_scene(self.data, camera=self.camera)
        color = np.asarray(self._renderer.render(), dtype=np.uint8)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        fovy = np.deg2rad(self.model.cam_fovy[cam_id])
        fy = self.height / (2 * np.tan(fovy / 2))
        fx = fy
        intr = CameraIntrinsics(self.width, self.height, fx, fy, self.width / 2, self.height / 2)
        fr3 = self.camera == "d435i_check"
        base_name = "fr3_link0" if fr3 else "base_link"
        base = self.model.body(base_name).id
        base_rotation = self.data.xmat[base].reshape(3, 3)
        if fr3:
            optical = self.model.site("d435i_color_optical_frame").id
            optical_position = self.data.site_xpos[optical]
            optical_rotation = self.data.site_xmat[optical].reshape(3, 3)
            frame_id = "d435i_color_optical_frame"
        else:
            optical = self.model.body("d435i_color_optical_frame").id
            optical_position = self.data.xpos[optical]
            optical_rotation = self.data.xmat[optical].reshape(3, 3)
            frame_id = self.frame_id
        transform = np.eye(4)
        transform[:3, :3] = base_rotation.T @ optical_rotation
        transform[:3, 3] = base_rotation.T @ (optical_position - self.data.xpos[base])
        timestamp = time.time()
        extrinsics = camera_extrinsics(transform, base_name, frame_id, timestamp)
        return RGBDFrame(color, depth, timestamp, frame_id, intr, 1.0, extrinsics)
