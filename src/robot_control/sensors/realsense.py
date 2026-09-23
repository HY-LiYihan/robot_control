from __future__ import annotations

import time
from ..errors import BackendUnavailableError
from .frame import CameraIntrinsics, RGBDFrame


class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30, frame_id="camera_color_optical_frame"):
        self.width, self.height, self.fps, self.frame_id = width, height, fps, frame_id
        self._pipeline = self._align = self._profile = None

    def connect(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise BackendUnavailableError("Install robot-control[camera] for RealSense support") from exc
        self._rs = rs
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)

    def disconnect(self):
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = self._align = self._profile = None

    def read(self, timeout_ms=5000) -> RGBDFrame:
        if self._pipeline is None:
            raise RuntimeError("camera is not connected")
        frames = self._align.process(self._pipeline.wait_for_frames(timeout_ms))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("RealSense returned an incomplete RGB-D frame")
        color_profile = color.profile.as_video_stream_profile()
        intr = color_profile.intrinsics
        depth_scale = self._profile.get_device().first_depth_sensor().get_depth_scale()
        import numpy as np
        return RGBDFrame(
            color=np.asanyarray(color.get_data()), depth=np.asanyarray(depth.get_data()), timestamp=time.time(), frame_id=self.frame_id,
            intrinsics=CameraIntrinsics(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy),
            depth_scale=depth_scale,
        )
