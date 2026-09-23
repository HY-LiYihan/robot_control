from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraExtrinsics:
    rotation: tuple[float, ...]
    translation: tuple[float, float, float]
    reference_frame: str = "base_link"
    camera_frame: str = "d435i_color_optical_frame"
    timestamp: float | None = None


@dataclass
class RGBDFrame:
    color: np.ndarray
    depth: np.ndarray
    timestamp: float
    frame_id: str
    intrinsics: CameraIntrinsics
    depth_scale: float
    extrinsics: CameraExtrinsics | None = None

    def __post_init__(self):
        if self.color.ndim != 3 or self.color.shape[2] != 3 or self.color.dtype != np.uint8:
            raise ValueError("color must be uint8 HxWx3 RGB")
        if self.depth.ndim != 2 or self.depth.shape != self.color.shape[:2]:
            raise ValueError("depth must be HxW and match color resolution")
