from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..api.types import Pose
from .frame import CameraExtrinsics


ORDINARY_LINK6_FROM_V100_LINK6 = np.array([
    [0.0, -1.0, 0.0, -0.00009777],
    [1.0, 0.0, 0.0, 0.00141648],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
PIPER_MOUNT_V100_POSITION = (-0.0315, 0.064, 0.027)
PIPER_MOUNT_V100_RPY = (0.0, -1.22, -1.57)
PIPER_CAMERA_LINK_OFFSET = (0.0106, 0.0175, 0.0125)
PIPER_COLOR_FRAME_OFFSET = (0.0, 0.015, 0.0)
PIPER_OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)


def rpy_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def pose_matrix(pose: Pose) -> np.ndarray:
    quaternion = np.asarray(pose.quaternion, dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    transform = np.eye(4)
    transform[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    transform[:3, 3] = pose.position
    return transform


def piper_link6_to_color_optical() -> np.ndarray:
    mount = np.eye(4)
    mount[:3, :3] = rpy_rotation(*PIPER_MOUNT_V100_RPY)
    mount[:3, 3] = PIPER_MOUNT_V100_POSITION
    camera_link = np.eye(4)
    camera_link[:3, 3] = PIPER_CAMERA_LINK_OFFSET
    color = np.eye(4)
    color[:3, 3] = PIPER_COLOR_FRAME_OFFSET
    optical = np.eye(4)
    optical[:3, :3] = rpy_rotation(*PIPER_OPTICAL_RPY)
    return ORDINARY_LINK6_FROM_V100_LINK6 @ mount @ camera_link @ color @ optical


@lru_cache(maxsize=1)
def fr3_link7_to_color_optical() -> np.ndarray:
    urdf = Path(__file__).resolve().parents[3] / "vendor/fr3_d435i/urdf/fr3_d435i_official.urdf"
    joints = {joint.find("child").get("link"): joint
              for joint in ET.parse(urdf).getroot().findall("joint")}
    transforms = []
    link = "d435i_color_optical_frame"
    while link != "fr3_link7":
        joint = joints[link]
        if joint.get("type") != "fixed":
            raise ValueError(f"FR3 camera joint {joint.get('name')} must be fixed")
        origin = joint.find("origin")
        transform = np.eye(4)
        if origin is not None:
            transform[:3, :3] = rpy_rotation(*(float(value) for value in origin.get("rpy", "0 0 0").split()))
            transform[:3, 3] = [float(value) for value in origin.get("xyz", "0 0 0").split()]
        transforms.append(transform)
        link = joint.find("parent").get("link")
    result = np.eye(4)
    for transform in reversed(transforms):
        result = result @ transform
    return result


def camera_extrinsics(transform: np.ndarray, reference_frame: str, camera_frame: str,
                      timestamp: float | None = None) -> CameraExtrinsics:
    return CameraExtrinsics(tuple(float(value) for value in transform[:3, :3].flat),
                            tuple(float(value) for value in transform[:3, 3]),
                            reference_frame, camera_frame, timestamp)
