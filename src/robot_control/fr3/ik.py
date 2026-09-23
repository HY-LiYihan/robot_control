from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..api.types import Pose
from ..errors import BackendUnavailableError


@dataclass
class IKResult:
    joints: np.ndarray
    success: bool
    position_error: float
    orientation_error: float
    iterations: int
    message: str = ""


class PinocchioIK:
    """Independent seven-DOF joint-limited solver on the supplied FR3 URDF."""

    def __init__(self, urdf_path: str | Path, damping: float = 1e-3):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise BackendUnavailableError("Install robot-control[mujoco] for Pinocchio IK") from exc
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("fr3_link7")
        names = [f"fr3_joint{index}" for index in range(1, 8)]
        if (self.model.nq != 9 or self.model.nv != 9 or self.frame_id >= self.model.nframes
                or any(not self.model.existJointName(name) for name in names)):
            raise ValueError("IK requires the supplied FR3 arm and two-finger URDF")
        joints = [self.model.joints[self.model.getJointId(name)] for name in names]
        self._q_indices = np.array([joint.idx_q for joint in joints])
        self._v_indices = np.array([joint.idx_v for joint in joints])
        self.lower = self.model.lowerPositionLimit[self._q_indices].copy()
        self.upper = self.model.upperPositionLimit[self._q_indices].copy()
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("IK damping must be finite and positive")
        self.damping = damping

    def _configuration(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("IK requires seven finite joint angles")
        configuration = self.pin.neutral(self.model)
        configuration[self._q_indices] = joints
        return configuration

    def _placement(self, joints):
        configuration = self._configuration(joints)
        self.pin.forwardKinematics(self.model, self.data, configuration)
        self.pin.updateFramePlacements(self.model, self.data)
        return configuration, self.data.oMf[self.frame_id]

    def forward(self, joints) -> Pose:
        _, placement = self._placement(joints)
        quat = self.pin.Quaternion(placement.rotation)
        return Pose(tuple(placement.translation), (quat.w, quat.x, quat.y, quat.z))

    def solve(self, target: Pose, seed=None, max_iterations: int = 300,
              tolerance: float = 1e-4, orientation_tolerance: float = 1e-3) -> IKResult:
        position = np.asarray(target.position, dtype=float)
        quaternion = np.asarray(target.quaternion, dtype=float)
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-9:
            raise ValueError("IK target must be finite")
        quaternion /= np.linalg.norm(quaternion)
        desired = self.pin.SE3(self.pin.Quaternion(*quaternion).matrix(), position)
        initial = np.mean([self.lower, self.upper], axis=0) if seed is None else np.asarray(seed, dtype=float).copy()
        self._configuration(initial)
        seeds = [initial, (self.lower + self.upper) / 2]
        best = None
        iterations = 0
        for start in seeds:
            joints = np.clip(start, self.lower, self.upper)
            for _ in range(max_iterations):
                iterations += 1
                configuration, current = self._placement(joints)
                transform = current.actInv(desired)
                error = self.pin.log6(transform).vector.copy()
                position_error = float(np.linalg.norm(desired.translation - current.translation))
                orientation_error = float(np.linalg.norm(self.pin.log3(transform.rotation)))
                cost = float(np.linalg.norm(error))
                if best is None or cost < best[0]:
                    best = cost, joints.copy(), position_error, orientation_error
                if position_error <= tolerance and orientation_error <= orientation_tolerance:
                    return IKResult(joints.copy(), True, position_error, orientation_error, iterations, "converged")
                jacobian = self.pin.computeFrameJacobian(
                    self.model, self.data, configuration, self.frame_id, self.pin.ReferenceFrame.LOCAL,
                )[:, self._v_indices]
                jacobian = -self.pin.Jlog6(transform.inverse()) @ jacobian
                step = -jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + self.damping ** 2 * np.eye(6), error)
                step *= min(1, 0.15 / max(float(np.max(np.abs(step))), 1e-12))
                accepted = False
                for scale in (1, .5, .25, .125, .0625):
                    candidate = np.clip(joints + scale * step, self.lower, self.upper)
                    _, trial = self._placement(candidate)
                    if np.linalg.norm(self.pin.log6(trial.actInv(desired)).vector) < cost - 1e-12:
                        joints = candidate
                        accepted = True
                        break
                if not accepted:
                    break
        _, joints, position_error, orientation_error = best
        return IKResult(joints, False, position_error, orientation_error, iterations, "no solution within joint limits")
