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
    """Joint-limited SE(3) IK on an independent URDF model, never simulation data."""

    def __init__(self, urdf_path: str | Path, damping: float = 1e-3):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise BackendUnavailableError("Install robot-control[mujoco] (the 'pin' package) for IK") from exc
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("link6")
        if self.model.nq != 6 or self.model.nv != 6 or self.frame_id >= self.model.nframes:
            raise ValueError("IK requires the six-axis Piper arm URDF with a link6 frame")
        names = [f"joint{i}" for i in range(1, 7)]
        if any(not self.model.existJointName(name) for name in names):
            raise ValueError("IK URDF must contain joint1 through joint6")
        joints = [self.model.joints[self.model.getJointId(name)] for name in names]
        if any(j.nq != 1 or j.nv != 1 for j in joints):
            raise ValueError("Piper IK requires six scalar revolute joints")
        self._q_indices = np.array([j.idx_q for j in joints])
        self._v_indices = np.array([j.idx_v for j in joints])
        self.lower = self.model.lowerPositionLimit[self._q_indices].copy()
        self.upper = self.model.upperPositionLimit[self._q_indices].copy()
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("IK damping must be positive and finite")
        self.damping = damping

    def _configuration(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("IK requires six finite joint angles")
        q = np.empty(6)
        q[self._q_indices] = joints
        return q

    def _placement(self, q):
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id]

    def forward(self, joints) -> Pose:
        placement = self._placement(self._configuration(joints))
        quat = self.pin.Quaternion(placement.rotation)
        return Pose(tuple(placement.translation), (quat.w, quat.x, quat.y, quat.z))

    def solve(self, target: Pose, seed: np.ndarray | None = None,
              enforce_limits: bool = True, max_iterations: int = 250,
              tolerance: float = 1e-4, orientation_tolerance: float = 1e-3) -> IKResult:
        position = np.asarray(target.position, dtype=float)
        quat = np.asarray(target.quaternion, dtype=float)
        if not np.isfinite(position).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-9:
            raise ValueError("IK target must contain a finite position and nonzero finite quaternion")
        if max_iterations < 1 or not np.isfinite([tolerance, orientation_tolerance]).all() or min(tolerance, orientation_tolerance) <= 0:
            raise ValueError("IK iteration count and tolerances must be positive")
        quat /= np.linalg.norm(quat)
        desired = self.pin.SE3(self.pin.Quaternion(*quat).matrix(), position)
        initial = np.zeros(6) if seed is None else np.asarray(seed, dtype=float).copy()
        self._configuration(initial)  # validate before any attempts
        if enforce_limits:
            initial = np.clip(initial, self.lower, self.upper)
        # Try the measured configuration first to preserve the local solution.
        # Deterministic restarts help escape singularities and active joint limits.
        midpoint = (self.lower + self.upper) / 2
        seeds = [initial, midpoint, midpoint + np.array([0, -.4, .4, 0, .3, 0])]
        weights = np.array([1., 1., 1., .3, .3, .3])
        best = None
        iterations = 0

        def residual(joints):
            current = self._placement(self._configuration(joints))
            transform = current.actInv(desired)
            error = self.pin.log6(transform).vector.copy()
            pos_error = float(np.linalg.norm(desired.translation - current.translation))
            rot_error = float(np.linalg.norm(self.pin.log3(transform.rotation)))
            return transform, error, pos_error, rot_error

        for start in seeds:
            q = np.clip(start, self.lower, self.upper) if enforce_limits else start.copy()
            damping = self.damping
            for _ in range(max_iterations):
                iterations += 1
                transform, error, pos_error, rot_error = residual(q)
                cost = float(np.linalg.norm(weights * error))
                if best is None or cost < best[0]:
                    best = (cost, q.copy(), pos_error, rot_error)
                if pos_error <= tolerance and rot_error <= orientation_tolerance:
                    return IKResult(q.copy(), True, pos_error, rot_error, iterations, "converged")
                jacobian = self.pin.computeFrameJacobian(
                    self.model, self.data, self._configuration(q), self.frame_id,
                    self.pin.ReferenceFrame.LOCAL,
                )[:, self._v_indices]
                jacobian = -self.pin.Jlog6(transform.inverse()) @ jacobian
                weighted_jac = weights[:, None] * jacobian
                step = -weighted_jac.T @ np.linalg.solve(
                    weighted_jac @ weighted_jac.T + damping ** 2 * np.eye(6), weights * error,
                )
                step *= min(1., .2 / max(float(np.max(np.abs(step))), 1e-12))
                accepted = False
                for scale in (1., .5, .25, .125, .0625):
                    candidate = q + scale * step
                    if enforce_limits:
                        candidate = np.clip(candidate, self.lower, self.upper)
                    _, trial, trial_pos, trial_rot = residual(candidate)
                    trial_cost = float(np.linalg.norm(weights * trial))
                    if trial_cost < cost - 1e-12:
                        q = candidate
                        if trial_cost < best[0]:
                            best = (trial_cost, q.copy(), trial_pos, trial_rot)
                        damping = max(self.damping, damping * .5)
                        accepted = True
                        break
                if not accepted:
                    damping *= 5
                    if damping > 1.:
                        break
        _, q, pos_error, rot_error = best
        success = pos_error <= tolerance and rot_error <= orientation_tolerance
        return IKResult(q, success, pos_error, rot_error, iterations,
                        "converged" if success else "no converged solution within joint limits")
