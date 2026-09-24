from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import math
import time

import numpy as np

from ..api.types import JointState, Pose, RobotState
from ..backends.joint_trajectory import JointTrajectory, positive_duration
from ..errors import BackendUnavailableError, IKError, NotConnectedError
from .ik import PinocchioIK
from .scene_builder import compile_scene, validate_scene

ASSETS = Path(__file__).resolve().parents[3] / "vendor/fr3_d435i"
DEFAULT_MODEL = ASSETS / "mujoco_fr3/fr3.xml"
DEFAULT_URDF = ASSETS / "urdf/fr3_d435i_official.urdf"
ARM_JOINTS = tuple(f"fr3_joint{index}" for index in range(1, 8))
FINGER_JOINTS = ("fr3_finger_joint1", "fr3_finger_joint2")


class MujocoBackend:
    camera_name = "d435i_check"

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, realtime: bool = False,
                 ik_urdf: str | Path = DEFAULT_URDF, scene: str | Path | None = None,
                 motion_duration_s: float = 4.0):
        self.model_path = Path(model_path)
        self.scene_path = validate_scene(scene)
        self.realtime = realtime
        self.motion_duration_s = positive_duration(motion_duration_s)
        self._motion: JointTrajectory | None = None
        self.ik_urdf = Path(ik_urdf)
        self.model = self.data = self.ik = None
        self._connected = False

    def connect(self) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise BackendUnavailableError("Install robot-control[mujoco] to use MuJoCo") from exc
        if not self.model_path.is_file():
            raise BackendUnavailableError(f"MuJoCo model not found: {self.model_path}")
        self.model = compile_scene(self.model_path, self.scene_path)
        self.data = mujoco.MjData(self.model)
        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home)
        joint_ids = np.array([self.model.joint(name).id for name in ARM_JOINTS])
        finger_ids = np.array([self.model.joint(name).id for name in FINGER_JOINTS])
        self._arm_qpos = self.model.jnt_qposadr[joint_ids].copy()
        self._arm_dofs = self.model.jnt_dofadr[joint_ids].copy()
        self._arm_actuators = np.array([self.model.actuator(name).id for name in ARM_JOINTS])
        self._finger_qpos = self.model.jnt_qposadr[finger_ids].copy()
        self._finger_dofs = self.model.jnt_dofadr[finger_ids].copy()
        self._finger_actuators = np.array([self.model.actuator(name).id for name in FINGER_JOINTS])
        self._gripper_max_width = float(sum(self.model.jnt_range[finger_ids, 1]))
        self.ik = PinocchioIK(self.ik_urdf)
        self.ik.lower = np.maximum(self.ik.lower, self.model.jnt_range[joint_ids, 0])
        self.ik.upper = np.minimum(self.ik.upper, self.model.jnt_range[joint_ids, 1])
        if np.any(self.ik.lower >= self.ik.upper):
            raise ValueError("Pinocchio and MuJoCo joint limits do not overlap")
        mujoco.mj_forward(self.model, self.data)
        pin = self.ik.pin
        mount = self.model.body("fr3_mount").id
        self._world_from_ik = pin.SE3(self.data.xmat[mount].reshape(3, 3).copy(),
                                     self.data.xpos[mount].copy())
        scratch = mujoco.MjData(self.model)
        for joints in (self.data.qpos[self._arm_qpos].copy(),
                       np.array([0.1, -0.2, 0.15, -1.4, 0.2, 1.5, -0.7])):
            scratch.qpos[self._arm_qpos] = joints
            mujoco.mj_forward(self.model, scratch)
            expected = self._transform_ik_pose(self.ik.forward(joints))
            body = self.model.body("fr3_link7").id
            if (not np.allclose(expected.position, scratch.xpos[body], atol=1e-6, rtol=0)
                    or abs(float(np.dot(expected.quaternion, scratch.xquat[body]))) < 1 - 1e-8):
                raise ValueError("Pinocchio URDF and MuJoCo FR3 kinematics disagree")
        self._connected = True

    def _transform_ik_pose(self, pose: Pose, *, inverse: bool = False) -> Pose:
        pin = self.ik.pin
        quaternion = np.asarray(pose.quaternion, dtype=float)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(pose.position).all() or not np.isfinite(norm) or norm < 1e-9:
            raise ValueError("Pose must contain finite position and quaternion")
        placement = pin.SE3(pin.Quaternion(*(quaternion / norm)).matrix(), np.asarray(pose.position))
        transform = self._world_from_ik.inverse() if inverse else self._world_from_ik
        result = transform * placement
        quat = pin.Quaternion(result.rotation)
        return Pose(tuple(result.translation), (quat.w, quat.x, quat.y, quat.z))

    def _require(self) -> None:
        if not self._connected:
            raise NotConnectedError("MuJoCo backend is not connected")

    def disconnect(self) -> None:
        self._connected = False
        self._motion = None
        self.model = self.data = self.ik = None

    def _step(self, steps: int = 1, *, pace: bool = True) -> None:
        self._require()
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        import mujoco
        started = time.monotonic()
        for _ in range(steps):
            if self._motion is not None:
                next_time = self.data.time + self.model.opt.timestep
                self.data.ctrl[self._arm_actuators] = self._motion.at(next_time)
                if self._motion.finished(next_time):
                    self._motion = None
            mujoco.mj_step(self.model, self.data)
        if self.realtime and pace:
            time.sleep(max(0, self.model.opt.timestep * steps - (time.monotonic() - started)))

    def step(self, steps: int = 1) -> None:
        self._step(steps)

    def wait_until_idle(self, timeout: float = 10.0) -> None:
        self._require()
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        for _ in range(int(math.ceil(timeout / self.model.opt.timestep))):
            if not self.state().moving:
                return
            self._step(1)
        if self.state().moving:
            raise TimeoutError("MuJoCo motion did not settle before the timeout")

    def state(self) -> RobotState:
        self._require()
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        body = self.model.body("fr3_link7").id
        pose = Pose(tuple(self.data.xpos[body]), tuple(self.data.xquat[body]))
        fingers = self.data.qpos[self._finger_qpos]
        joints = JointState(self.data.qpos[self._arm_qpos].copy(),
                            self.data.qvel[self._arm_dofs].copy(), float(sum(fingers)))
        moving = bool(self._motion is not None
                      or np.max(np.abs(self.data.ctrl[self._arm_actuators] - joints.positions)) > 1e-3
                      or np.max(np.abs(joints.velocities)) > 1e-2
                      or np.max(np.abs(self.data.ctrl[self._finger_actuators] - fingers)) > 2e-4
                      or np.max(np.abs(self.data.qvel[self._finger_dofs])) > 2e-3)
        return RobotState(True, moving, joints, pose)

    def move_joints(self, joints, duration_s: float | None = None) -> None:
        self._require()
        values = np.asarray(joints, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("move_joints requires seven finite joint values in radians")
        if np.any(values < self.ik.lower) or np.any(values > self.ik.upper):
            raise ValueError("joint target is outside FR3 joint limits")
        self._motion = JointTrajectory(self.data.qpos[self._arm_qpos], values, self.data.time,
                                       self.motion_duration_s if duration_s is None else duration_s)
        self.data.ctrl[self._arm_actuators] = self._motion.start

    def move_p(self, pose: Pose, duration_s: float | None = None) -> None:
        self._require()
        result = self.ik.solve(self._transform_ik_pose(pose, inverse=True),
                               seed=self.data.qpos[self._arm_qpos])
        if not result.success:
            raise IKError(f"Pinocchio IK failed: {result.message}; position={result.position_error:.6g}; orientation={result.orientation_error:.6g}")
        self.move_joints(result.joints, duration_s=duration_s)

    def gripper(self, width: float, effort: float | None = None) -> None:
        self._require()
        if not np.isfinite(width) or not 0 <= width <= self._gripper_max_width:
            raise ValueError(f"gripper width must be between 0 and {self._gripper_max_width:g} metres")
        self.data.ctrl[self._finger_actuators] = width / 2

    def stop(self) -> None:
        self._require()
        self._motion = None
        self.data.ctrl[self._arm_actuators] = self.data.qpos[self._arm_qpos]
        self.data.ctrl[self._finger_actuators] = self.data.qpos[self._finger_qpos]

    def run_gui(self, duration: float = 0.0, lock=None) -> None:
        self._require()
        import mujoco.viewer
        started = previous = time.monotonic()
        remainder = 0.0
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                frame_start = time.monotonic()
                if duration > 0 and frame_start - started >= duration:
                    break
                remainder += min(frame_start - previous, 0.1)
                previous = frame_start
                steps = int(remainder / self.model.opt.timestep)
                remainder -= steps * self.model.opt.timestep
                with lock if lock is not None else nullcontext():
                    self._step(steps, pace=False)
                    viewer.sync()
                time.sleep(max(0, 1 / 60 - (time.monotonic() - frame_start)))
