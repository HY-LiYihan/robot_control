from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
import tempfile
import time
import xml.etree.ElementTree as ET
import math
import numpy as np
from ..api.types import JointState, Pose, RobotState
from ..config import PIPER_INITIAL_JOINTS_RAD
from ..errors import BackendUnavailableError, IKError, NotConnectedError
from ..kinematics.ik import PinocchioIK
from ..sensors.extrinsics import (ORDINARY_LINK6_FROM_V100_LINK6,
                                  PIPER_MOUNT_V100_POSITION, PIPER_MOUNT_V100_RPY,
                                  PIPER_CAMERA_LINK_OFFSET, PIPER_COLOR_FRAME_OFFSET,
                                  PIPER_OPTICAL_RPY)
from .piper_model import ASSET_ROOT, DEFAULT_MODEL, ARM_JOINTS, FINGER_JOINTS, build_piper_scene
from .joint_trajectory import JointTrajectory, positive_duration
from .scene_builder import MOUNT_NAME, compile_scene, validate_scene


WRIST_D435_MESH = Path(__file__).parents[3] / "vendor/piper_isaac_sim/realsense2_description/meshes/d435.dae"
WRIST_STAND_MESH = Path(__file__).parents[3] / "vendor/piper_isaac_sim/piper_description/meshes/dae/realsense_mid_stand.dae"

# The upstream Isaac asset is authored against its V100 wrist frame.  The
# ordinary Piper MuJoCo model has a fixed tool-frame rotation relative to that
# frame; this constant converts the official V100 wrist mount into link6 of the
# ordinary Piper without changing the arm's own kinematics.
class MujocoBackend:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, realtime: bool = False,
                 settle_steps: int = 200, wrist_camera: bool = True,
                 ik_urdf: str | Path = ASSET_ROOT / "piper/urdf/piper_description.urdf",
                 scene: str | Path | None = None, motion_duration_s: float = 2.0, **_: object):
        self.model_path = Path(model_path)
        self.scene_path = None
        if self.model_path.suffix in (".urdf", ".xacro"):
            self.scene_path = validate_scene(scene)
        elif scene is not None:
            raise ValueError("scene requires the Piper URDF/Xacro component; a custom model_path XML is already a complete model")
        self.realtime = realtime
        self.motion_duration_s = positive_duration(motion_duration_s)
        self._motion: JointTrajectory | None = None
        if settle_steps < 1:
            raise ValueError("settle_steps must be positive")
        self.settle_steps = settle_steps
        self.wrist_camera = wrist_camera
        self.ik_urdf = Path(ik_urdf)
        self._temporary_files: list[Path] = []
        self.model = self.data = self.ik = None
        self._connected = False

    def connect(self) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise BackendUnavailableError("Install robot-control[mujoco] to use MuJoCo") from exc
        if not self.model_path.exists():
            raise BackendUnavailableError(f"MuJoCo model not found: {self.model_path}")
        temporary_model = None
        try:
            if self.model_path.suffix in (".urdf", ".xacro"):
                tree = build_piper_scene(self.model_path)
                if self.wrist_camera:
                    self._add_wrist_camera(tree)
                self.model = compile_scene(tree, self.scene_path)
            else:
                # Preserve explicit MJCF overrides, including relative mesh paths.
                if self.wrist_camera:
                    tree = ET.parse(self.model_path)
                    self._add_wrist_camera(tree)
                    with tempfile.NamedTemporaryFile(prefix="piper_wrist_", suffix=".xml",
                                                     dir=self.model_path.parent, delete=False) as file:
                        temporary_model = Path(file.name)
                        tree.write(file, encoding="utf-8", xml_declaration=True)
                self.model = mujoco.MjModel.from_xml_path(str(temporary_model or self.model_path))
        finally:
            if temporary_model is not None:
                temporary_model.unlink(missing_ok=True)
            for temporary_file in self._temporary_files:
                temporary_file.unlink(missing_ok=True)
            self._temporary_files.clear()
        self.data = mujoco.MjData(self.model)
        joint_ids = np.array([self.model.joint(name).id for name in ARM_JOINTS])
        self._arm_qpos = self.model.jnt_qposadr[joint_ids].copy()
        self._arm_dofs = self.model.jnt_dofadr[joint_ids].copy()
        self._arm_actuators = np.array([self.model.actuator(name).id for name in ARM_JOINTS])
        finger_names = FINGER_JOINTS if self.model_path.suffix in (".urdf", ".xacro") else ("joint7", "joint8")
        finger_ids = np.array([self.model.joint(name).id for name in finger_names])
        self._finger_qpos = self.model.jnt_qposadr[finger_ids].copy()
        self._finger_dofs = self.model.jnt_dofadr[finger_ids].copy()
        self._finger_actuators = np.array([self.model.actuator(name).id for name in finger_names])
        finger_limits = self.model.jnt_range[finger_ids]
        self._gripper_max_width = float(min(finger_limits[0, 1], -finger_limits[1, 0]) * 2)
        if self.model_path.suffix in (".urdf", ".xacro"):
            opening = 0.07 / 2
            self.data.qpos[self._finger_qpos] = (opening, -opening)
            self.data.ctrl[self._finger_actuators] = (opening, -opening)
        lower = self.model.jnt_range[joint_ids, 0].copy()
        upper = self.model.jnt_range[joint_ids, 1].copy()
        self.ik = PinocchioIK(self.ik_urdf)
        pin = self.ik.pin
        self._world_from_ik = pin.SE3.Identity()
        if self.scene_path is not None:
            mujoco.mj_forward(self.model, self.data)
            mount = self.model.body(MOUNT_NAME).id
            self._world_from_ik = pin.SE3(self.data.xmat[mount].reshape(3, 3).copy(),
                                         self.data.xpos[mount].copy())
        self.ik.lower = np.maximum(self.ik.lower, lower)
        self.ik.upper = np.minimum(self.ik.upper, upper)
        if np.any(self.ik.lower >= self.ik.upper):
            raise ValueError("Pinocchio and MuJoCo joint limits do not overlap")
        initial_joints = np.asarray(PIPER_INITIAL_JOINTS_RAD, dtype=float)
        if initial_joints.shape != (6,) or np.any(initial_joints < lower) or np.any(initial_joints > upper):
            raise ValueError("Piper initial joints are outside the MuJoCo joint limits")
        self.data.qpos[self._arm_qpos] = initial_joints
        self.data.ctrl[self._arm_actuators] = initial_joints
        # Check custom MJCF/URDF pairs on scratch data, never the live scene.
        scratch = mujoco.MjData(self.model)
        for q in (np.zeros(6), np.array([.2, .6, -1., .2, -.3, .4])):
            scratch.qpos[self._arm_qpos] = q
            mujoco.mj_forward(self.model, scratch)
            expected = self._transform_ik_pose(self.ik.forward(q))
            body = self.model.body("link6").id
            actual_quat = scratch.xquat[body]
            if (not np.allclose(expected.position, scratch.xpos[body], atol=1e-7, rtol=0)
                    or abs(float(np.dot(expected.quaternion, actual_quat))) < 1 - 1e-10):
                raise ValueError("Pinocchio URDF and MuJoCo link6 kinematics disagree; provide matching ik_urdf")
        mujoco.mj_forward(self.model, self.data)
        self._connected = True

    def _transform_ik_pose(self, pose: Pose, *, inverse: bool = False) -> Pose:
        """Convert between the scene world and the independent IK model frame."""
        pin = self.ik.pin
        quaternion = np.asarray(pose.quaternion, dtype=float)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(pose.position).all() or not np.isfinite(norm) or norm < 1e-9:
            raise ValueError("Pose must contain a finite position and nonzero finite quaternion")
        placement = pin.SE3(pin.Quaternion(*(quaternion / norm)).matrix(), np.asarray(pose.position))
        transform = self._world_from_ik.inverse() if inverse else self._world_from_ik
        result = transform * placement
        quat = pin.Quaternion(result.rotation)
        return Pose(tuple(result.translation), (quat.w, quat.x, quat.y, quat.z))

    def _add_wrist_camera(self, tree: ET.ElementTree) -> None:
        """Attach the retained Isaac D435i assets to the model's link6 frame."""
        root = tree.getroot()
        link6 = next((body for body in root.iter("body") if body.get("name") == "link6"), None)
        if link6 is None:
            raise BackendUnavailableError("Piper MuJoCo model has no link6 wrist body")
        if not WRIST_D435_MESH.exists() or not WRIST_STAND_MESH.exists():
            raise BackendUnavailableError("Pinned Piper D435i mesh assets are missing")
        def quat(rpy: list[float]) -> list[float]:
            roll, pitch, yaw = rpy
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]

        def values(items: list[float]) -> str:
            return " ".join(f"{item:.9g}" for item in items)

        def mat_mul(first, second):
            return [[sum(first[row][k] * second[k][column] for k in range(4))
                     for column in range(4)] for row in range(4)]

        def rpy_matrix(rpy):
            roll, pitch, yaw = rpy
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            cy, sy = math.cos(yaw), math.sin(yaw)
            return [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
                [-sp, cp * sr, cp * cr, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

        def transform_xyz_rpy(xyz, rpy):
            transform = rpy_matrix(rpy)
            for index, value in enumerate(xyz):
                transform[index][3] = value
            return transform

        def matrix_quat(transform):
            trace = transform[0][0] + transform[1][1] + transform[2][2]
            if trace > 0.0:
                scale = math.sqrt(trace + 1.0) * 2.0
                return [0.25 * scale,
                        (transform[2][1] - transform[1][2]) / scale,
                        (transform[0][2] - transform[2][0]) / scale,
                        (transform[1][0] - transform[0][1]) / scale]
            diagonal = [transform[0][0], transform[1][1], transform[2][2]]
            pivot = max(range(3), key=lambda index: diagonal[index])
            next_index = (pivot + 1) % 3
            last_index = (pivot + 2) % 3
            scale = math.sqrt(max(0.0, 1.0 + diagonal[pivot] - diagonal[next_index] - diagonal[last_index])) * 2.0
            result = [0.0, 0.0, 0.0, 0.0]
            result[pivot + 1] = 0.25 * scale
            result[0] = (transform[last_index][next_index] - transform[next_index][last_index]) / scale
            result[next_index + 1] = (transform[next_index][pivot] + transform[pivot][next_index]) / scale
            result[last_index + 1] = (transform[last_index][pivot] + transform[pivot][last_index]) / scale
            return result

        def body_from_transform(parent, name, transform):
            return ET.SubElement(parent, "body", name=name,
                                 gravcomp="1",
                                 pos=values([transform[0][3], transform[1][3], transform[2][3]]),
                                 quat=values(matrix_quat(transform)))

        ordinary_from_v100 = [list(row) for row in ORDINARY_LINK6_FROM_V100_LINK6]

        asset = root.find("asset")
        if asset is None:
            asset = ET.Element("asset")
            root.insert(0, asset)
        d435_obj = self._dae_to_obj(WRIST_D435_MESH)
        stand_obj = self._dae_to_obj(WRIST_STAND_MESH)
        ET.SubElement(asset, "mesh", name="d435i_housing", file=str(d435_obj))
        ET.SubElement(asset, "mesh", name="d435i_printed_stand", file=str(stand_obj),
                      scale="0.001 0.001 0.001")
        ET.SubElement(asset, "material", name="d435i_silver_aluminum",
                      rgba="0.70 0.72 0.75 1", specular="0.8", shininess="0.5")
        ET.SubElement(asset, "material", name="d435i_black_print",
                      rgba="0.025 0.025 0.025 1", specular="0.1", shininess="0.1")

        # The printed stand is authored directly in the ordinary model's
        # link6 frame.  Applying the V100-to-ordinary conversion here shifts
        # the bracket away from the camera, even though the camera itself is
        # already positioned correctly.
        stand_transform = transform_xyz_rpy([-0.032, -0.002, 0.025], [0.0, 3.14, 1.57])
        stand = body_from_transform(link6, "camera_stand_link", stand_transform)
        ET.SubElement(stand, "geom", type="mesh", mesh="d435i_printed_stand",
                      material="d435i_black_print", contype="0", conaffinity="0")

        mount_v100 = transform_xyz_rpy(PIPER_MOUNT_V100_POSITION, PIPER_MOUNT_V100_RPY)
        mount = body_from_transform(link6, "d435i_link", mat_mul(ordinary_from_v100, mount_v100))
        camera_link = ET.SubElement(mount, "body", name="d435i_camera_link",
                                    pos=values(PIPER_CAMERA_LINK_OFFSET))
        # MuJoCo requires every dynamic body to have a positive mass.  The
        # camera is fixed and non-colliding, so use a negligible inertial proxy
        # instead of letting the visual mesh change the Piper dynamics.
        ET.SubElement(camera_link, "inertial", pos="0 0 0", mass="1e-6",
                      diaginertia="1e-9 1e-9 1e-9")
        ET.SubElement(camera_link, "geom", type="mesh", mesh="d435i_housing",
                      material="d435i_silver_aluminum",
                      pos="0.0043 -0.0175 0", quat=values(quat([math.pi / 2, 0, math.pi / 2])),
                      contype="0", conaffinity="0", density="0")
        def fixed_frame(parent, name, pos="0 0 0", rpy=None):
            return ET.SubElement(parent, "body", name=name, pos=pos,
                                 quat=values(quat(rpy or [0.0, 0.0, 0.0])))

        depth = fixed_frame(camera_link, "d435i_depth_frame")
        depth_optical = fixed_frame(depth, "d435i_depth_optical_frame", rpy=[-math.pi / 2, 0.0, -math.pi / 2])
        color = fixed_frame(camera_link, "d435i_color_frame", pos=values(PIPER_COLOR_FRAME_OFFSET))
        color_optical = fixed_frame(color, "d435i_color_optical_frame", rpy=PIPER_OPTICAL_RPY)
        infra1 = fixed_frame(camera_link, "d435i_infra1_frame")
        fixed_frame(infra1, "d435i_infra1_optical_frame", rpy=[-math.pi / 2, 0.0, -math.pi / 2])
        infra2 = fixed_frame(camera_link, "d435i_infra2_frame", pos="0 -0.05 0")
        fixed_frame(infra2, "d435i_infra2_optical_frame", rpy=[-math.pi / 2, 0.0, -math.pi / 2])
        accel = fixed_frame(camera_link, "d435i_accel_frame", pos="-0.01174 -0.00552 0.0051")
        fixed_frame(accel, "d435i_accel_optical_frame", rpy=[-math.pi / 2, 0.0, -math.pi / 2])
        gyro = fixed_frame(camera_link, "d435i_gyro_frame", pos="-0.01174 -0.00552 0.0051")
        fixed_frame(gyro, "d435i_gyro_optical_frame", rpy=[-math.pi / 2, 0.0, -math.pi / 2])
        # MuJoCo cameras look along local -Z with local +Y as up.  ROS optical
        # frames look along +Z with +Y down, so a pi rotation about X converts
        # the optical frame convention without changing the camera origin.
        optical_to_mujoco = values(quat([math.pi, 0.0, 0.0]))
        ET.SubElement(color_optical, "camera", name="d435i_color_optical_camera",
                      pos="0 0 0", quat=optical_to_mujoco, fovy="60")
        # Render depth from the color optical pose so the returned depth is
        # color-aligned, matching the RealSense ``align(color)`` stream.
        ET.SubElement(color_optical, "camera", name="d435i_depth_optical_camera",
                      pos="0 0 0", quat=optical_to_mujoco, fovy="60")


    def _dae_to_obj(self, source: Path) -> Path:
        """Convert an upstream COLLADA mesh to a temporary OBJ for MuJoCo.

        The geometry remains entirely sourced from the pinned DAE; this is
        only a format bridge because MuJoCo does not load COLLADA directly.
        """
        namespace = "{http://www.collada.org/2005/11/COLLADASchema}"
        root = ET.parse(source).getroot()
        output = tempfile.NamedTemporaryFile(prefix=f"{source.stem}_", suffix=".obj", delete=False)
        output_path = Path(output.name)
        vertex_offset = 0
        with output:
            output.write(b"# Converted at runtime from pinned upstream COLLADA asset\n")
            for geometry in root.findall(f".//{namespace}geometry"):
                mesh = geometry.find(f"{namespace}mesh")
                if mesh is None:
                    continue
                sources = {}
                for source_element in mesh.findall(f"{namespace}source"):
                    array = source_element.find(f"{namespace}float_array")
                    if array is not None and array.text:
                        sources[f"#{source_element.get('id')}"] = [float(value) for value in array.text.split()]
                vertices = mesh.find(f"{namespace}vertices")
                position_input = None if vertices is None else next(
                    (item for item in vertices.findall(f"{namespace}input") if item.get("semantic") == "POSITION"), None
                )
                if position_input is None or position_input.get("source") not in sources:
                    continue
                position_values = sources[position_input.get("source")]
                vertex_count = len(position_values) // 3
                for index in range(vertex_count):
                    x, y, z = position_values[index * 3:index * 3 + 3]
                    output.write(f"v {x:.9g} {y:.9g} {z:.9g}\n".encode())
                for primitive_name in ("triangles", "polylist"):
                    for primitive in mesh.findall(f"{namespace}{primitive_name}"):
                        inputs = primitive.findall(f"{namespace}input")
                        vertex_input = next((item for item in inputs if item.get("semantic") == "VERTEX"), None)
                        if vertex_input is None or not primitive.find(f"{namespace}p").text:
                            continue
                        stride = max(int(item.get("offset", "0")) for item in inputs) + 1
                        values = [int(value) for value in primitive.find(f"{namespace}p").text.split()]
                        if primitive_name == "triangles":
                            counts = [3] * (len(values) // stride // 3)
                        else:
                            vcount = primitive.find(f"{namespace}vcount")
                            counts = [int(value) for value in vcount.text.split()] if vcount is not None and vcount.text else []
                        cursor = 0
                        for count in counts:
                            polygon = []
                            for _ in range(count):
                                polygon.append(values[cursor * stride + int(vertex_input.get("offset", "0"))] + 1 + vertex_offset)
                                cursor += 1
                            for index in range(1, len(polygon) - 1):
                                output.write(f"f {polygon[0]} {polygon[index]} {polygon[index + 1]}\n".encode())
                vertex_offset += vertex_count
        self._temporary_files.append(output_path)
        return output_path

    def run_gui(self, duration: float = 0.0, lock=None) -> None:
        """Run a native MuJoCo viewer until closed or duration expires.

        When a shared-scene ``lock`` is supplied, stepping and viewer sync are
        serialized against concurrent SceneServer commands.
        """
        self._require()
        try:
            import mujoco.viewer
        except ImportError as exc:
            raise BackendUnavailableError("MuJoCo viewer is unavailable in this installation") from exc
        started = time.monotonic()
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            previous = time.monotonic()
            remainder = 0.0
            while viewer.is_running():
                frame_start = time.monotonic()
                if duration > 0 and frame_start - started >= duration:
                    break
                # Run physics at its timestep, independently of viewer frame
                # rate. Bound catch-up after UI pauses to keep commands responsive.
                remainder += min(frame_start - previous, 0.1)
                previous = frame_start
                steps = int(remainder / self.model.opt.timestep)
                remainder -= steps * self.model.opt.timestep
                with lock if lock is not None else nullcontext():
                    self._step(steps, pace=False)
                    viewer.sync()
                delay = 1 / 60 - (time.monotonic() - frame_start)
                if delay > 0:
                    time.sleep(delay)

    def _require(self):
        if not self._connected or self.model is None or self.data is None:
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
            elapsed = time.monotonic() - started
            delay = max(0.0, self.model.opt.timestep * steps - elapsed)
            if delay:
                time.sleep(delay)

    def step(self, steps: int = 1) -> None:
        """Advance physical simulation; movement commands only change controls."""
        self._step(steps)

    def wait_until_idle(self, timeout: float = 10.0) -> None:
        """Advance a standalone simulation until settled, or raise on timeout.

        The budget is simulation time, so this also works faster than real time.
        Shared-scene clients wait while the GUI owns physics stepping instead.
        """
        self._require()
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        steps = int(math.ceil(timeout / self.model.opt.timestep))
        for _ in range(steps):
            if not self.state().moving:
                return
            self._step(1)
        if self.state().moving:
            raise TimeoutError("MuJoCo motion did not settle before the timeout")

    def state(self) -> RobotState:
        self._require()
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        body = self.model.body("link6").id
        quat = self.data.xquat[body]
        pose = Pose(tuple(self.data.xpos[body]), tuple(quat))
        fingers = self.data.qpos[self._finger_qpos]
        gripper_width = float(fingers[0] - fingers[1])
        # Copy: MuJoCo array slices alias the live simulation data, so a
        # returned JointState must be an independent snapshot.
        joints = JointState(self.data.qpos[self._arm_qpos].copy(), self.data.qvel[self._arm_dofs].copy(), gripper_width)
        arm_error = self.data.ctrl[self._arm_actuators] - joints.positions
        finger_error = self.data.ctrl[self._finger_actuators] - fingers
        moving = bool(self._motion is not None or np.max(np.abs(arm_error)) > 1e-3
                      or np.max(np.abs(joints.velocities)) > 1e-2
                      or np.max(np.abs(finger_error)) > 2e-4
                      or np.max(np.abs(self.data.qvel[self._finger_dofs])) > 2e-3)
        return RobotState(True, moving, joints, pose)

    def move_joints(self, joints, duration_s: float | None = None) -> None:
        self._require()
        q = np.asarray(joints, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("move_joints requires six finite joint values in radians")
        q = np.clip(q, self.ik.lower, self.ik.upper)
        self._motion = JointTrajectory(self.data.qpos[self._arm_qpos], q, self.data.time,
                                       self.motion_duration_s if duration_s is None else duration_s)
        self.data.ctrl[self._arm_actuators] = self._motion.start

    def move_p(self, pose: Pose, duration_s: float | None = None) -> None:
        self._require()
        result = self.ik.solve(self._transform_ik_pose(pose, inverse=True), seed=self.data.qpos[self._arm_qpos])
        if not result.success:
            raise IKError(f"Pinocchio IK failed: {result.message}; position={result.position_error:.6g}; orientation={result.orientation_error:.6g}")
        self.move_joints(result.joints, duration_s=duration_s)

    def gripper(self, width: float, effort: float | None = None) -> None:
        self._require()
        if not 0.0 <= width <= self._gripper_max_width:
            raise ValueError(f"gripper width must be between 0 and {self._gripper_max_width:g} metres")
        targets = [width / 2.0, -width / 2.0]
        self.data.ctrl[self._finger_actuators] = targets

    def stop(self) -> None:
        self._require()
        self._motion = None
        self.data.ctrl[self._arm_actuators] = self.data.qpos[self._arm_qpos]
        self.data.ctrl[self._finger_actuators] = self.data.qpos[self._finger_qpos]
