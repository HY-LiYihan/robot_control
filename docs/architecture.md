# Architecture

`Robot` exposes one typed facade. A backend implements robot lifecycle, state, joint/cartesian motion, gripper and stop.

The real backend delegates Cartesian `move_p` to the pinned Piper SDK/firmware.
The MuJoCo backend uses Pinocchio on the official six-axis Piper URDF. The IK
solver uses an SE(3) logarithmic residual, its matching local frame Jacobian,
damped least squares, bounded steps, backtracking and deterministic restarts.
Position and orientation convergence are tested separately (metres/radians),
including large orientation errors. Joint limits are enforced during solving.
Pinocchio is installed as the `pin` Python package; no new Git repository is needed.

IK only returns joint targets. Commands write `data.ctrl`, while `mj_step` alone
advances the live configuration and velocity. IK never holds or mutates the live
MuJoCo data. State reports measured positions, velocities and tracking status.
The GUI owns stepping for the shared scene; standalone clients can step or wait
explicitly. CLI commands wait for motion to settle before reporting a result.
The GUI batches physical timesteps against elapsed wall time and refreshes the
viewer at about 60 Hz; pacing waits happen outside the shared-scene lock.
Custom MJCF overrides require a matching `ik_urdf`: connect checks the two FK
models on scratch data and rejects a mismatch instead of executing incorrect IK.

`backends/scene_builder.py` validates a native MJCF scene before loading the Piper
assets. Its main XML must explicitly declare a single empty, fixed `piper_mount`
body directly under worldbody, with finite position and an optional quaternion.
The complete robot and wrist camera are attached using MuJoCo's `MjSpec` API
(3.2.7+), preserving environment includes and relative resources while isolating
the robot's defaults. The scene controls environment geometry, gravity and lights;
the adapter retains the robot's timestep, integrator and explicit inertias.
The bundled scene supplies the familiar blue gradient sky and checker floor.
Robot control addresses are still resolved by name when objects add DOFs or
actuators. Cartesian commands and state remain in world coordinates; the fixed
mount transform maps between world and Pinocchio coordinates for IK/FK checks.
`--scene` is propagated through the macOS viewer launcher and headless run path.
The shared server exposes its scene path so explicit Python scene requests cannot
silently attach to a different environment. Switching scenes restarts the host.

`backends/piper_model.py` builds MJCF at startup from the pinned
`agx_arm_urdf/piper/urdf/piper_with_gripper_description.xacro` and its base URDF.
It expands this concrete include-only Xacro without ROS. Transforms, joint limits,
inertial tensors and collision STL meshes come from that repository.
`backends/collada_visual.py` converts the official visual DAEs into inline MJCF
meshes, preserving scene transforms, units, normals and Lambert diffuse colors.
Equal colors are grouped per visual; conversion is cached in memory by source
path, modification time and size. The adapter supports the pinned triangle/matrix
COLLADA assets, not arbitrary COLLADA or image textures, and adds no dependencies.
Display geoms use group 1 with zero mass and contact masks; unchanged collision
STLs use group 3, hidden by default in both the viewer and RGB-D renderer.
Fixed bodies including
`flange_link` and `gripper_base` are retained, and the end-effector remains `link6`.
The massless virtual `gripper` driver is eliminated; its +/-0.5 mimic relationship
becomes a MuJoCo equality between the two physical finger joints. This keeps eight
simulation coordinates and eight position actuators, with a 0.10 m total opening.
Joint/actuator addresses in the backend are resolved by name.

Simulation-specific position gains, damping, effort limits and the implicitfast
integrator are configured in the adapter. Effort/control limits follow the URDF;
gains and damping are local tuning, not manufacturer controller parameters.
Body gravity compensation is routed through the joint actuators using
`actuatorgravcomp`, so actuator force limits still apply. Joint damping provides
velocity feedback; position targets evolve through physical integration, with
inertia, contacts and force limits retained. No collision-aware motion planning
or trajectory generator is added; IK supplies a target, not a collision-free path.
Only the base/link1 adjacent mounting pair is explicitly excluded from contact:
their upstream meshes overlap at the joint, and MuJoCo's default parent-child
filter does not cover the world-welded base. Other collision pairs remain enabled.

The old `piper_isaac_sim` repository is read only for `d435.dae` and
`realsense_mid_stand.dae` on the default path. Camera attachment transforms stay
in `backends/mujoco.py`; they retain the previous nominal mounting calibration.
No old arm XML/URDF or arm/gripper mesh is used. Explicit legacy MJCF overrides
remain supported via `model_path`.

Both camera providers return `RGBDFrame`: RGB is `uint8 HxWx3`, depth is `float32 HxW`, and metadata contains resolution, intrinsics, timestamp, frame id and depth scale. RealSense uses `pyrealsense2`; simulation uses the official D435i color-optical pose for both color and color-aligned depth rendering.

The public unit conventions are metres, radians and `(w, x, y, z)` quaternions. ROS 2 is intentionally not a first-stage dependency.

`RobotState.pose` is the measured end-effector pose for both MuJoCo and the
Piper real backend. MuJoCo reads `link6` directly; the real backend computes
the same `link6` FK from joint feedback and the pinned Piper URDF. RGB-D
frames may carry `T_base_color_optical` extrinsics: the fixed wrist-camera
transform is shared by the Piper model and the real-camera path, while FR3
uses `fr3_link0` as its reference frame. The CLI captures and reports these
extrinsics with each `camera` frame; `--no-extrinsics` keeps camera-only
capture available.

The upstream Isaac asset repository contains DAE paths that differ only by letter case. On case-insensitive macOS filesystems that submodule can appear dirty immediately after checkout, so `.gitmodules` ignores submodule worktree dirt while the pinned gitlink SHA remains authoritative.
# FR3 selection

`Robot.connect` selects the MuJoCo implementation with a `robot` argument (default `piper`). `fr3/mujoco.py` uses the supplied FR3 MJCF and `fr3/ik.py` the supplied URDF; the existing Piper adapters remain unchanged. `JointState` accepts six or seven positions with matching velocity lengths. `robot_control` is the primary CLI; invoking it with `--backend mujoco` and no subcommand hosts a GUI (or a headless, physically stepped scene with `--no-gui`). Subcommands probe both distinct scene sockets to infer the unique active robot; if both are active, a missing `--robot` is an error. Explicit `--backend real|twin --robot piper|franka_fr3` prevents implicit real-arm selection. The shared scene protocol advertises `robot` and rejects cross-robot connections. The server selects `d435i_check` when rendering FR3 RGB-D. FR3 real control uses `backends/franka_direct.py` and local `pylibfranka`/`libfranka`; FR3 twin has a separate local socket and mirrors measured joints/gripper without stepping MuJoCo while routing control to the direct FCI connection.
