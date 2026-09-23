# API contract

`Pose.position` uses metres and `Pose.quaternion` uses `(w, x, y, z)`. `Robot.move_joints` takes six Piper or seven FR3 radians. `gripper` takes opening width in metres. `state()` returns the selected arm's joints, gripper state, optional end-effector pose, connection status and timestamps.

In MuJoCo, both `state().pose` and `move_p(Pose(...))` use the scene **world**
frame. The backend transforms targets into the independent Pinocchio model frame
using the fixed `piper_mount` pose, including its rotation. Real-robot coordinates
retain the SDK convention.

`robot_control --backend mujoco --scene scenes/tabletop.xml` opens a GUI and loads a native
MJCF environment. Omit `--scene` to use the bundled blue sky/checker ground scene;
`--no-gui` starts a persistent headless scene; `--scene` also works with legacy `run --steps N`. It is rejected for `real`.
The main XML must contain exactly one empty `worldbody/body` named `piper_mount`
with an explicit finite three-value `pos` and optional nonzero `quat` (wxyz).
It cannot be nested, moving, or defined in an include. Other environment content
supports native includes and relative assets. Validation precedes robot loading.

Python callers can use `Robot.connect("mujoco", {"scene": "scenes/tabletop.xml"})`
or `MujocoBackend(scene="scenes/tabletop.xml")`. If a shared server is already
running, an explicit scene request must match its resolved scene path, otherwise
connection raises an error. Omitting `scene` connects to whichever shared scene
is running, or uses the default scene when running standalone. Scene changes
require restarting the host. Explicit legacy `model_path` XML overrides remain
complete models and cannot be combined with `scene`.

MuJoCo motion commands are nonblocking setpoint commands. `move_p` solves IK
with Pinocchio using the current measured arm configuration as its seed, then
writes the resulting joint position targets to MuJoCo position-actuator controls.
`move_joints` and `gripper` also update controls only. None of these commands
writes the live `qpos` or `qvel`. Pinocchio operates on an independent model;
failed IK leaves both the existing controls and measured state unchanged.

The GUI advances physics continuously. Standalone scripts call `robot.step(n)`
to advance physics explicitly, or `robot.wait_until_idle(timeout=10)` to step
until measured position errors and velocities settle. The standalone timeout
is in simulation seconds; shared-scene waiting polls the GUI using wall time.
CLI motion commands wait for settling before printing measured results.
Timeout raises `TimeoutError`; it does not force completion or cancel the target.
`state().moving` reflects both tracking error and measured velocity. `stop()`
sets holding targets to the current measured positions; it does not erase
velocity, so physical deceleration is still required.

The default simulation uses the official `agx_arm_urdf` Piper model and reports
the `link6` pose. Its gripper accepts 0–0.10 m total opening; the two physical
fingers move by +/- half that width. The real backend retains its existing
0–0.07 m command limit and SDK Cartesian control. The model migration does not
change hardware limits or replace firmware IK on the real backend.
# Robot selection

The default `Robot.connect("mujoco")` remains the six-joint Piper. Use `Robot.connect("mujoco", robot="franka_fr3")` for the seven-joint FR3. The unified CLI uses `robot_control [--backend mujoco|real|twin] [--robot piper|franka_fr3] COMMAND`. Without a command, the MuJoCo backend starts a GUI; `--no-gui` hosts a headless scene. MuJoCo commands without `--robot` select the single running scene; if both are running, explicitly specify one. Only FR3 requires `--j7` for `move-joints`. `pepper` is accepted as an alias for Piper. Real-arm commands require explicit `--backend real --robot piper` (or `--backend twin --robot piper` with the twin running); FR3 real-arm control is unavailable. `robot_control --backend real camera --no-extrinsics` uses RealSense independently of the arm. See `docs/fr3.md`.

# Camera access

`Robot.camera(width=1280, height=720)` returns an `RGBDFrame` for either a standalone or shared MuJoCo scene, or for the Piper `real` / `twin` backend. The frame contains RGB, depth, intrinsics, and camera-to-base extrinsics; real-arm extrinsics require synchronized arm feedback. Connect with `Robot.connect(...)`, capture, then call `robot.disconnect()`. The twin's camera is the real RealSense, not the simulation renderer.

For a RealSense capture without connecting the arm, use `robot_control --backend real camera --no-extrinsics` or:

```python
from robot_control.sensors.service import CameraService

frame = CameraService("real", include_extrinsics=False).capture()
```

The camera-only frame has no extrinsics. CLI real-arm commands and real camera captures **with** extrinsics still require explicit `--backend real --robot piper`; FR3 real-arm control is not available.
