# Robot Control — Piper + Franka FR3

统一的 Piper / Franka FR3 Python API、MuJoCo 仿真、Piper 真机和 D435i RGB-D 接口。主命令为 `robot_control`；旧命令 `piper` 保留兼容。启动时不传 `--robot` 默认 Piper；`--robot franka_fr3` 启动七轴 FR3。`pepper` 也作为 Piper 的别名接受。

Python 包和安装包分别为 `robot_control` 与 `robot-control`。旧版 `piper_control` 导入路径和 `PiperRobot` 类名仍可使用；升级后请重新运行下方安装命令。

## 上游版本

| 组件 | URL | 固定版本 |
| --- | --- | --- |
| Piper SDK | https://github.com/agilexrobotics/piper_sdk | `0.6.2`, `c9e8a28174e71eeaac448593cb65f8ab258a92fe` |
| Piper 官方 URDF / 夹爪 / meshes | https://github.com/agilexrobotics/agx_arm_urdf | `f6642ce0d7872c686f29c99e9e10cd23d1d49313` |
| D435 外壳和支架资源 | https://github.com/agilexrobotics/piper_isaac_sim | `8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2` |
| FR3 + D435i 腕部支架模型 | 用户提供的 `fr3_d435i_wrist_camera_mount_release(1)` | 收录于 `vendor/fr3_d435i/` |

Piper 模型仓库作为 submodule 保留。机械臂与夹爪使用 `agx_arm_urdf/piper`；旧 Isaac 仓库仅提供 D435 外壳和打印支架，不要求 Isaac Sim 或 ROS。FR3 模型为随仓库提交的快照，无需额外 submodule；来源、许可和仿真修改见 `docs/fr3.md`。

## 安装

```bash
git clone --recurse-submodules git@github.com:HY-LiYihan/robot_control.git
cd robot_control
git submodule update --init --recursive
python -m pip install -e ".[mujoco,dev]"
pip install -e ".[real,camera]"  # 需要真机/RealSense 时
```

FR3 的 `vendor` 模型目录按源码路径读取，使用上述**可编辑安装**。FR3 真机通过本机匹配的 `pylibfranka`/`libfranka` 直连 FCI；这些依赖不包含在仓库中。Piper 和 FR3 twin 同时需要 `[mujoco,real]`，采集真相机还需 `[camera]`。真相机取帧可独立使用 `robot_control --backend real camera --no-extrinsics`。

## 统一命令与自动选择

```text
robot_control [--backend mujoco|real|twin] [--robot piper|franka_fr3] [--scene 场景.xml] [--no-gui] [子命令] [子命令参数]
```

**启动场景**：在第一个终端执行下列命令并保持其运行。只选 `mujoco`（也是默认后端）时自动打开 GUI；`--no-gui` 则启动无窗口、持续推进物理仿真的共享场景，按 Ctrl+C 退出。无 `--robot` 时启动 Piper。

```bash
robot_control --backend mujoco
robot_control --backend mujoco --robot franka_fr3
robot_control --backend mujoco --robot franka_fr3 --scene scenes/fr3_tabletop.xml
robot_control --backend mujoco --robot franka_fr3 --no-gui
```

**控制场景**：在第二个终端执行，子命令不会再开 GUI。假设第一个终端已启动 FR3，下面省略 `--robot` 的命令会自动控制 **FR3**：

```bash
robot_control state
robot_control pose
robot_control move-joints --j1 0 --j2 -0.7854 --j3 0 --j4 -2.3562 --j5 0 --j6 1.5708 --j7 0.7854
robot_control move-p --x 0.55 --y 0 --z 0.73
robot_control gripper 0.04
robot_control camera --rgb-out wrist_rgb.png --depth-out wrist_depth.npy
robot_control stop
```

控制命令的 `--robot` 选择顺序为：显式指定 > 当前唯一运行的 MuJoCo 场景 > 无场景时默认 Piper（独立仿真）。Piper 和 FR3 场景同时运行时，必须显式指定 `--robot`，例如 `robot_control --robot franka_fr3 state`。两种机器人的 GUI 使用不同 socket，客户端不会误连另一种机器人。FR3 `move-joints` 必须传 `--j7`，Piper 不可传。

仿真机械臂关节和末端运动使用平滑轨迹：Piper 默认 2 秒，FR3 默认 4 秒；用子命令 `--duration 秒数` 覆盖，例如 `robot_control --backend mujoco --robot piper move-joints --j1 0 --j2 0.5 --j3 -0.5 --j4 0 --j5 0 --j6 0 --duration 3`。时间是仿真时间中的目标轨迹时长，实际关节稳定可能略晚；`wait_until_idle` 的超时不是运动时长。Python API 可用 `robot.move_joints(joints, duration_s=3)`、`robot.move_p(pose, duration_s=3)`，也可在 `Robot.connect("mujoco", {"motion_duration_s": 3}, robot="piper")` 中设置当前连接的默认时长；共享 GUI 同样支持。

**真机**默认不使用 MuJoCo GUI。为了避免误操作，真机运动或状态命令必须显式写 `--backend real` 和机器人型号：

```bash
robot_control --backend real --robot piper state
robot_control --backend real --robot piper gripper 0.02
robot_control --backend real --robot franka_fr3 state
robot_control --backend real camera --no-extrinsics   # 仅采集本机 D435i，无需连接机械臂
```

FR3 真机使用 `FRANKA_ROBOT_IP`（默认 `192.168.1.6`）直连；关节和末端运动默认用 4 秒，可通过 `FRANKA_MOVE_DURATION_S` 或每条命令的 `--duration` 覆盖（twin 运行时也支持逐条设置）。Piper 真机仍由 SDK 控制运动速度，**不支持 `--duration`**。CLI 的 FR3 运动命令会立即发送，不再逐次要求输入确认。执行前须确认目标和现场安全；底层仍会检查机器人状态和目标参数。Python API 使用 `Robot.connect("mujoco", robot="franka_fr3")` 或 `Robot.connect("real", robot="franka_fr3")`。FR3 场景必须包含 `fr3_mount`，Piper 场景仍使用 `piper_mount`，参见 `docs/fr3.md`。

## Piper 真机 twin 镜像

只需两个终端，不用额外开真机控制服务。确保真机 CAN 已连接并有新鲜的关节反馈，再安装 `python -m pip install -e ".[mujoco,real]"`。第一个终端启动 GUI 并保持运行（默认 `can0`）：

```bash
robot_control --backend twin --robot piper
# 指定 CAN 接口或自定义 Piper 场景：
robot_control --backend twin --robot piper run --gui --can-name can1 --scene scenes/tabletop.xml
```

第二个终端直接使用真机命令；运动命令仍需显式指定 `--backend real --robot piper`，避免误操作：

```bash
robot_control --backend real --robot piper state
robot_control --backend real --robot piper pose
# 或显式通过 twin 进程的真机连接执行相同命令：
robot_control --backend twin --robot piper state
```

只要 twin GUI 在运行，同机 `--backend real` 会自动通过仅本机可访问的 twin socket 复用它的真机连接；`--backend twin` 的运动和状态命令必须有运行中的 twin，绝不回退为仿真控制。如果使用 `can1`，第二终端的命令也要加 `--can-name can1`（放在子命令之后）。Python API 的 `Robot.connect("twin")` 同样复用该真机连接。窗口只同步真机测得的六轴关节角与可用的夹爪开口：不执行 MuJoCo 物理步进，不将窗口中的操作或仿真目标发送到真机。`camera` 仍读取**真机** RealSense，相机命令还需安装 `[camera]`。**注意：使用 `move-joints`、`gripper` 等控制命令会真实驱动机械臂；先检查现场安全。** 关闭窗口会断开 twin 的真机连接。Piper twin 尚未经过真实硬件验证。

## FR3 真机 twin 镜像

第一个终端启动 FR3 twin；它连接一次真机，并用 MuJoCo GUI 镜像七轴和夹爪反馈：

```bash
export FRANKA_ROBOT_IP=192.168.1.6
robot_control --backend twin --robot franka_fr3
```

无图形桌面时可以使用：

```bash
robot_control --backend twin --robot franka_fr3 --no-gui
```

第二个终端使用同一个虚拟环境执行命令；控制仍然发送给 FR3 真机：

```bash
export FRANKA_ROBOT_IP=192.168.1.6
robot_control --backend real --robot franka_fr3 state
robot_control --backend real --robot franka_fr3 pose
```

FR3 twin 默认使用本机 `/tmp/fr3_twin.sock`，也可以通过 `FR3_TWIN_SOCKET` 修改。启动 twin 不会主动移动机械臂；MuJoCo 只显示真机状态，不推进仿真物理。`robot_control --backend real --robot franka_fr3 camera` 会采集真机 D435i RGB-D，并根据七轴反馈及模型自动输出相对于 `fr3_link0` 的外参；独立采集、不连接机械臂时使用 `robot_control --backend real camera --no-extrinsics`。详细 Ubuntu 部署步骤见 `docs/hardware-setup.md`。

## 使用

```python
from robot_control import Robot

robot = Robot.connect("mujoco")
robot.move_joints([0.2, 0.8, -1.2, 0.2, -0.3, 0.4])
robot.wait_until_idle()
target = robot.state().pose
robot.move_joints([0.22, 0.82, -1.22, 0.22, -0.32, 0.42])
robot.wait_until_idle()
robot.move_p(target)
robot.wait_until_idle()
print(robot.state())
frame = robot.camera(width=640, height=480)  # RGB-D + 相机相对机器人基座的外参
robot.stop()
robot.disconnect()
```

`Robot.camera()` 在独立仿真或共享 MuJoCo 场景中读取仿真相机；在 Piper 或 FR3 的 `real` / `twin` 中读取真机 RealSense，并在有同步关节反馈时提供基座外参。只采真相机、完全不连接机械臂时，可用 `robot_control --backend real camera --no-extrinsics`，或在 Python 中调用 `CameraService("real", include_extrinsics=False).capture()`（从 `robot_control.sensors.service` 导入）。真机控制及包含外参的真机相机命令仍须显式指定 `--backend real --robot piper` 或 `--backend real --robot franka_fr3`。

## Piper 场景与坐标细节

默认 Piper 仿真启动时夹爪开口为 0.07 m（两侧各 0.035 m），对应当前真机接口的 0.07 m 上限；0.7 m 超出机械行程，仿真模型自身允许的最大开口为 0.10 m。FR3 仿真启动时夹爪开口为模型最大值 0.08 m。twin 模式只显示真机反馈，不会为初始姿态主动控制真机夹爪。

打开 MuJoCo GUI（窗口关闭前持续运行）：

```bash
robot_control --backend mujoco
```

不传 `--scene` 时使用 MuJoCo 示例常见的蓝色渐变天空和棋盘地面。指定其他场景：

```bash
robot_control --backend mujoco --scene scenes/tabletop.xml
# 无窗口、持续运行的共享场景：
robot_control --backend mujoco --scene scenes/tabletop.xml --no-gui
# 原有仅执行固定步数的命令仍可使用：
robot_control --backend mujoco run --steps 100 --scene scenes/tabletop.xml
```

`--scene` 用于启动 MuJoCo 场景或 twin 镜像窗口，纯真机后端会拒绝该参数。场景采用原生 MJCF XML；加载机器人和资源前，会检查主 XML 的 `<worldbody>` 下是否有且只有一个空的固定挂载节点：

```xml
<body name="piper_mount" pos="-0.3 0 0.75" quat="1 0 0 0"/>
```

`pos` 必填，是基座在世界中的位置，单位米；`quat` 是 wxyz 四元数，可省略以使用单位朝向。该节点不能包含关节、子节点或其他属性，不能嵌在其他 body/frame 中，也不能放在 include 文件内。缺少节点、缺少位置、非有限数值或无效四元数都会报错，不会静默放到原点。其余场景内容可使用 MuJoCo 的 include、相对资源路径、静态物体和可运动物体；机器人及其夹爪、支架、相机由程序整体挂载。可复制 `scenes/tabletop.xml` 修改桌子、物体、灯光和安装位姿。切换场景需要关闭原 GUI 后重新启动，后续控制和相机命令连接这个运行中的场景。

MuJoCo 的 `pose` 输出和 `move-p` 输入统一使用**场景世界坐标**；后端自动转换成 Pinocchio 模型坐标后求解 IK，因此移动或旋转基座不需要修改目标求解器。场景碰撞会影响实际运动，但 IK 本身不提供避障路径。环境通过 MuJoCo 3.2.7+ 的模型装配接口加载；机械臂继续采用 0.002 s 步长、implicitfast 积分器和原有控制参数。

读取当前末端位姿（位置 m，四元数顺序 wxyz；MuJoCo 下为世界坐标）：

```bash
robot_control pose
```

移动到指定位置；不提供四元数时保持当前姿态：

```bash
robot_control move-p --x 0.0561352 --y 0.0 --z 0.2131783
```

指定完整四元数：

```bash
robot_control move-p --x 0.0561352 --y 0.0 --z 0.2131783 \
  --qw -0.73727734 --qx 0.0 --qy -0.67559020 --qz 0.0
```

控制夹爪，单位为米：

```bash
robot_control gripper 0.02
```

获取腕部 D435 RGB-D。RGB 为 PNG，depth 为米制 `float32` NumPy 文件，分辨率固定 1280x720：

```bash
robot_control camera \
  --rgb-out wrist_rgb.png --depth-out wrist_depth.npy
```

`camera` also prints the camera extrinsics in JSON. `translation_m` and the
row-major `rotation_row_major` describe `T_base_color_optical`: points in
`d435i_color_optical_frame` are transformed into `base_link` for Piper or
`fr3_link0` for FR3. On Piper real hardware, the transform is computed from
joint feedback and the shared URDF. On FR3 real/twin hardware, the transform
uses measured joints and the FR3 URDF camera mount. Use `--no-extrinsics` for a
standalone RealSense capture without a connected arm.

关节命令使用弧度，并采用选项形式以支持负数：

```bash
robot_control move-joints \
  --j1 0.1 --j2 0.2 --j3 -0.2 --j4 0 --j5 0 --j6 0
```

MuJoCo 启动时读取 `vendor/agx_arm_urdf/piper/urdf/piper_with_gripper_description.xacro` 及其引用的 `piper_description.urdf`，转换为仿真模型。关节坐标、限位、质量和惯量来自这些文件，末端固定为 `link6`。显示使用官方 `visual` 引用的 DAE 网格，保留部件变换、法线和材质颜色；转换器按颜色分组生成内嵌 MJCF 网格，无需额外依赖或手工生成资源。显示网格位于 group 1，关闭碰撞且质量为零；碰撞继续使用原有 STL，位于 group 3 且完全透明，即使开启该显示组也不会将碰撞网格叠在夹爪外观上，物理碰撞不受影响。上游 DAE 没有图片贴图，本次恢复的是原有几何和材质颜色。法兰与夹爪结构来自新 Xacro；仿真夹爪开口范围为 0–0.10 m，两侧手指通过等式约束同步运动。

IK 使用 Pinocchio（安装包名 `pin`），读取同一官方机械臂 URDF，求解末端 `link6` 的关节目标。`move_p`、`move_joints` 和 `gripper` 仅下发执行器控制目标，不改写实际关节位置或速度。GUI 持续推进物理仿真；独立脚本可调用 `robot.step(n)` 或 `robot.wait_until_idle()`。CLI 运动命令会等待实际运动完成后返回；超时会报错，不会强制关节到位。`stop()` 下发当前位置保持目标，通过执行器减速。

位置执行器、阻尼和经执行器限力的重力补偿由本项目配置。D435 外壳及打印支架仍使用旧 Isaac 仓库的两个 DAE 文件，外壳显示为带高光的银色，打印支架为黑色；材质设置位于 `backends/mujoco.py`，属于本地外观配置。默认照明适当调亮以显示深色机械臂细节。相机安装变换和 nominal extrinsics 保留在本项目代码中，未做实机重新标定。真机 `state()` 与仿真统一返回关节反馈和末端 `link6` 位姿；真机末端位姿由同一份 Piper URDF 对反馈关节做 FK 得到。真机 SDK/固件 IK 控制路径与原有 0–0.07 m 夹爪限制暂不改变。已经运行的 GUI/共享场景需重启才能加载新模型和控制逻辑。macOS GUI 使用当前 Python 环境旁的 `mjpython`，请在同一环境安装依赖。

## 开发

```bash
pytest
git submodule update --init --recursive
```
