# Franka FR3 仿真与 twin

FR3 + Franka Hand + CNC 腕部连接件 + D435i 模型来自用户提供的 `fr3_d435i_wrist_camera_mount_release(1)`。`vendor/fr3_d435i/` 保存了原始 MJCF、URDF、网格、预览图、README 和许可文件。模型许可请分别查看 `vendor/fr3_d435i/mujoco_fr3/LICENSE`、`vendor/fr3_d435i/licenses/`；仓库根目录 MIT 许可仅适用于本项目代码。模型快照不是 submodule，克隆后即可从源码加载。

与单独 FR3 仓库一样，模型的无质量相机节点增加了 70 g **仿真**惯量，七个机械臂链节开启 MuJoCo 重力补偿，使位置执行器收敛。这不是官方硬件动力学标定值。机械臂七轴、双指夹爪、相机和支架均来自附件，七轴 IK 使用其 URDF，通过两组位姿对照验证 MJCF / URDF 的运动学一致性；IK 不避障。

FR3 启动时的七轴初始位姿与执行器目标均为 `[0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854]` 弧度，即 `[0°, -45°, 0°, -135°, 0°, 90°, 45°]`；夹爪初始开口为模型最大值 0.08 米（每侧手指 0.04 米）。

使用 `robot_control --backend mujoco --robot franka_fr3` 在默认棋盘场景启动 GUI；或添加 `--scene scenes/fr3_tabletop.xml`。启动后在另一个终端执行 `robot_control state` 等命令时，可省略 `--robot`，程序会选择当前运行的 FR3；两种机器人同时运行时需显式指定。自定义 MJCF 主文件的 `worldbody` 必须直接包含一个空的固定安装节点：

```xml
<body name="fr3_mount" pos="0 0 0" quat="1 0 0 0"/>
```

`pos` 必填，单位米；`quat` 是可选的 wxyz 四元数。FR3 的 `move-p` 位姿对应 `fr3_link7`，仿真相机使用模型中的 `d435i_check`。仿真深度是 MuJoCo 渲染深度，不等同于 D435i 物理测量。若有 USB RealSense，`robot_control --backend real camera --no-extrinsics` 仅采集相机，不控制机械臂。FR3 真机通过本地匹配的 `pylibfranka`/`libfranka` 直连 FCI；`robot_control --backend twin --robot franka_fr3` 会镜像真机七轴和夹爪反馈，第二个终端的 `real`/`twin` 命令复用该连接。启动 twin 不会主动移动真机，FR3 真机相机外参尚未支持。

FR3 仿真和真机的关节、末端轨迹默认均为 4 秒；仿真按仿真时间推进轨迹，实际稳定时间可能略晚。`move-joints` 和 `move-p` 可追加 `--duration 6` 指定单条命令的轨迹时长，在 GUI、无窗口共享场景、独立仿真和 twin 中均适用。夹爪不使用该时长参数；Piper 仿真默认 2 秒，Piper 真机保持原有 SDK 控制方式、不支持 `--duration`。

Python 示例：

```python
from robot_control import Robot

arm = Robot.connect("mujoco", robot="franka_fr3")
try:
    print(arm.state())
    arm.move_joints([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
    arm.wait_until_idle()
    arm.gripper(0.04)
    arm.wait_until_idle()
finally:
    arm.disconnect()
```
