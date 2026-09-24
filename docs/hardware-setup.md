# Hardware setup

Install `robot-control[real,camera]`, configure the CAN interface using the scripts in `vendor/piper_sdk`, and connect a D435i. The default camera stream is aligned 1280x720 RGB/depth at 30 FPS.

Use `robot_control doctor` before connecting. The Piper real backend defaults to `can0`; override with `--can-name` after the subcommand. Always specify `--backend real` and `--robot piper` or `--robot franka_fr3` before real-arm commands, for example `robot_control --backend real --robot franka_fr3 state`. Motion is available after connection, so test with the arm clear and keep the physical emergency stop accessible. FR3 real-arm control requires compatible local `pylibfranka` and `libfranka` installations.

For local simulation, use `robot_control --backend mujoco` to open the GUI, or add `--robot franka_fr3` for FR3. The wrist RGB-D command is `robot_control camera --rgb-out wrist_rgb.png --depth-out wrist_depth.npy`; with one scene running it targets that scene automatically and prints `T_base_color_optical` extrinsics. On real Piper hardware, use `robot_control --backend real --robot piper camera` to capture RGB-D and compute the same extrinsics from joint feedback and the shared URDF. Standalone RealSense capture without arm extrinsics is `robot_control --backend real camera --no-extrinsics`.

## Ubuntu FR3 deployment

Use an Ubuntu control PC with Python 3.10 or 3.11, network connectivity to the FR3, an enabled FCI, and a graphical desktop if the twin GUI is needed:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev
git clone --recurse-submodules git@github.com:HY-LiYihan/robot_control.git
cd robot_control
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[mujoco,dev]"
python -m pytest -q
robot_control --help
```

The repository does not bundle `pylibfranka` or `libfranka`. Install a pair matching the FR3 robot-system release and the direct backend API; do not blindly replace it with an unrelated latest wheel:

```bash
source ~/robot_control/.venv/bin/activate
python -m pip install /path/to/compatible/pylibfranka-*.whl
sudo cp /path/to/libfranka.so.0.13.5 /usr/local/lib/
sudo ldconfig
python -c 'import pylibfranka; print("pylibfranka import OK")'
python -c 'from robot_control.backends.franka_direct import _bindings; print(_bindings().__name__)'
```

Set the actual robot address and check connectivity:

```bash
export FRANKA_ROBOT_IP=192.168.1.6
ping -c 3 "$FRANKA_ROBOT_IP"
```

In the first terminal, start the twin. This opens the MuJoCo mirror and creates one direct FCI connection:

```bash
cd ~/robot_control
source .venv/bin/activate
export FRANKA_ROBOT_IP=192.168.1.6
robot_control --backend twin --robot franka_fr3
```

Without a graphical desktop:

```bash
robot_control --backend twin --robot franka_fr3 --no-gui
```

In a second terminal on the same Ubuntu PC, use the same virtual environment and IP:

```bash
cd ~/robot_control
source .venv/bin/activate
export FRANKA_ROBOT_IP=192.168.1.6
robot_control --backend real --robot franka_fr3 state
robot_control --backend real --robot franka_fr3 pose
```

When the twin is running, `real` and `twin` clients reuse its local `/tmp/fr3_twin.sock` connection. The GUI only mirrors measured seven-joint and gripper feedback; it does not step MuJoCo or send simulated targets to the robot. Movement commands still move the real FR3 and execute without an additional CLI confirmation prompt; backend state and target validation remain enabled. Use `robot_control --backend real --robot franka_fr3 camera` to capture the real RGB-D stream and output nominal `fr3_link0`-to-color-optical extrinsics from measured joints. The URDF and simulation camera mount model the D435i upside down (180° about its viewing axis); check the physical mount and calibrate for precision. Use `robot_control --backend real camera --no-extrinsics` for standalone RealSense capture. Real hardware motion has not been validated in this repository; inspect the workspace and verify the physical emergency stop before any movement.
