# Hardware setup

Install `robot-control[real,camera]`, configure the CAN interface using the scripts in `vendor/piper_sdk`, and connect a D435i. The default camera stream is aligned 1280x720 RGB/depth at 30 FPS.

Use `robot_control doctor` before connecting. The real backend defaults to `can0`; override with `--can-name` after the subcommand. Always specify `--backend real --robot piper` before real-arm commands, for example `robot_control --backend real --robot piper state`. Motion is available after connection, so test with the arm clear and keep the physical emergency stop accessible. FR3 real-arm control is unavailable.

For local simulation, use `robot_control --backend mujoco` to open the GUI, or add `--robot franka_fr3` for FR3. The wrist RGB-D command is `robot_control camera --rgb-out wrist_rgb.png --depth-out wrist_depth.npy`; with one scene running it targets that scene automatically and prints `T_base_color_optical` extrinsics. On real Piper hardware, use `robot_control --backend real --robot piper camera` to capture RGB-D and compute the same extrinsics from joint feedback and the shared URDF. Standalone RealSense capture without arm extrinsics is `robot_control --backend real camera --no-extrinsics`.
