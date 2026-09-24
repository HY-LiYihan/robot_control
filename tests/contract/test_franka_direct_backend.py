import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from robot_control import Robot
from robot_control.api.types import Pose
from robot_control.backends.franka_direct import FrankaDirectBackend
from robot_control.cli import app
from robot_control.errors import BackendUnavailableError


IDENTITY = [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.5, 0.0, 0.4, 1.0]


class FakeCommand:
    def __init__(self, values):
        self.values = list(values)
        self.motion_finished = False


class FakeControl:
    def __init__(self, robot):
        self.robot = robot
        self.commands = []

    def readOnce(self):
        state = self.robot.read_once()
        state.q_d = [0.11] * 7
        state.O_T_EE_c = [*IDENTITY]
        state.O_T_EE_c[12] = 0.501
        return state, SimpleNamespace(to_sec=lambda: 0.01)

    def writeOnce(self, command):
        self.commands.append(command)


class FakeRobot:
    def __init__(self, address):
        self.address = address
        self.control = None
        self.stopped = False

    def read_once(self):
        return SimpleNamespace(robot_mode="Idle", current_errors=False, last_motion_errors=False,
                               q=[0.1] * 7, dq=[0.0] * 7, O_T_EE=IDENTITY, O_T_EE_c=IDENTITY)

    def start_joint_position_control(self, mode):
        assert mode == "CartesianImpedance"
        self.control = FakeControl(self)
        return self.control

    def start_cartesian_pose_control(self, mode):
        assert mode == "JointImpedance"
        self.control = FakeControl(self)
        return self.control

    def stop(self):
        self.stopped = True


class FakeGripper:
    def __init__(self, address):
        self.address = address
        self.moves = []

    def read_once(self):
        return SimpleNamespace(width=0.06, max_width=0.08)

    def move(self, width, speed):
        self.moves.append((width, speed))
        return True


@pytest.fixture
def bindings(monkeypatch):
    module = SimpleNamespace(Robot=FakeRobot, Gripper=FakeGripper,
                             JointPositions=FakeCommand, CartesianPose=FakeCommand,
                             ControllerMode=SimpleNamespace(CartesianImpedance="CartesianImpedance",
                                                             JointImpedance="JointImpedance"))
    monkeypatch.setitem(sys.modules, "pylibfranka", module)
    return module


def test_direct_selection_and_state_are_passive(bindings):
    instance = Robot.connect("real", robot="franka_fr3", config={"rt_priority": 0})
    try:
        backend = instance._backend
        assert isinstance(backend, FrankaDirectBackend)
        assert backend._robot.address == "192.168.1.6"
        state = instance.state()
        assert state.pose == Pose((0.5, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))
        assert state.joints.gripper == pytest.approx(0.06)
        assert len(state.joints.positions) == 7
        assert backend._robot.control is None
    finally:
        instance.disconnect()


def test_direct_joint_and_cartesian_first_frame_and_final_command(bindings):
    backend = FrankaDirectBackend(motion_duration_s=0.02, rt_priority=0)
    backend.connect()
    backend.move_joints([0.2] * 7)
    commands = backend._robot.control.commands
    np.testing.assert_allclose(commands[0].values, [0.11] * 7)
    np.testing.assert_allclose(commands[-1].values, [0.2] * 7)
    assert commands[-1].motion_finished
    backend.move_p(Pose((0.52, 0.0, 0.4)))
    poses = backend._robot.control.commands
    assert poses[0].values[12] == pytest.approx(0.501)
    assert poses[-1].values[12:15] == pytest.approx([0.52, 0.0, 0.4])
    assert poses[-1].motion_finished


def test_direct_gripper_stop_and_orientation_guard(bindings):
    backend = FrankaDirectBackend(rt_priority=0)
    backend.connect()
    with pytest.raises(ValueError, match="10 degrees"):
        backend.move_p(Pose((0.5, 0.0, 0.4),
                            (math.cos(math.radians(6)), 0, 0, math.sin(math.radians(6)))))
    assert backend._robot.control is None
    backend.gripper(0.04)
    assert backend._gripper.moves == [(0.04, 0.05)]
    backend.stop()
    assert backend._robot.stopped


def test_franka_real_cli_needs_confirmation(bindings):
    runner = CliRunner()
    common = ["--backend", "real", "--robot", "franka_fr3"]
    assert runner.invoke(app, common + ["state"]).exit_code == 0
    denied = runner.invoke(app, common + ["move-joints", "--j1", "0", "--j2", "0",
                                           "--j3", "0", "--j4", "0", "--j5", "0",
                                           "--j6", "0", "--j7", "0"], input="no\n")
    assert denied.exit_code == 0, denied.output
    assert "Cancelled; no command sent." in denied.output


def test_franka_twin_remains_unavailable(bindings):
    with pytest.raises(BackendUnavailableError, match="twin"):
        Robot.connect("twin", robot="franka_fr3")
