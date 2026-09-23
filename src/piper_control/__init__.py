"""Compatibility entry point for the former package name."""

from importlib import import_module
from pkgutil import walk_packages
import sys

import robot_control
from robot_control import JointState, PiperRobot, Pose, Robot, RobotState

for package in walk_packages(robot_control.__path__, robot_control.__name__ + "."):
    alias = package.name.replace("robot_control", __name__, 1)
    module = import_module(package.name)
    sys.modules[alias] = module
    parent, _, child = alias.rpartition(".")
    setattr(sys.modules[parent], child, module)

__all__ = ["Robot", "PiperRobot", "Pose", "JointState", "RobotState"]
