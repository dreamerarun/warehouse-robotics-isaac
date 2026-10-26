#!/usr/bin/env python3
# ============================================================================
# teleop.launch.py -- starts keyboard teleoperation for the LeKiwi robot
# ----------------------------------------------------------------------------
# Requires `control.launch.py` to be running (it provides the /cmd_vel consumer
# and the arm trajectory controller).
#
#   ros2 launch lekIwi_ros2_control teleop.launch.py
#
# §22: use_sim_time defaults to true here so the publish timer is synchronised
# with the Isaac Sim /clock.  If the simulation is paused the teleop timer
# stops publishing, the /cmd_vel watchdog in lekIwi_base_controller trips
# after 0.5 s and the base stops -- which is the safe behaviour.
# ============================================================================

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG = "lekIwi_ros2_control"

# entries of config/joint_limits.yaml that lekIwi_teleop declares
TELEOP_PARAMS = [
    "linear_speed",
    "strafe_speed",
    "angular_speed",
    "arm_step",
    "arm_joint_names",
    "arm_position_min",
    "arm_position_max",
    "arm_home_positions",
]


def generate_launch_description():
    pkg_share = get_package_share_directory(PKG)
    joint_limits_yaml = os.path.join(pkg_share, "config", "joint_limits.yaml")

    with open(joint_limits_yaml, "r") as fh:
        limits = yaml.safe_load(fh).get("joint_limits", {})
    teleop_params = {k: limits[k] for k in TELEOP_PARAMS if k in limits}

    use_sim_time = LaunchConfiguration("use_sim_time")
    own_terminal = LaunchConfiguration("use_own_terminal")

    common = dict(
        package=PKG,
        executable="lekIwi_teleop",
        name="lekIwi_teleop",
        output="screen",
        respawn=False,
        parameters=[
            teleop_params,
            {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
        ],
    )

    # Raw terminal input only works on a real tty.  Option A: give the node its
    # own xterm (use_own_terminal:=true).  Option B (default): run it in the
    # terminal you launched from, where the keyboard is attached.
    teleop_own_tty = Node(prefix="xterm -e", condition=IfCondition(own_terminal), **common)
    teleop_here = Node(condition=UnlessCondition(own_terminal), **common)

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use /clock from Isaac Sim (required by §22).",
        ),
        DeclareLaunchArgument(
            "use_own_terminal",
            default_value="false",
            description=(
                "Launch teleop in its own xterm window. Keep false when you "
                "launch it from a terminal you can type into."
            ),
        ),
        teleop_own_tty,
        teleop_here,
    ])
