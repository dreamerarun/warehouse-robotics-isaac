#!/usr/bin/env python3
# ============================================================================
# control.launch.py -- starts the ros2_control stack for the LeKiwi robot
# ----------------------------------------------------------------------------
# Isaac Sim is deliberately NOT launched from here (§21): it must be started
# separately by scripts/import_lekIwi_urdf.py + create_lekIwi_action_graph.py.
# This launch file only brings up the ROS 2 side:
#
#   1. controller_manager (ros2_control_node)
#        robot_description = xacro lekIwi_control.urdf.xacro
#        hardware          = lekiwi_isaac_hardware/LekIwiIsaacSystem
#                            (bridges to Isaac Sim over ROS 2 topics)
#   2. joint_state_broadcaster  -> /joint_states
#   3. arm_controller           -> /arm_controller/follow_joint_trajectory
#   4. base_controller          -> /base_controller/commands
#   5. lekIwi_base_controller   -> /cmd_vel IK + limits + watchdog
#
# The three spawners are serialised with OnProcessExit so controller_manager
# never sees concurrent configure/activate requests.
# ============================================================================

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG = "lekIwi_ros2_control"

# entries of config/joint_limits.yaml that lekIwi_base_controller declares
BASE_CTRL_PARAMS = [
    "wheel_radius",
    "base_radius",
    "max_linear_velocity",
    "max_angular_velocity",
    "max_wheel_velocity",
    "linear_accel_limit",
    "angular_accel_limit",
    "cmd_vel_timeout",
    "wheel_joint_names",
]


def generate_launch_description():
    pkg_share = get_package_share_directory(PKG)
    urdf_xacro = os.path.join(pkg_share, "urdf", "lekIwi_control.urdf.xacro")
    controllers_yaml = os.path.join(pkg_share, "config", "controllers.yaml")
    joint_limits_yaml = os.path.join(pkg_share, "config", "joint_limits.yaml")

    use_sim_time = LaunchConfiguration("use_sim_time")

    # ---- robot_description: xacro evaluated at launch time ----------------
    robot_description = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", urdf_xacro]),
        value_type=str,
    )

    # ---- config/joint_limits.yaml -> parameter dict for the base node ------
    with open(joint_limits_yaml, "r") as fh:
        limits = yaml.safe_load(fh).get("joint_limits", {})
    base_params = {k: limits[k] for k in BASE_CTRL_PARAMS if k in limits}

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
            },
            controllers_yaml,
        ],
        output="screen",
    )

    # `--param-file` guarantees each controller node receives its own section
    # of config/controllers.yaml before it is configured.
    spawn_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--param-file", controllers_yaml,
            "--controller-manager-timeout", "60",
        ],
        output="screen",
    )

    spawn_arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_controller",
            "--controller-manager", "/controller_manager",
            "--param-file", controllers_yaml,
            "--controller-manager-timeout", "60",
        ],
        output="screen",
    )

    spawn_base = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "base_controller",
            "--controller-manager", "/controller_manager",
            "--param-file", controllers_yaml,
            "--controller-manager-timeout", "60",
        ],
        output="screen",
    )

    base_controller_node = Node(
        package=PKG,
        executable="lekIwi_base_controller",
        name="lekIwi_base_controller",
        output="screen",
        parameters=[
            base_params,
            {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description=(
                "Use /clock from Isaac Sim for ROS 2 timing. Keep false for "
                "controller_manager: if the simulation pauses the clock stops "
                "and the control loop would stall with the wheels still "
                "commanded. The /cmd_vel watchdog therefore runs on wall time."
            ),
        ),
        control_node,
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_jsb, on_exit=[spawn_arm])
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_arm, on_exit=[spawn_base])
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_base, on_exit=[base_controller_node]
            )
        ),
        spawn_jsb,
    ])
