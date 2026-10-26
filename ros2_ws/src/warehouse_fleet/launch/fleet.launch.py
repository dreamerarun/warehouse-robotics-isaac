"""Launch the warehouse autonomy stack (fleet manager + all robot agents).

The Isaac Sim process is started separately with ``scripts/run_sim.sh``; this
launch file only starts the ROS 2 side::

    ros2 launch warehouse_fleet fleet.launch.py
    ros2 launch warehouse_fleet fleet.launch.py monitor:=false rviz:=true
    ros2 launch warehouse_fleet fleet.launch.py robots:=amr1,amr2 franka:=false

Robot names, Franka and humanoid flags default to whatever is enabled in
``config/warehouse.yaml``, so there is only one place to change the fleet.
"""

from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_config():
    share = get_package_share_directory("warehouse_fleet")
    installed = os.path.join(share, "config", "warehouse.yaml")
    if os.path.isfile(installed):
        return installed
    return os.path.join(os.getcwd(), "config", "warehouse.yaml")


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _launch_setup(context, *args, **kwargs):
    config = LaunchConfiguration("config").perform(context)
    if not os.path.isfile(config):
        raise RuntimeError(f"warehouse config not found: {config}")
    cfg = _load(config)

    robots_arg = LaunchConfiguration("robots").perform(context).strip()
    if robots_arg:
        robots = [r.strip() for r in robots_arg.split(",") if r.strip()]
    else:
        robots = [a["name"] for a in cfg.get("amrs", [])]

    def flag(name):
        return LaunchConfiguration(name).perform(context).lower() in (
            "1", "true", "yes", "on",
        )

    actions = [
        LogInfo(msg=f"[warehouse] config: {config}"),
        LogInfo(msg=f"[warehouse] AMRs: {', '.join(robots) or 'none'}"),
    ]

    domain = LaunchConfiguration("domain_id").perform(context).strip()
    if domain:
        actions.append(SetEnvironmentVariable("ROS_DOMAIN_ID", domain))

    nodes = [
        Node(
            package="warehouse_fleet",
            executable="fleet_manager",
            name="fleet_manager",
            output="screen",
            parameters=[
                {
                    "config": config,
                    "publish_markers": flag("markers"),
                }
            ],
        )
    ]
    for robot in robots:
        nodes.append(
            Node(
                package="warehouse_fleet",
                executable="amr_agent",
                name=f"{robot}_agent",
                output="screen",
                parameters=[{"robot": robot, "config": config}],
            )
        )

    if flag("franka") and cfg.get("franka", {}).get("enabled", False):
        nodes.append(
            Node(
                package="warehouse_fleet",
                executable="franka_arm",
                name="franka_arm",
                output="screen",
                parameters=[{"config": config, "loop": flag("franka_loop")}],
            )
        )

    if flag("humanoid") and cfg.get("humanoid", {}).get("enabled", False):
        nodes.append(
            Node(
                package="warehouse_fleet",
                executable="humanoid_patrol",
                name="humanoid_patrol",
                output="screen",
                parameters=[{"config": config}],
            )
        )

    if flag("monitor"):
        nodes.append(
            Node(
                package="warehouse_fleet",
                executable="fleet_monitor",
                name="fleet_monitor",
                output="screen",
                parameters=[{"config": config}],
            )
        )

    if flag("rviz"):
        share = get_package_share_directory("warehouse_fleet")
        rviz_config = os.path.join(share, "rviz", "fleet.rviz")
        if os.path.isfile(rviz_config):
            nodes.append(
                Node(
                    package="rviz2",
                    executable="rviz2",
                    name="rviz2",
                    output="screen",
                    arguments=["-d", rviz_config],
                )
            )

    return actions + nodes


def generate_launch_description():
    defaults = [
        DeclareLaunchArgument(
            "config",
            default_value=_default_config(),
            description="Path to warehouse.yaml",
        ),
        DeclareLaunchArgument(
            "robots",
            default_value="",
            description="Comma separated AMR names (default: all in config)",
        ),
        DeclareLaunchArgument("franka", default_value="true"),
        DeclareLaunchArgument("franka_loop", default_value="true"),
        DeclareLaunchArgument("humanoid", default_value="true"),
        DeclareLaunchArgument("monitor", default_value="true"),
        DeclareLaunchArgument("markers", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument(
            "domain_id",
            default_value="",
            description="ROS_DOMAIN_ID; leave empty to inherit the shell",
        ),
    ]
    return LaunchDescription(defaults + [OpaqueFunction(function=_launch_setup)])
