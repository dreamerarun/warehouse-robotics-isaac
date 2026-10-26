# SPDX-License-Identifier: Apache-2.0
"""LeKiwi model constants shared by the three Isaac Sim scripts.

Everything here is *derived from the source URDF*
``/home/arun/Desktop/lekiwi.urdf`` (45 links / 44 joints / 9 moving joints) and
mirrors what the ROS 2 side declares in:

* ``lekIwi_ros2_control/urdf/lekIwi_control.urdf.xacro``  (``<ros2_control>``)
* ``lekIwi_ros2_control/config/controllers.yaml``
* ``lekIwi_ros2_control/config/joint_limits.yaml``

``config/joint_limits.yaml`` is read back for the numbers that both sides must
agree on, so a divergence fails loudly instead of silently.

Naming note
-----------
The requested arm joint names omit the leading ``S`` (``ST3215_03a_...``).  The
URDF actually contains ``STS3215_03a_...``.  These constants use the **real**
URDF names -- otherwise nothing would ever match.
"""

import os
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get(
    "LEKIWI_PROJECT_ROOT", "/home/arun/warehouse_robotics_isaac"
)
ISAAC_SIM_ROOT = os.environ.get(
    "ISAAC_SIM_ROOT", "/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64"
)

SOURCE_URDF = os.environ.get("LEKIWI_URDF", "/home/arun/Desktop/lekiwi.urdf")
MESHES_DIR = os.environ.get("LEKIWI_MESHES", "/home/arun/Desktop/meshes")

ASSET_DIR = os.path.join(PROJECT_ROOT, "assets", "lekiwi")
IMPORTED_USD = os.path.join(ASSET_DIR, "lekiwi_imported.usd")
ROS2_USD = os.path.join(ASSET_DIR, "lekiwi_ros2.usd")

CONFIG_YAML = os.path.join(
    PROJECT_ROOT, "ros2_ws", "src", "lekIwi_ros2_control", "config", "joint_limits.yaml"
)

# --------------------------------------------------------------------------
# Prim / OmniGraph layout (must match README + ROS 2 side)
# --------------------------------------------------------------------------
ROBOT_PRIM = "/World/LeKiwi"
GRAPH_PRIM = "/World/ActionGraph"
PHYSICS_SCENE_PRIM = "/physicsScene"
GROUND_PRIM = "/groundPlane"

# --------------------------------------------------------------------------
# Joint groups
# --------------------------------------------------------------------------
#: 3 omni-directional wheels -- velocity commanded.
WHEEL_JOINTS = [
    "ST3215_Servo_Motor_v1_2_Revolute_60",
    "ST3215_Servo_Motor_v1_1_Revolute_62",
    "ST3215_Servo_Motor_v1_Revolute_64",
]

#: 6 arm joints -- position commanded.  NOTE: the sixth (``..._Revolute_57``)
#: drives the gripper jaw link ``Moving_Jaw_08d_v1``.
ARM_JOINTS = [
    "STS3215_03a_v1_Revolute_45",
    "STS3215_03a_v1_1_Revolute_49",
    "STS3215_03a_v1_2_Revolute_51",
    "STS3215_03a_v1_3_Revolute_53",
    "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "STS3215_03a_v1_4_Revolute_57",
]

ALL_JOINTS = WHEEL_JOINTS + ARM_JOINTS

#: Gripper jaw -- only joint whose travel is smaller than +-180 deg.
GRIPPER_JOINTS = ["STS3215_03a_v1_4_Revolute_57"]

# --------------------------------------------------------------------------
# ROS 2 topic names (no leading slash: Isaac resolves them against the empty
# node namespace, producing exactly these absolute names)
# --------------------------------------------------------------------------
TOPIC_JOINT_STATES = "lekIwi/joint_states_isaac"   # Isaac -> ROS 2
TOPIC_WHEEL_COMMAND = "lekIwi/wheel_command"       # ROS 2 -> Isaac
TOPIC_ARM_COMMAND = "lekIwi/arm_command"           # ROS 2 -> Isaac
TOPIC_CLOCK = "clock"
TOPIC_TF = "tf"

# --------------------------------------------------------------------------
# Geometry (measured from the URDF / the STL meshes)
# --------------------------------------------------------------------------
#: Wheel radius in metres: STL `name_omni_directional_wheel_single_body_...`
#: is 101.569 mm in diameter.  Same value as `wheel_radius` in joint_limits.yaml.
WHEEL_RADIUS = 0.0507845

#: z of all three wheel joint origins expressed in the robot root frame
#: (identical for the three wheels in the URDF).
WHEEL_CENTER_Z = 0.01786


def wheel_lift(urdf_path=None):
    """Height at which the robot root must sit so the wheels touch z = 0."""
    path = urdf_path or SOURCE_URDF
    zs = []
    if os.path.isfile(path):
        root = ET.parse(path).getroot()
        for joint in root.findall("joint"):
            if joint.get("name") in WHEEL_JOINTS:
                origin = joint.find("origin")
                zs.append(float(origin.get("xyz", "0 0 0").split()[2]))
    if not zs:
        zs = [WHEEL_CENTER_Z]
    return WHEEL_RADIUS - sum(zs) / len(zs)


# --------------------------------------------------------------------------
# Joint drive / limit tuning (USD: angles in DEGREES, drives per unit as below)
# --------------------------------------------------------------------------
# Velocity drive for the wheels: stiffness 0, pure damping, hard force cap.
# This is the pattern used by Isaac Sim's own KAYA sample (see
# impl/samples/import_kaya.py::_on_config_drives).  The ROS 2 side already
# ramps /cmd_vel (acceleration limiting) and the forward controller clamps wheel
# speed, so a stiff velocity servo here is safe.
WHEEL_DRIVE = {"stiffness": 0.0, "damping": 20.0, "max_force": 10.0}

# Position drive for the arm.  The source URDF declares every arm joint as
# `continuous` with no <limit>, so these are *conservative simulation* values
# that must be calibrated against real STS3215 servos before any hardware use.
ARM_DRIVE = {"stiffness": 100.0, "damping": 10.0, "max_force": 10.0}

#: USD `physics:lowerLimit` / `physics:upperLimit` in DEGREES.
ARM_LIMIT_DEG = {name: (-90.0, 90.0) for name in GRIPPER_JOINTS}
ARM_LIMIT_DEG.update(
    {name: (-180.0, 180.0) for name in ARM_JOINTS if name not in GRIPPER_JOINTS}
)
#: Wheels spin without end stops.
WHEEL_LIMIT_DEG = (-1.0e9, 1.0e9)

# --------------------------------------------------------------------------
# ros2_control hardware topic names (must match lekIwi_control.urdf.xacro)
# --------------------------------------------------------------------------
STATE_TOPIC = "/lekIwi/joint_states_isaac"
VELOCITY_COMMAND_TOPIC = "/lekIwi/wheel_command"
POSITION_COMMAND_TOPIC = "/lekIwi/arm_command"


def load_joint_limits():
    """Parse the ROS 2 joint_limits.yaml, returning None if it is unavailable."""
    if not os.path.isfile(CONFIG_YAML):
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    with open(CONFIG_YAML, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("joint_limits")


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    import json

    limits = load_joint_limits()
    print("wheel_lift            =", wheel_lift())
    print("WHEEL_RADIUS          =", WHEEL_RADIUS)
    print("robot prim            =", ROBOT_PRIM)
    print("graph prim            =", GRAPH_PRIM)
    print("wheels                =", WHEEL_JOINTS)
    print("arm                   =", ARM_JOINTS)
    print("joint_limits.yaml     =", "loaded" if limits else "NOT FOUND", CONFIG_YAML)
    if limits:
        check = {
            "wheel_radius matches": abs(limits.get("wheel_radius", 0) - WHEEL_RADIUS) < 1e-9,
            "wheel joints match": limits.get("wheel_joint_names") == WHEEL_JOINTS,
            "arm joints match": limits.get("arm_joint_names") == ARM_JOINTS,
            "cmd_vel_timeout": limits.get("cmd_vel_timeout"),
        }
        print(json.dumps(check, indent=2))
