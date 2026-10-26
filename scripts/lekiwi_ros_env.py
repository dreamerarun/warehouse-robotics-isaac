# SPDX-License-Identifier: Apache-2.0
"""ROS 2 environment normalisation for the LeKiwi Isaac Sim scripts.

Isaac Sim 5.1 embeds **CPython 3.11** plus its own ROS 2 Humble runtime under::

    <ISAAC_SIM_ROOT>/exts/isaacsim.ros2.bridge/humble/lib
    <ISAAC_SIM_ROOT>/exts/isaacsim.ros2.bridge/humble/rclpy

The host ROS 2 Humble at ``/opt/ros/humble`` ships **CPython 3.10** binaries.
If the host ROS 2 environment has been sourced before launching Isaac Sim, its
``.../lib/python3.10/site-packages`` directory sits on ``PYTHONPATH`` and the
embedded interpreter picks up ``rclpy`` compiled for a different Python, which
makes ``import rclpy`` fail with::

    The C extension '_rclpy_pybind11.cpython-310-...' isn't present for ...

``python.sh`` inherits ``PYTHONPATH`` from the calling shell, so this module
removes every ``/opt/ros`` entry from the environment **before**
``from isaacsim import SimulationApp`` runs (and therefore before Kit starts).

The shared *library* paths are deliberately left alone: ``librclcpp`` and
friends are Python-agnostic, and Isaac Sim's bridge is fully interoperable with
the host ROS 2 Humble DDS graph when both are on ``ROS_DOMAIN_ID``.
``scripts/lekiwi_isaac.sh`` additionally guarantees that some ROS 2 runtime is
reachable via ``LD_LIBRARY_PATH`` for the case where neither ROS has been
sourced at all.

Import this module first::

    import lekiwi_ros_env
    lekiwi_ros_env.prepare()

    from isaacsim import SimulationApp
"""

import os
import sys

#: Installed Isaac Sim 5.1 location (override with the ISAAC_SIM_ROOT variable).
ISAAC_SIM_ROOT = os.environ.get(
    "ISAAC_SIM_ROOT", "/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64"
)

#: ROS 2 distro that matches the host ROS 2 Humble installation.
ISAAC_ROS_DISTRO = "humble"


def prepare(ros_domain_id=None):
    """Make the embedded interpreter safe to load Isaac Sim's ROS 2 bridge.

    Args:
        ros_domain_id (int | str | None): If given, forces ``ROS_DOMAIN_ID`` so
            that Isaac Sim and the host ROS 2 graph definitely agree.

    Returns:
        dict: diagnostic summary of what changed.
    """
    report = {}

    # 1. Drop host ROS 2 Python packages (wrong interpreter binary).
    pypath = os.environ.get("PYTHONPATH", "")
    kept = [p for p in pypath.split(":") if p and not p.startswith("/opt/ros/")]
    dropped = [p for p in pypath.split(":") if p and p.startswith("/opt/ros/")]
    if dropped:
        os.environ["PYTHONPATH"] = ":".join(kept)
    before = len(sys.path)
    sys.path[:] = [p for p in sys.path if not p.startswith("/opt/ros/")]
    report["dropped_pythonpath"] = dropped
    report["dropped_syspath"] = before - len(sys.path)

    # 2. ROS domain: the bridge honours ROS_DOMAIN_ID through
    #    ROS2Context(useDomainIDEnvVar=True), so just make sure it is set.
    if ros_domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(ros_domain_id)
    report["ROS_DOMAIN_ID"] = os.environ.get("ROS_DOMAIN_ID", "<unset>")

    # 3. Same RMW on both sides keeps DDS discovery simple.
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
    report["RMW_IMPLEMENTATION"] = os.environ["RMW_IMPLEMENTATION"]

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    report["ld_has_host_ros"] = "/opt/ros/" in ld
    report["ld_has_isaac_ros"] = any(
        ISAAC_ROS_DISTRO in p and "ros2.bridge" in p for p in ld.split(":") if p
    )
    return report


def ros_library_dirs():
    """Return candidate ROS 2 shared-library directories, most preferred first."""
    dirs = []
    host = "/opt/ros/{}/lib".format(ISAAC_ROS_DISTRO)
    if os.path.isdir(host):
        dirs.append(host)
    bundled = os.path.join(ISAAC_SIM_ROOT, "exts", "isaacsim.ros2.bridge", ISAAC_ROS_DISTRO, "lib")
    if os.path.isdir(bundled):
        dirs.append(bundled)
    return dirs


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    import json

    print(json.dumps(prepare(), indent=2))
    print("ros_library_dirs() ->", ros_library_dirs())
