#!/usr/bin/env python3
"""Normalise the process environment before Isaac Sim starts.

Isaac Sim ships its own CPython 3.11 and a ROS 2 Humble runtime inside the
``isaacsim.ros2.bridge`` extension.  The host ROS 2 Humble installation uses
CPython 3.10.  If the host environment is sourced when ``python.sh`` starts,
Isaac's interpreter picks up the host ``rclpy`` (compiled for 3.10), and the
bridge refuses to start.

This module must be imported *before* ``from isaacsim import SimulationApp``::

    import warehouse_ros_env
    report = warehouse_ros_env.prepare(domain_id=0)
    from isaacsim import SimulationApp

It also points ``LD_LIBRARY_PATH`` at the bridge's bundled Humble libraries so
that the bridge's own ``rclpy`` can load every ROS 2 shared object it needs.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.environ.get(
    "WAREHOUSE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
ISAAC_SIM_ROOT = os.environ.get(
    "ISAAC_SIM_ROOT", "/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64"
)
ISAAC_ROS_DISTRO = "humble"


def _drop_host_ros_python():
    """Remove /opt/ros entries from PYTHONPATH and sys.path (wrong ABI)."""
    dropped = []
    kept = []
    for entry in os.environ.get("PYTHONPATH", "").split(":"):
        if entry and entry.startswith("/opt/ros/"):
            dropped.append(entry)
        elif entry:
            kept.append(entry)
    os.environ["PYTHONPATH"] = ":".join(kept)

    before = list(sys.path)
    sys.path[:] = [p for p in sys.path if not p.startswith("/opt/ros/")]
    dropped += [p for p in before if p.startswith("/opt/ros/")]
    return dropped


def prepare(domain_id=None, verbose=True):
    """Make the embedded interpreter safe for ``isaacsim.ros2.bridge``.

    Args:
        domain_id: ROS_DOMAIN_ID to force, or None to keep the current value.
        verbose: print a one-line summary.

    Returns:
        dict: diagnostic summary.
    """
    dropped = _drop_host_ros_python()

    # The bridge chooses its bundled Humble runtime from ROS_DISTRO.
    os.environ.setdefault("ROS_DISTRO", ISAAC_ROS_DISTRO)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
    if domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)

    bundled = os.path.join(
        ISAAC_SIM_ROOT, "exts", "isaacsim.ros2.bridge", ISAAC_ROS_DISTRO, "lib"
    )
    ld_entries = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if os.path.isdir(bundled) and bundled not in ld_entries:
        ld_entries.insert(0, bundled)
    # Keep the host runtime available as a fallback for tools that only look
    # for system libraries; the bundled copy is first in the search order.
    host = f"/opt/ros/{ISAAC_ROS_DISTRO}/lib"
    if os.path.isdir(host) and host not in ld_entries:
        ld_entries.append(host)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_entries)

    report = {
        "dropped_host_ros_pythonpath": dropped,
        "ROS_DISTRO": os.environ["ROS_DISTRO"],
        "RMW_IMPLEMENTATION": os.environ["RMW_IMPLEMENTATION"],
        "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "<unset>"),
        "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
    }
    if verbose:
        print(
            "[warehouse_ros_env] ROS_DISTRO={ROS_DISTRO} "
            "RMW={RMW_IMPLEMENTATION} DOMAIN={ROS_DOMAIN_ID}".format(**report),
            flush=True,
        )
        if dropped:
            print(
                "[warehouse_ros_env] dropped host ROS 2 Python paths: "
                + ", ".join(dropped),
                flush=True,
            )
    return report


if __name__ == "__main__":  # diagnostic entry point
    import json

    print(json.dumps(prepare(), indent=2))
