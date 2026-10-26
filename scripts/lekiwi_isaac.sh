#!/usr/bin/env bash
# ==============================================================================
# lekiwi_isaac.sh -- run a LeKiwi Isaac Sim script with a working ROS 2 runtime
# ------------------------------------------------------------------------------
# Usage:
#   ./scripts/lekiwi_isaac.sh scripts/import_lekIwi_urdf.py [script args...]
#   ./scripts/lekiwi_isaac.sh scripts/create_lekIwi_action_graph.py
#   ./scripts/lekiwi_isaac.sh scripts/validate_lekIwi.py
#
# What it does (and why §21 keeps Isaac Sim out of the ROS 2 launch files):
#
#   1. forces ROS_DOMAIN_ID (default 42) so Isaac Sim and the host ROS 2
#      discover each other;
#   2. makes sure *some* ROS 2 runtime is reachable through LD_LIBRARY_PATH.
#      glibc reads LD_LIBRARY_PATH when the process starts, so it cannot be
#      fixed from Python after the fact:
#        - host /opt/ros/humble/lib is preferred when it exists, otherwise
#        - Isaac Sim's bundled exts/isaacsim.ros2.bridge/humble/lib.
#   3. execs Isaac Sim's python.sh, which embeds CPython 3.11 + Kit.
#
# A sourced host ROS 2 environment is fine: the Python half of it is removed
# from PYTHONPATH by scripts/lekiwi_ros_env.py before Kit starts.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64}"
ISAAC_ROS_DISTRO="${ISAAC_ROS_DISTRO:-humble}"
export ISAAC_SIM_ROOT ISAAC_ROS_DISTRO
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <script.py> [args...]" >&2
    exit 2
fi

if [ ! -x "$ISAAC_SIM_ROOT/python.sh" ]; then
    echo "ERROR: ISAAC_SIM_ROOT=$ISAAC_SIM_ROOT does not contain python.sh" >&2
    exit 1
fi

ROS_LIB=""
for candidate in \
    "/opt/ros/$ISAAC_ROS_DISTRO/lib" \
    "$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/$ISAAC_ROS_DISTRO/lib"; do
    if [ -d "$candidate" ]; then
        ROS_LIB="$candidate"
        break
    fi
done

if [ -z "$ROS_LIB" ]; then
    echo "ERROR: no ROS 2 runtime found (looked in /opt/ros/$ISAAC_ROS_DISTRO/lib" >&2
    echo "       and $ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/$ISAAC_ROS_DISTRO/lib)" >&2
    exit 1
fi

case ":${LD_LIBRARY_PATH:-}:" in
    *":$ROS_LIB:"*) ;;
    *) export LD_LIBRARY_PATH="$ROS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

exec "$ISAAC_SIM_ROOT/python.sh" "$@"
