#!/usr/bin/env bash
# Run only the ROS 2 autonomy stack against an already running simulation.
#
#   scripts/run_autonomy.sh
#   scripts/run_autonomy.sh monitor:=false rviz:=true
#   scripts/run_autonomy.sh robots:=amr1,amr2 franka:=false
#
# Anything after the script name is forwarded to `ros2 launch`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${ROOT}/ros2_ws"

set +u
if [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
else
    echo "error: /opt/ros/humble/setup.bash not found (ROS 2 Humble required)" >&2
    exit 1
fi

if [[ -f "${WS}/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "${WS}/install/setup.bash"
else
    echo "error: workspace not built. Run:" >&2
    echo "       cd ${WS} && colcon build --symlink-install" >&2
    exit 1
fi
set -u

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export WAREHOUSE_CONFIG="${WAREHOUSE_CONFIG:-${ROOT}/config/warehouse.yaml}"

echo "[autonomy] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  RMW=${RMW_IMPLEMENTATION}"
echo "[autonomy] config=${WAREHOUSE_CONFIG}"
exec ros2 launch warehouse_fleet fleet.launch.py "$@"
