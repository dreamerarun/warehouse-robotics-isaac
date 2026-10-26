#!/usr/bin/env bash
# Sanity-check the live ROS 2 topics produced by the simulation.
#
#   scripts/check_topics.sh
#
# Run this in a second terminal while scripts/run_demo.sh is running.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
set +u
source /opt/ros/humble/setup.bash
set -u
if [[ -f "${ROOT}/ros2_ws/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/ros2_ws/install/setup.bash"
fi
set -u
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  RMW=${RMW_IMPLEMENTATION}"
echo
echo "=== topics ==="
ros2 topic list | sort
echo

check() {
    local topic="$1"
    printf '%-28s ' "${topic}"
    if ! ros2 topic list 2>/dev/null | grep -qx "${topic}"; then
        echo "MISSING"
        return
    fi
    local out
    out="$(timeout 6 ros2 topic hz "${topic}" 2>&1 | grep -m1 'average rate' || true)"
    if [[ -n "${out}" ]]; then
        echo "${out}"
    else
        echo "present, no messages in 6 s"
    fi
}

echo "=== rates ==="
for topic in /amr1/odom /amr2/odom /amr3/odom \
             /amr1/scan /amr2/scan /amr3/scan \
             /tf /fleet/status /franka/joint_states /humanoid/odom \
             /ur10/joint_states /forklift/odom; do
    check "${topic}"
done
