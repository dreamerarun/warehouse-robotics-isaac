#!/usr/bin/env bash
# One-command demo: Isaac Sim + ROS 2 fleet autonomy.
#
#   scripts/run_demo.sh                       # visual simulation + autonomy
#   scripts/run_demo.sh --headless            # headless (recommended on laptops)
#   scripts/run_demo.sh --headless --duration 120
#   scripts/run_demo.sh --camera ur10 --no-hud
#   scripts/run_demo.sh --no-franka --no-humanoid --no-industrial
#   scripts/run_demo.sh --people --headless --nvidia-assets --duration 60
#   scripts/run_demo.sh -- --ros-args ...     # everything after -- goes to launch
#
# Ctrl+C stops both processes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${ROOT}/ros2_ws"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64}"
PYTHON_SH="${ISAAC_SIM_ROOT}/python.sh"

HEADLESS=0
BUILD=0
AUTONOMY=1
SIM_ARGS=()
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless) HEADLESS=1; SIM_ARGS+=(--headless); shift ;;
        --duration) SIM_ARGS+=(--duration "$2"); shift 2 ;;
        --no-franka) SIM_ARGS+=(--no-franka); LAUNCH_ARGS+=(franka:=false); shift ;;
        --no-humanoid) SIM_ARGS+=(--no-humanoid); LAUNCH_ARGS+=(humanoid:=false); shift ;;
        --no-industrial) SIM_ARGS+=(--no-industrial); shift ;;
        --nvidia-assets) SIM_ARGS+=(--nvidia-assets); shift ;;
        --no-nvidia-assets) SIM_ARGS+=(--no-nvidia-assets); shift ;;
        --camera) SIM_ARGS+=(--camera "$2"); shift 2 ;;
        --no-hud) SIM_ARGS+=(--no-hud); shift ;;
        --people) SIM_ARGS+=(--people); shift ;;
        --no-people) SIM_ARGS+=(--no-people); shift ;;
        --no-autonomy) AUTONOMY=0; shift ;;
        --build) BUILD=1; shift ;;
        --domain-id) SIM_ARGS+=(--domain-id "$2"); export ROS_DOMAIN_ID="$2"; shift 2 ;;
        --) shift; LAUNCH_ARGS+=("$@"); break ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "error: unknown option: $1 (use --help)" >&2
            exit 2 ;;
    esac
done

if [[ ! -x "${PYTHON_SH}" ]]; then
    echo "error: Isaac Sim launcher not found: ${PYTHON_SH}" >&2
    echo "       set ISAAC_SIM_ROOT to your Isaac Sim 5.1 installation" >&2
    exit 1
fi

set +u
if [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
else
    echo "error: /opt/ros/humble/setup.bash not found (ROS 2 Humble required)" >&2
    exit 1
fi

if [[ ! -f "${WS}/install/setup.bash" || "${BUILD}" -eq 1 ]]; then
    # The warehouse demo only needs warehouse_fleet.  The workspace also
    # contains the optional LeKiwi packages, whose ros2_control dependencies
    # are not required to run this demo; build those separately if wanted.
    echo "[demo] building warehouse_fleet ..."
    ( cd "${WS}" && colcon build --symlink-install --packages-select warehouse_fleet )
fi
# shellcheck disable=SC1091
source "${WS}/install/setup.bash"
set -u

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export WAREHOUSE_CONFIG="${WAREHOUSE_CONFIG:-${ROOT}/config/warehouse.yaml}"

SIM_PID=""
LAUNCH_PID=""
READY=0

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "[demo] shutting down ..."
    if [[ -n "${LAUNCH_PID}" ]]; then
        kill -INT -- "-${LAUNCH_PID}" 2>/dev/null || kill -INT "${LAUNCH_PID}" 2>/dev/null || true
    fi
    if [[ -n "${SIM_PID}" ]]; then
        kill -INT -- "-${SIM_PID}" 2>/dev/null || kill -INT "${SIM_PID}" 2>/dev/null || true
    fi
    sleep 1
    if [[ -n "${LAUNCH_PID}" ]]; then
        kill -TERM -- "-${LAUNCH_PID}" 2>/dev/null || kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
    fi
    if [[ -n "${SIM_PID}" ]]; then
        kill -TERM -- "-${SIM_PID}" 2>/dev/null || kill -TERM "${SIM_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "[demo] stopped"
}
trap cleanup EXIT INT TERM

cd "${ROOT}"
echo "[demo] starting Isaac Sim (${ISAAC_SIM_ROOT}) ..."
setsid "${PYTHON_SH}" scripts/warehouse_sim.py \
    --domain-id "${ROS_DOMAIN_ID}" "${SIM_ARGS[@]}" &
SIM_PID=$!

echo "[demo] waiting for /amr1/odom (the first launch can take ~30 s) ..."
for _ in $(seq 1 120); do
    if ! kill -0 "${SIM_PID}" 2>/dev/null; then
        echo "error: Isaac Sim exited during startup" >&2
        wait "${SIM_PID}" || true
        exit 1
    fi
    if ros2 topic list 2>/dev/null | grep -qx "/amr1/odom"; then
        READY=1
        break
    fi
    sleep 1
done

if [[ "${READY}" -eq 0 ]]; then
    echo "warning: /amr1/odom not seen yet; starting autonomy anyway" >&2
fi

if [[ "${AUTONOMY}" -eq 0 ]]; then
    echo "[demo] simulation only. Ctrl+C to stop."
    wait "${SIM_PID}" || true
    exit 0
fi

echo "[demo] starting autonomy stack ..."
setsid ros2 launch warehouse_fleet fleet.launch.py "${LAUNCH_ARGS[@]}" &
LAUNCH_PID=$!

echo "[demo] running. Press Ctrl+C to stop."
wait -n "${SIM_PID}" "${LAUNCH_PID}" 2>/dev/null || true

# If the simulator was started with --duration it exits first; stop autonomy
# with it.  If autonomy died, also tear the simulation down.
sleep 1
exit 0
