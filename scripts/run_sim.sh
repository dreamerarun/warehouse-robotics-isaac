#!/usr/bin/env bash
# Run the Isaac Sim warehouse scene only (no ROS 2 autonomy).
#
#   scripts/run_sim.sh                     # visual
#   scripts/run_sim.sh --headless          # headless, real-time
#   scripts/run_sim.sh --headless --no-realtime --duration 30
#   scripts/run_sim.sh --no-franka --no-humanoid
#
# All arguments are passed straight to scripts/warehouse_sim.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64}"
PYTHON_SH="${ISAAC_SIM_ROOT}/python.sh"

if [[ ! -x "${PYTHON_SH}" ]]; then
    echo "error: Isaac Sim launcher not found: ${PYTHON_SH}" >&2
    echo "       set ISAAC_SIM_ROOT to your Isaac Sim 5.1 installation" >&2
    exit 1
fi

cd "${ROOT}"
exec "${PYTHON_SH}" scripts/warehouse_sim.py "$@"
