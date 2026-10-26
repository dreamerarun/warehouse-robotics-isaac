#!/usr/bin/env python3
"""
Carter Robot Test for Isaac Sim 5.1

Uses the exact initialization from the official test_carter_v2.py:
1. Open USD stage first
2. Create World
3. Initialize simulation context (KEY STEP)
4. Play timeline
5. Create SingleArticulation
6. Initialize articulation view
7. Control via articulation_view
"""

import sys
import os
import numpy as np

# Write start marker
with open("/tmp/carter_results.txt", "w") as f:
    f.write("TEST_STARTED\n")

# ============================================================
# STEP 1: Start Isaac Sim 5.1
# ============================================================
from isaacsim import SimulationApp

sim_config = {
    "renderer": "iray",
    "headless": False,
    "window_width": 1280,
    "window_height": 720,
    "limit_sampling_buffer_size": True,
    "depth_value_enabled": False,
}

try:
    simulation_app = SimulationApp(sim_config)
except Exception as e:
    with open("/tmp/carter_results.txt", "w") as f:
        f.write(f"FAILED_LAUNCH: {e}\n")
    sys.exit(1)

# ============================================================
# STEP 2: Import Isaac Sim APIs (AFTER SimulationApp)
# ============================================================
import omni
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path

# ============================================================
# STEP 3: Create USD stage and load Carter
# ============================================================
stage = get_current_stage()
assets_root = get_assets_root_path()

# Load Carter USD via add_reference (this works)
floor_path = "/World/Floor"
add_reference_to_stage(
    usd_path=assets_root + "/Isaac/Props/Grid/plane.usd",
    prim_path=floor_path
)

CARTER_USD_PATH = assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
carter_path = "/World/Carter"
add_reference_to_stage(usd_path=CARTER_USD_PATH, prim_path=carter_path)
simulation_app.update()

# ============================================================
# STEP 4: Create World and initialize simulation context
# ============================================================
my_world = World(stage_units_in_meters=1.0)

# CRITICAL: Initialize simulation context - this is the key step from the official test
# that makes the physics scene ready before creating the articulation
try:
    my_world.initialize_simulation_context()
    print("Simulation context initialized")
except AttributeError as e:
    print(f"initialize_simulation_context not available: {e}")
    # Try async version or just proceed
except Exception as e:
    print(f"initialize_simulation_context error: {e}")

# Wait for physics to be ready after initializing context
for _ in range(30):
    simulation_app.update()

# Play the world timeline (per official test)
my_world.play()

# Wait for physics to be ready after playing
for _ in range(30):
    simulation_app.update()

# ============================================================
# STEP 5: Create SingleArticulation Carter
# ============================================================
# The official test uses "/nova_carter" as the prim path
# But our USD is loaded at /World/Carter - let's try both
ar = None

# Try /World/Carter first
try:
    ar = SingleArticulation("/World/Carter")
    print("SingleArticulation created at /World/Carter")
except Exception as e:
    print(f"SingleArticulation /World/Carter failed: {e}")

# If that failed, try /nova_carter
if ar is None:
    try:
        ar = SingleArticulation("/nova_carter")
        print("SingleArticulation created at /nova_carter")
    except Exception as e:
        print(f"SingleArticulation /nova_carter failed: {e}")

if ar is None:
    print("ERROR: Could not create SingleArticulation")
    sys.exit(1)

# Get the articulation view
articulation_view = ar._articulation_view

# Initialize the articulation view (per official test pattern)
# After timeline.play() and sufficient updates
try:
    articulation_view.initialize()
    print("Articulation view initialized successfully")
except Exception as e:
    print(f"Articulation view initialize info: {e}")

# ============================================================
# STEP 6: Get wheel joint indices and command Carter
# ============================================================
left_wheel_joint_idx = articulation_view.get_dof_index("joint_wheel_left")
right_wheel_joint_idx = articulation_view.get_dof_index("joint_wheel_right")

with open("/tmp/carter_results.txt", "a") as f:
    f.write(f"LEFT_WHEEL_IDX: {left_wheel_joint_idx}\n")
    f.write(f"RIGHT_WHEEL_IDX: {right_wheel_joint_idx}\n")
    try:
        pos = ar.get_world_pose()[0]
        f.write(f"INITIAL_POSITION: {pos}\n")
    except:
        f.write("INITIAL_POSITION: error getting pose\n")

# ============================================================
# STEP 7: Command Carter to move forward
# ============================================================
if left_wheel_joint_idx is not None and right_wheel_joint_idx is not None:
    forward_velocity = 1.0  # m/s
    articulation_view.set_joint_velocity_targets(
        velocities=np.array([forward_velocity, forward_velocity]),
        joint_indices=[left_wheel_joint_idx, right_wheel_joint_idx]
    )

    # Wait for motion
    for _ in range(30):
        simulation_app.update()

    # Get final position
    try:
        final_pos = ar.get_world_pose()[0]
    except:
        final_pos = [0.0, 0.0, 0.0]

    # Read initial position from file
    initial_pos = [0.0, 0.0, 0.0]
    try:
        with open("/tmp/carter_results.txt", "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                initial_pos = eval(lines[1].strip())
    except:
        pass

    distance_moved = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))

    with open("/tmp/carter_results.txt", "a") as f:
        f.write(f"FINAL_POSITION: {final_pos}\n")
        f.write(f"DISTANCE_MOVED: {distance_moved:.4f}\n")
        f.write(f"TEST_RESULT: {'PASS' if distance_moved > 0.01 else 'PARTIAL'}\n")
else:
    # If joint indices are None, just wait and check positions
    for _ in range(30):
        simulation_app.update()

    try:
        final_pos = ar.get_world_pose()[0]
    except:
        final_pos = [0.0, 0.0, 0.0]

    # Try to read initial position
    initial_pos_str = None
    try:
        with open("/tmp/carter_results.txt", "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                initial_pos_str = lines[1].strip()
    except:
        pass

    try:
        initial_pos = eval(initial_pos_str) if initial_pos_str else [0.0, 0.0, 0.0]
    except:
        initial_pos = [0.0, 0.0, 0.0]

    distance_moved = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))

    with open("/tmp/carter_results.txt", "a") as f:
        f.write(f"FINAL_POSITION: {final_pos}\n")
        f.write(f"DISTANCE_MOVED: {distance_moved:.4f}\n")
        f.write(f"TEST_RESULT: PARTIAL\n")

# ============================================================
# STEP 8: Stop Carter and close Isaac Sim
# ============================================================
if left_wheel_joint_idx is not None and right_wheel_joint_idx is not None:
    articulation_view.set_joint_velocity_targets(
        velocities=np.array([0.0, 0.0]),
        joint_indices=[left_wheel_joint_idx, right_wheel_joint_idx]
    )

for _ in range(10):
    simulation_app.update()

my_world.stop()
simulation_app.update()
simulation_app.close()

# Write completion marker
with open("/tmp/carter_results.txt", "a") as f:
    f.write("TEST_COMPLETED\n")

print("Carter test completed - check /tmp/carter_results.txt for results")