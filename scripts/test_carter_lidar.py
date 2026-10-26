#!/usr/bin/env python3
"""
Carter LiDAR Perception Test in a Static Warehouse - Isaac Sim 5.1

Milestone 4: lightweight LiDAR perception for Carter.

Reuses the proven initialization pattern from test_carter_warehouse.py:
- Same SimulationApp config (headless, iray)
- Same Carter asset (NovaCarter/nova_carter.usd)
- Same lightweight warehouse (floor + 4 walls + 6 static racks)
- Same physics ground plane + World init + SingleArticulation

Adds ONE LiDAR sensor following the official Isaac Sim 5.1 example
standalone_examples/api/isaacsim.sensors.physx/rotating_lidar_physX.py
(PhysX ray-cast LiDAR -> lightweight, headless friendly, no ROS2/Nav2).

Layout (top view, meters):
    x: -10 .. 10,  y: -10 .. 10
    Racks (4 x 1.5 x 2) at (+-5, -4), (+-5, 0), (+-5, 4)
    Carter START (0, -7) in central aisle
    -> south wall ~2.8m behind, racks ~3.8m away (all within LiDAR range)

Verification output:
    LIDAR_INITIALIZED: YES
    LIDAR_DATA_AVAILABLE: YES
    MIN_DISTANCE: <meters>
    MAX_DISTANCE: <meters>
    VALID_RAYS: <count>
    LIDAR_TEST: PASS
"""

import sys
import os
import numpy as np

# Write start marker
with open("/tmp/carter_lidar_results.txt", "w") as f:
    f.write("TEST_STARTED\n")


def log_print(msg):
    """Print to console and write to file"""
    print(msg)
    with open("/tmp/carter_lidar_results.txt", "a") as f:
        f.write(msg + "\n")


# ============================================================
# STEP 1: Start Isaac Sim 5.1
# ============================================================
from isaacsim import SimulationApp

sim_config = {
    "renderer": "iray",
    "headless": True,
    "window_width": 1280,
    "window_height": 720,
    "limit_sampling_buffer_size": True,
    "depth_value_enabled": False,
}

simulation_app = SimulationApp(sim_config)
log_print("SimulationApp created")

# ============================================================
# STEP 2: Import Isaac Sim APIs (AFTER SimulationApp)
# ============================================================
import omni
import carb
from pxr import Gf, PhysicsSchemaTools, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
# PhysX LiDAR (official example: rotating_lidar_physX.py)
from isaacsim.sensors.physx import RotatingLidarPhysX

log_print("Imports complete")

# ============================================================
# STEP 3: Configure physics rate (per official test)
# ============================================================
physics_rate = 60
carb.settings.get_settings().set_bool("/app/runLoops/main/rateLimitEnabled", True)
carb.settings.get_settings().set_int("/app/runLoops/main/rateLimitFrequency", int(physics_rate))
carb.settings.get_settings().set_int("/persistent/simulation/minFrameRate", int(physics_rate))
physics_dt = 1 / physics_rate

# ============================================================
# STEP 4: Create USD stage and load Carter
# ============================================================
stage = get_current_stage()
assets_root = get_assets_root_path()

# Load Carter USD as reference (matching test_carter.py approach that works)
CARTER_USD_PATH = assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
carter_path = "/World/Carter"
add_reference_to_stage(usd_path=CARTER_USD_PATH, prim_path=carter_path)

# Add proper physics ground plane (per official test)
PhysicsSchemaTools.addGroundPlane(
    stage, "/groundPlane", "Z", 1500, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
)

# Also add visual floor
floor_path = "/World/Floor"
add_reference_to_stage(
    usd_path=assets_root + "/Isaac/Props/Grid/plane.usd",
    prim_path=floor_path
)


# ------------------------------------------------------------
# Warehouse helper: static collision box (UsdGeom.Cube + CollisionAPI)
# ------------------------------------------------------------
def add_warehouse_box(prim_path, position, size, color):
    """Create a static box prim with proper collision geometry."""
    xform = UsdGeom.Xformable(stage.DefinePrim(prim_path, "Xform"))
    cube = UsdGeom.Cube.Define(stage, prim_path + "/Geom")
    cube.CreateSizeAttr(1.0)  # unit cube, full size comes from Xform scale
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform.AddTranslateOp().Set(Gf.Vec3f(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    # Static collider: CollisionAPI only (no RigidBodyAPI -> static, zero cost)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return xform


# ------------------------------------------------------------
# Warehouse layout: 20x20m floor, outer walls, 3 pairs of racks
# (identical to test_carter_warehouse.py)
# ------------------------------------------------------------
WALL_COLOR = (0.6, 0.6, 0.65)
RACK_COLOR = (0.25, 0.4, 0.7)

# Outer walls (height 3m, thickness 0.4m) at the +/-10m boundaries
add_warehouse_box("/World/Wall_South", (0, -10, 1.5), (20.6, 0.4, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_North", (0, 10, 1.5), (20.6, 0.4, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_West", (-10, 0, 1.5), (0.4, 20.6, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_East", (10, 0, 1.5), (0.4, 20.6, 3.0), WALL_COLOR)

# 3 pairs of rack/box obstacles (4m wide x 1.5m deep x 2m tall)
RACK_SIZE = (4.0, 1.5, 2.0)
RACK_Y = (-4.0, 0.0, 4.0)
for i, y in enumerate(RACK_Y):
    add_warehouse_box(f"/World/Rack_L{i + 1}", (-5.0, y, 1.0), RACK_SIZE, RACK_COLOR)
    add_warehouse_box(f"/World/Rack_R{i + 1}", (5.0, y, 1.0), RACK_SIZE, RACK_COLOR)

log_print("Warehouse loaded (walls + 6 racks)")

# ============================================================
# STEP 4b: Add ONE LiDAR sensor to Carter
# ============================================================
# Follows the official Isaac Sim 5.1 example exactly:
#   standalone_examples/api/isaacsim.sensors.physx/rotating_lidar_physX.py
#   RotatingLidarPhysX(prim_path="/World/Carter/chassis_link/lidar",
#                      translation=np.array([-0.06, 0, 0.38]))
# PhysX ray casting -> actual distance measurements, no renderer needed.
LIDAR_PATH = "/World/Carter/chassis_link/lidar"
LIDAR_MAX_RANGE = 50.0  # meters (warehouse max dimension is ~28m diagonal)

lidar = RotatingLidarPhysX(
    prim_path=LIDAR_PATH,
    name="carter_lidar",
    translation=np.array([-0.06, 0.0, 0.38]),
    fov=(360.0, 30.0),              # horizontal 360deg, vertical 30deg
    resolution=(1.0, 2.0),          # 360 cols x 16 rows -> lightweight scan
    valid_range=(0.4, LIDAR_MAX_RANGE),
)
# Request linear depth (METERS) in the acquired frame
lidar.add_linear_depth_data_to_frame()

log_print(f"LiDAR prim created at {LIDAR_PATH}")
log_print(f"LIDAR_RANGE: {lidar.get_valid_range()}")

simulation_app.update()
log_print("USD loaded")

# Set stage time codes per second
stage.SetTimeCodesPerSecond(60.0)

# ============================================================
# STEP 5: Create World and initialize simulation context
# ============================================================
my_world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=physics_dt)
log_print("World created")

# CRITICAL: Initialize simulation context - this is the key step from the official test
try:
    my_world.initialize_simulation_context()
    log_print("Simulation context initialized")
except Exception as e:
    log_print(f"initialize_simulation_context error: {e}")

# Wait for physics to be ready after initializing context
for _ in range(30):
    simulation_app.update()

# Play the world timeline (per official test)
my_world.play()
log_print("World played")

# Wait for physics to be ready after playing
for _ in range(30):
    simulation_app.update()

# ============================================================
# STEP 6: Create SingleArticulation Carter
# ============================================================
ar = SingleArticulation("/World/Carter")
log_print("SingleArticulation created at /World/Carter")

# Get the articulation view
articulation_view = ar._articulation_view

# Initialize the articulation view
try:
    articulation_view.initialize()
    log_print("Articulation view initialized successfully")
except Exception as e:
    log_print(f"Articulation view initialize info: {e}")

# ============================================================
# STEP 7: Initialize LiDAR sensor (subscribe to physics steps)
# ============================================================
lidar_initialized = True
try:
    lidar.initialize()
    log_print("LiDAR sensor initialized")
except Exception as e:
    lidar_initialized = False
    log_print(f"LiDAR initialize error: {e}")

# Place Carter in the central aisle, where walls (~2.8m) and
# racks (~3.8m) are well within the LiDAR range.
ar.set_world_pose(position=[0, -7, 0.1], orientation=[0, 0, 0, 1])
ar.set_world_velocity([0, 0, 0, 0, 0, 0])

# Wait for robot to settle
for _ in range(10):
    simulation_app.update()

start_pos, start_rot = ar.get_world_pose()
log_print(f"START_POSITION: {start_pos}")
log_print(f"LIDAR_MOUNT: {LIDAR_PATH} translation [-0.06, 0, 0.38]")

# ============================================================
# STEP 8: Run simulation for a fixed number of steps
# ============================================================
max_steps = 180  # 3 seconds at 60 Hz (short fixed run)

for step in range(max_steps):
    simulation_app.update()
    if step % 60 == 0:
        frame = lidar.get_current_frame()
        log_print(
            f"STEP {step}: PHYSICS_STEP: {frame.get('physics_step')}, "
            f"LINEAR_DEPTH_PRESENT: {frame.get('linear_depth') is not None}"
        )

log_print(f"Simulated {max_steps} steps")

# ============================================================
# STEP 9: Read LiDAR data and verify
# ============================================================
def to_numpy(data):
    """Convert acquired tensor (numpy/torch/warp) to a numpy array."""
    if data is None:
        return None
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    return np.asarray(data, dtype=np.float64)


frame = lidar.get_current_frame()
frame_keys = list(frame.keys())
log_print(f"FRAME_KEYS: {frame_keys}")

linear_depth = to_numpy(frame.get("linear_depth"))

lidar_data_available = linear_depth is not None and linear_depth.size > 0
log_print(f"LIDAR_INITIALIZED: {'YES' if lidar_initialized else 'NO'}")
log_print(f"LIDAR_DATA_AVAILABLE: {'YES' if lidar_data_available else 'NO'}")

min_distance = float("nan")
max_distance = float("nan")
valid_rays = 0

if lidar_data_available:
    log_print(f"SCAN_SHAPE: {linear_depth.shape}")
    log_print(f"SCAN_ROWS: {lidar.get_num_rows()}, SCAN_COLS: {lidar.get_num_cols()}")

    # Valid hit: distance strictly inside (0, max_range)
    # -> excludes no-return rays (0 or >= max_range) and un-ticked columns
    valid = (linear_depth > 0.0) & (linear_depth < LIDAR_MAX_RANGE)
    valid_rays = int(np.count_nonzero(valid))

    if valid_rays > 0:
        min_distance = float(np.min(linear_depth[valid]))
        max_distance = float(np.max(linear_depth[valid]))

    log_print(f"MIN_DISTANCE: {min_distance:.4f}")
    log_print(f"MAX_DISTANCE: {max_distance:.4f}")
    log_print(f"VALID_RAYS: {valid_rays}")
else:
    log_print("MIN_DISTANCE: nan")
    log_print("MAX_DISTANCE: nan")
    log_print("VALID_RAYS: 0")

# ============================================================
# STEP 10: Determine test result
# ============================================================
# Physically reasonable: 0 < min distance < sensor max range,
# and at least one obstacle (wall/rack) detected nearby (< 15m).
distance_ok = (
    valid_rays > 0
    and min_distance > 0.0
    and min_distance < LIDAR_MAX_RANGE
    and min_distance < 15.0
)

test_passed = lidar_initialized and lidar_data_available and distance_ok

if lidar_data_available:
    log_print(f"MIN_DISTANCE_REASONABLE: {distance_ok}")

log_print(f"LIDAR_TEST: {'PASS' if test_passed else 'FAIL'}")

# ============================================================
# STEP 11: Cleanup
# ============================================================
my_world.stop()
simulation_app.update()
simulation_app.close()

log_print("TEST_COMPLETED")

sys.exit(0 if test_passed else 1)
