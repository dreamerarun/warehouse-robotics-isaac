#!/usr/bin/env python3
"""Carter LiDAR Static-Obstacle Detection Test - Isaac Sim 5.1

Milestone 5: prove Carter's LiDAR detects a deliberately placed static
obstacle in FRONT of the robot (stationary perception test only).

Reuses the proven initialization pattern from test_carter_lidar.py:
- Same SimulationApp config (headless, iray)
- Same Carter asset (NovaCarter/nova_carter.usd)
- Same lightweight warehouse (floor + 4 walls + 6 static racks)
- Same physics ground plane + World init + SingleArticulation
- Same LiDAR mount (/World/Carter/chassis_link/lidar, translation
  [-0.06, 0, 0.38], identity rotation -> LiDAR frame == chassis frame)
- Same LiDAR config: 360 deg hFOV, 30 deg vFOV, 1 deg x 2 deg resolution,
  valid range (0.4, 50.0) meters. PhysX ray-cast (RotatingLidarPhysX).

Adds exactly ONE static obstacle box in front of Carter:
- Carter forward direction (nose) = chassis +X axis. With identity world
  orientation Carter's nose points toward world -X (verified from the
  navigation/warehouse runs: quintic planner initial motion was -X and the
  chassis yaw read ~180 deg). The box is therefore placed along the
  RUNTIME-COMPUTED forward vector (fallback: world -X), 4.0 m ahead:
  center (-4, -7), size (0.5, 2.0, 1.5) -> front face at 3.75 m from
  Carter center, ~3.81 m from the LiDAR.

LiDAR analysis:
- rotation_frequency = 0.0 (static full scan, exactly like the official
  test_lidar.py static test) so every frame holds all 360 columns and
  frame["azimuth"] aligns with depth axis 0.
- Azimuth convention (verified in test_lidar.py): column azimuth runs
  from -180 deg (col 0) to +180 deg, measured in the sensor frame from
  +X (the nose) - i.e. azimuth 0 deg IS Carter's forward direction.
- Forward cone = azimuth within +/-30 deg of 0 deg, elevation within
  +/-3 deg of the horizon (keeps floor rays out: floor at -3 deg is
  >= 7 m away while the obstacle is ~3.8 m).
- Known self-hits: returns <= 0.5 m are min-range clamps on Carter's own
  body; returns <= 2.0 m inside the cone are Carter body geometry (the
  nearest scene object forward is the obstacle at ~3.8 m).

Verification output (exact markers):
    LIDAR_INITIALIZED: YES
    LIDAR_DATA_AVAILABLE: YES
    FORWARD_OBSTACLE_DETECTED: YES
    FORWARD_MIN_DISTANCE: <meters>
    VALID_RAYS: <count>
    EXPECTED_OBSTACLE_DISTANCE: 3.8100 m
    LIDAR_OBSTACLE_TEST: PASS
"""

import sys
import os
import numpy as np

# Write start marker
with open("/tmp/carter_lidar_obstacle_results.txt", "w") as f:
    f.write("TEST_STARTED\n")


def log_print(msg):
    """Print to console and write to file"""
    print(msg)
    with open("/tmp/carter_lidar_obstacle_results.txt", "a") as f:
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
from pxr import Gf, Usd, PhysicsSchemaTools, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.prims import get_prim_at_path
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
# (identical to test_carter_lidar.py)
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
#
# rotation_frequency=0.0 -> static full 360 deg scan every physics step,
# exactly like the official test_lidar.py static test (rotation_rate=0.0).
# That guarantees frame["azimuth"] (one value per column) stays aligned
# with linear_depth axis 0, so forward rays can be selected by azimuth.
LIDAR_PATH = "/World/Carter/chassis_link/lidar"
LIDAR_MAX_RANGE = 50.0  # meters (warehouse max dimension is ~28m diagonal)

lidar = RotatingLidarPhysX(
    prim_path=LIDAR_PATH,
    name="carter_lidar",
    rotation_frequency=0.0,
    translation=np.array([-0.06, 0.0, 0.38]),
    fov=(360.0, 30.0),              # horizontal 360deg, vertical 30deg
    resolution=(1.0, 2.0),          # 360 cols x 16 rows -> lightweight scan
    valid_range=(0.4, LIDAR_MAX_RANGE),
)
# Request linear depth (METERS) + per-ray angles in the acquired frame
lidar.add_linear_depth_data_to_frame()
lidar.add_azimuth_data_to_frame()
lidar.add_zenith_data_to_frame()

log_print(f"LiDAR prim created at {LIDAR_PATH}")
log_print(f"LIDAR_RANGE: {lidar.get_valid_range()}")
log_print("LIDAR_ROTATION_FREQUENCY: 0.0 (static full scan)")

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

# Place Carter in the central aisle with identity orientation
ar.set_world_pose(position=[0, -7, 0.1], orientation=[0, 0, 0, 1])
ar.set_world_velocity([0, 0, 0, 0, 0, 0])

# Wait for robot to settle
for _ in range(10):
    simulation_app.update()

start_pos, start_rot = ar.get_world_pose()
log_print(f"START_POSITION: {start_pos}")
log_print(f"LIDAR_MOUNT: {LIDAR_PATH} translation [-0.06, 0, 0.38]")

# ============================================================
# STEP 7b: Compute Carter's forward direction, place ONE obstacle
# ============================================================
# Forward = chassis +X axis in world, read from the USD composed transform
# (fallback: world -X, verified from the navigation/warehouse runs where
# Carter with identity orientation drove along -X).
CARTEL_CENTER = np.array([0.0, -7.0])
OBSTACLE_DISTANCE_M = 4.0  # center of obstacle ahead of Carter
OBSTACLE_SIZE = (0.5, 2.0, 1.5)  # depth 0.5, width 2.0, height 1.5
OBSTACLE_CENTER_Z = 0.75
OBSTACLE_COLOR = (0.9, 0.3, 0.2)

try:
    chassis_prim = get_prim_at_path("/World/Carter/chassis_link")
    m_c2w = UsdGeom.Xformable(chassis_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    fwd3 = m_c2w.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    forward = np.array([float(fwd3[0]), float(fwd3[1])], dtype=np.float64)
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        raise RuntimeError("degenerate forward vector")
    forward = forward / norm
except Exception as e:
    log_print(f"FORWARD_VECTOR_FALLBACK ({e}): using (-1, 0)")
    forward = np.array([-1.0, 0.0])

log_print(f"FORWARD_WORLD_DIR: ({forward[0]:.4f}, {forward[1]:.4f})")

# Size the box so its 0.5m depth lies along the forward axis
if abs(forward[0]) >= abs(forward[1]):
    obstacle_size = OBSTACLE_SIZE
else:
    obstacle_size = (OBSTACLE_SIZE[1], OBSTACLE_SIZE[0], OBSTACLE_SIZE[2])

obstacle_pos = np.array(
    [
        CARTEL_CENTER[0] + forward[0] * OBSTACLE_DISTANCE_M,
        CARTEL_CENTER[1] + forward[1] * OBSTACLE_DISTANCE_M,
        OBSTACLE_CENTER_Z,
    ]
)
add_warehouse_box("/World/Obstacle", obstacle_pos, obstacle_size, OBSTACLE_COLOR)
log_print(
    f"OBSTACLE_POSITION: ({obstacle_pos[0]:.3f}, {obstacle_pos[1]:.3f}, "
    f"{obstacle_pos[2]:.3f}) size {obstacle_size}"
)

# Expected LiDAR -> obstacle front-face distance:
# box front face is (4.0 - 0.25) = 3.75m from Carter center along forward;
# the LiDAR sits 0.06m behind the chassis origin along forward
# (mount x = -0.06 in the chassis frame), so ~3.81m.
depth_along_fwd = obstacle_size[0] if abs(forward[0]) >= abs(forward[1]) else obstacle_size[1]
EXPECTED_OBSTACLE_DISTANCE = OBSTACLE_DISTANCE_M - depth_along_fwd / 2.0 + 0.06
log_print(f"EXPECTED_OBSTACLE_DISTANCE: {EXPECTED_OBSTACLE_DISTANCE:.4f} m")

# Let the new collider register with physics
for _ in range(10):
    simulation_app.update()

# ============================================================
# STEP 8: Run simulation, collect LiDAR frames
# ============================================================
max_steps = 180  # 3 seconds at 60 Hz (short fixed run)
collected = []  # (linear_depth, azimuth, zenith) per step (last frames kept)

for step in range(max_steps):
    simulation_app.update()
    frame = lidar.get_current_frame()
    if step % 60 == 0:
        log_print(
            f"STEP {step}: PHYSICS_STEP: {frame.get('physics_step')}, "
            f"LINEAR_DEPTH_PRESENT: {frame.get('linear_depth') is not None}"
        )
    # keep the last 30 frames for merging
    collected.append(
        (frame.get("linear_depth"), frame.get("azimuth"), frame.get("zenith"))
    )
    if len(collected) > 30:
        collected.pop(0)

log_print(f"Simulated {max_steps} steps")


# ============================================================
# STEP 9: Read LiDAR data and verify forward obstacle detection
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


def wrap_deg(a):
    """Wrap angles in degrees to [-180, 180)."""
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


frame = lidar.get_current_frame()
frame_keys = list(frame.keys())
log_print(f"FRAME_KEYS: {frame_keys}")

lidar_data_available = False
depth_acc = None
azimuth_ref = None
zenith_ref = None

for d_raw, a_raw, z_raw in collected:
    d = to_numpy(d_raw)
    if d is None or d.size == 0:
        continue
    if depth_acc is None:
        depth_acc = d.copy()
        azimuth_ref = to_numpy(a_raw)
        zenith_ref = to_numpy(z_raw)
    elif d.shape == depth_acc.shape:
        valid_new = (d > 0.0) & (d < LIDAR_MAX_RANGE)
        valid_old = (depth_acc > 0.0) & (depth_acc < LIDAR_MAX_RANGE)
        merged = np.where(valid_new & valid_old, np.minimum(depth_acc, d), depth_acc)
        merged = np.where(valid_new & ~valid_old, d, merged)
        depth_acc = merged
    # else: shape changed (should not happen) -> keep first frame

lidar_data_available = depth_acc is not None and depth_acc.size > 0
log_print(f"LIDAR_INITIALIZED: {'YES' if lidar_initialized else 'NO'}")
log_print(f"LIDAR_DATA_AVAILABLE: {'YES' if lidar_data_available else 'NO'}")

forward_min_distance = float("nan")
valid_rays = 0
forward_rays = 0
forward_valid_rays = 0
forward_min_valid_rays = 0
near_expected_rays = 0
self_hit_rays = 0
vehicle_body_rays = 0

if lidar_data_available:
    linear_depth = depth_acc
    log_print(f"SCAN_SHAPE: {linear_depth.shape}")
    log_print(f"SCAN_ROWS: {lidar.get_num_rows()}, SCAN_COLS: {lidar.get_num_cols()}")
    log_print(f"AZIMUTH_SHAPE: {None if azimuth_ref is None else azimuth_ref.shape}")
    log_print(f"ZENITH_SHAPE: {None if zenith_ref is None else zenith_ref.shape}")

    # Valid hit: distance strictly inside (0, max_range)
    valid = (linear_depth > 0.0) & (linear_depth < LIDAR_MAX_RANGE)
    valid_rays = int(np.count_nonzero(valid))

    # ---- azimuth / zenith vectors (aligned with depth axes) ----
    n_az, n_ze = linear_depth.shape
    if azimuth_ref is not None and azimuth_ref.ndim == 2:
        azimuth_ref = azimuth_ref[:, 0]
    if azimuth_ref is None or azimuth_ref.size != n_az:
        log_print(
            f"AZIMUTH_ALIGNMENT: using column-index convention "
            f"(az size {None if azimuth_ref is None else azimuth_ref.size} vs {n_az})"
        )
        azimuth_deg = -180.0 + np.arange(n_az) * 1.0  # hFOV 360, hres 1
    else:
        azimuth_deg = azimuth_ref.copy()
        if np.nanmax(np.abs(azimuth_deg)) <= 6.5:
            azimuth_deg = np.degrees(azimuth_deg)  # stored in radians
        log_print(
            f"AZIMUTH_ALIGNMENT: aligned, range "
            f"[{np.nanmin(azimuth_deg):.2f}, {np.nanmax(azimuth_deg):.2f}] deg"
        )

    if zenith_ref is not None and zenith_ref.ndim == 2:
        zenith_ref = zenith_ref[0, :]
    if zenith_ref is None or zenith_ref.size != n_ze:
        log_print(
            f"ZENITH_ALIGNMENT: using row-index convention "
            f"(ze size {None if zenith_ref is None else zenith_ref.size} vs {n_ze})"
        )
        zenith_deg = 90.0 - 15.0 + np.arange(n_ze) * 2.0  # vFOV 30, vres 2
    else:
        zenith_deg = zenith_ref.copy()
        if np.nanmax(np.abs(zenith_deg)) <= 6.5:
            zenith_deg = np.degrees(zenith_deg)  # stored in radians

    # elevation above horizon (deg): zenith measured from +Z -> elev = 90 - zen
    if np.nanmedian(zenith_deg) > 45.0:
        elevation_deg = 90.0 - zenith_deg
    else:
        elevation_deg = zenith_deg
    log_print(
        f"ELEVATION_RANGE: [{np.nanmin(elevation_deg):.2f}, "
        f"{np.nanmax(elevation_deg):.2f}] deg"
    )

    # ---- forward cone (Carter's nose = sensor +X = azimuth 0 deg) ----
    FORWARD_CONE_DEG = 30.0
    ELEV_CONE_DEG = 3.0
    SELF_HIT_MAX = 0.5    # min-range clamp on Carter's own body
    VEHICLE_BODY_MAX = 2.0  # nearest scene object forward is the obstacle (~3.8m)

    fwd_mask = (np.abs(wrap_deg(azimuth_deg)) <= FORWARD_CONE_DEG)[:, None] & (
        np.abs(elevation_deg) <= ELEV_CONE_DEG
    )[None, :]

    forward_rays = int(np.count_nonzero(fwd_mask))
    fwd_valid = fwd_mask & valid
    forward_valid_rays = int(np.count_nonzero(fwd_valid))

    self_hit_rays = int(np.count_nonzero(fwd_valid & (linear_depth <= SELF_HIT_MAX)))
    vehicle_body_rays = int(
        np.count_nonzero(
            fwd_valid
            & (linear_depth > SELF_HIT_MAX)
            & (linear_depth <= VEHICLE_BODY_MAX)
        )
    )
    non_body = fwd_valid & (linear_depth > VEHICLE_BODY_MAX)
    if np.any(non_body):
        forward_min_distance = float(np.min(linear_depth[non_body]))
        forward_min_valid_rays = int(np.count_nonzero(non_body))
        near_expected_rays = int(
            np.count_nonzero(
                non_body
                & (np.abs(linear_depth - EXPECTED_OBSTACLE_DISTANCE) <= 0.5)
            )
        )

    log_print(f"VALID_RAYS: {valid_rays}")
    log_print(f"FORWARD_RAYS: {forward_rays}")
    log_print(f"FORWARD_VALID_RAYS: {forward_valid_rays}")
    log_print(f"FORWARD_BODY_RAYS (<=0.5m self-hit): {self_hit_rays}")
    log_print(f"FORWARD_BODY_RAYS (<=2.0m vehicle): {vehicle_body_rays}")
    log_print(f"FORWARD_MIN_DISTANCE: {forward_min_distance:.4f}")
    log_print(f"FORWARD_RAYS_NEAR_EXPECTED: {near_expected_rays}")
else:
    log_print("FORWARD_MIN_DISTANCE: nan")
    log_print("VALID_RAYS: 0")

# ============================================================
# STEP 10: Determine test result
# ============================================================
# The obstacle is detected when a cluster of forward, non-body rays lands
# at the expected physical distance (expected value comes purely from the
# chosen placement: front face 3.75m from Carter center + 0.06m LiDAR
# offset behind the chassis origin).
obstacle_detected = near_expected_rays >= 5
distance_ok = (
    forward_min_valid_rays > 0
    and np.isfinite(forward_min_distance)
    and forward_min_distance > 0.4            # above min sensor range / self-hit
    and abs(forward_min_distance - EXPECTED_OBSTACLE_DISTANCE) <= 1.0  # physically reasonable
)

log_print(f"FORWARD_OBSTACLE_DETECTED: {'YES' if obstacle_detected else 'NO'}")

test_passed = (
    lidar_initialized
    and lidar_data_available
    and valid_rays > 0
    and obstacle_detected
    and distance_ok
)

log_print(f"LIDAR_OBSTACLE_TEST: {'PASS' if test_passed else 'FAIL'}")

# ============================================================
# STEP 11: Cleanup
# ============================================================
my_world.stop()
simulation_app.update()
simulation_app.close()

log_print("TEST_COMPLETED")

sys.exit(0 if test_passed else 1)
