#!/usr/bin/env python3
"""Carter LiDAR Reactive Obstacle-Avoidance Test - Isaac Sim 5.1

Milestone 6: closed-loop perception -> action (STOP) behavior.

  Carter navigation -> LiDAR -> forward obstacle detection
      -> safety threshold (1.5 m) -> STOP Carter (and stay stopped)

This is NOT path planning / avoidance: Carter never goes around the
obstacle. It simply proves:

  LiDAR detects obstacle -> threshold crossed -> velocity command
  becomes zero -> Carter stops safely.

Proven pipelines reused (no existing file modified):
- Initialization: identical to test_carter_lidar.py /
  test_carter_lidar_obstacle.py (SimulationApp config, NovaCarter USD,
  lightweight warehouse, World + initialize_simulation_context,
  SingleArticulation, RotatingLidarPhysX with rotation_frequency=0.0).
- Navigation/control: identical OmniGraph to test_carter_navigation.py
  (QuinticPathPlanner -> CheckGoal2D -> StanleyControlPID ->
   DifferentialController -> ArticulationController + computeOdom).
- LiDAR config: 360 deg hFOV, 30 deg vFOV, 1 deg x 2 deg resolution,
  0.4-50 m valid range; forward cone analysis exactly as Milestone 5
  (azimuth +/-30 deg around 0 deg = the nose, elevation +/-3 deg,
  min-range self-hits <= 0.5 m excluded).
- Obstacle placement: runtime-computed forward vector from Milestone 5
  (nose = world -X with identity orientation -> obstacle at (-4, -7)).

Behavior under test:
  forward distance > 1.5 m  -> keep driving toward the goal
  forward distance <= 1.5 m -> disconnect Stanley velocity commands,
                               command zero velocity, remain stopped
                               for the remainder of the test.
"""

import sys
import os
import numpy as np

# Write start marker
with open("/tmp/carter_lidar_avoidance_results.txt", "w") as f:
    f.write("TEST_STARTED\n")


def log_print(msg):
    """Print to console and write to file"""
    print(msg)
    with open("/tmp/carter_lidar_avoidance_results.txt", "a") as f:
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
import omni.graph.core as og
import usdrt.Sdf
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
# (identical to test_carter_lidar.py / test_carter_lidar_obstacle.py)
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
# Follows the official Isaac Sim 5.1 example exactly
# (RotatingLidarPhysX on /World/Carter/chassis_link/lidar,
#  translation [-0.06, 0, 0.38] -> LiDAR frame == chassis frame).
# rotation_frequency=0.0 -> static full 360 deg scan every physics step,
# exactly like the official test_lidar.py static test, so frame["azimuth"]
# stays aligned with linear_depth axis 0 while Carter moves.
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
# Use /World/Carter path (our USD is loaded here)
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

# Place Carter in the central aisle with identity orientation, facing -X
# (nose = chassis +X = world -X with identity world orientation, verified
#  in milestones 1-5). The obstacle and goal are placed along -X.
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
# (fallback: world -X, verified in milestones 1-5).
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

# Half extents of the obstacle in XY (for the separation safety check)
if abs(forward[0]) >= abs(forward[1]):
    OBS_HALF = np.array([obstacle_size[0] / 2.0, obstacle_size[1] / 2.0])
else:
    OBS_HALF = np.array([obstacle_size[1] / 2.0, obstacle_size[0] / 2.0])

# Expected LiDAR -> obstacle front-face distance (same as Milestone 5):
# front face 3.75m from Carter center along forward + 0.06m LiDAR offset
# behind the chassis origin -> 3.81m.
depth_along_fwd = obstacle_size[0] if abs(forward[0]) >= abs(forward[1]) else obstacle_size[1]
EXPECTED_OBSTACLE_DISTANCE = OBSTACLE_DISTANCE_M - depth_along_fwd / 2.0 + 0.06
log_print(f"EXPECTED_OBSTACLE_DISTANCE: {EXPECTED_OBSTACLE_DISTANCE:.4f} m")

# Let the new collider register with physics
for _ in range(10):
    simulation_app.update()

# ============================================================
# STEP 8: Setup OmniGraph with the proven navigation pipeline
# ============================================================
# Identical to test_carter_navigation.py (verified), only the goal moves:
# a simple goal BEYOND the obstacle, on the straight line Carter faces.
GOAL_POS = (-8.0, -7.0, 0.0)  # obstacle at x=-4 (rear face -4.25), wall at -10
SAFETY_DISTANCE = 1.5  # meters: forward threshold that commands STOP

graph_path = "/ActionGraph"

# Setup base robot omnigraph (DifferentialController, ArticulationController, computeOdom)
keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": graph_path, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("DifferentialController", "isaacsim.robot.wheeled_robots.DifferentialController"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ("computeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
            # Navigation pipeline nodes
            ("GetPrimLocalToWorldTransform", "omni.graph.nodes.GetPrimLocalToWorldTransform"),
            ("GetRotationQuaternion", "omni.graph.nodes.GetMatrix4Quaternion"),
            ("GetTranslation", "omni.graph.nodes.GetMatrix4Translation"),
            ("QuinticPathPlanner", "isaacsim.robot.wheeled_robots.QuinticPathPlanner"),
            ("CheckGoal2D", "isaacsim.robot.wheeled_robots.CheckGoal2D"),
            ("StanleyControlPID", "isaacsim.robot.wheeled_robots.StanleyControlPID"),
        ],
        keys.CONNECT: [
            # Base robot connections
            ("OnPlaybackTick.outputs:tick", "DifferentialController.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "computeOdom.inputs:execIn"),
            ("DifferentialController.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),

            # Navigation pipeline connections
            ("OnPlaybackTick.outputs:tick", "QuinticPathPlanner.inputs:execIn"),
            ("QuinticPathPlanner.outputs:execOut", "CheckGoal2D.inputs:execIn"),
            ("CheckGoal2D.outputs:execOut", "StanleyControlPID.inputs:execIn"),
            ("GetPrimLocalToWorldTransform.outputs:localToWorldTransform", "GetRotationQuaternion.inputs:matrix"),
            ("GetPrimLocalToWorldTransform.outputs:localToWorldTransform", "GetTranslation.inputs:matrix"),
            ("GetRotationQuaternion.outputs:quaternion", "QuinticPathPlanner.inputs:currentOrientation"),
            ("GetRotationQuaternion.outputs:quaternion", "CheckGoal2D.inputs:currentOrientation"),
            ("GetRotationQuaternion.outputs:quaternion", "StanleyControlPID.inputs:currentOrientation"),
            ("GetTranslation.outputs:translation", "QuinticPathPlanner.inputs:currentPosition"),
            ("GetTranslation.outputs:translation", "CheckGoal2D.inputs:currentPosition"),
            ("GetTranslation.outputs:translation", "StanleyControlPID.inputs:currentPosition"),
            ("QuinticPathPlanner.outputs:pathArrays", "StanleyControlPID.inputs:pathArrays"),
            ("QuinticPathPlanner.outputs:target", "CheckGoal2D.inputs:target"),
            ("QuinticPathPlanner.outputs:target", "StanleyControlPID.inputs:target"),
            ("QuinticPathPlanner.outputs:targetChanged", "CheckGoal2D.inputs:targetChanged"),
            ("QuinticPathPlanner.outputs:targetChanged", "StanleyControlPID.inputs:targetChanged"),
            ("CheckGoal2D.outputs:reachedGoal", "StanleyControlPID.inputs:reachedGoal"),
            ("StanleyControlPID.outputs:angularVelocity", graph_path + "/DifferentialController.inputs:angularVelocity"),
            ("StanleyControlPID.outputs:linearVelocity", graph_path + "/DifferentialController.inputs:linearVelocity"),
            (graph_path + "/computeOdom.outputs:linearVelocity", "StanleyControlPID.inputs:currentSpeed"),
        ],
        keys.SET_VALUES: [
            # Target position: goal BEYOND the obstacle.
            # Target orientation is re-set below to Carter's actual runtime
            # chassis orientation so start yaw == goal yaw -> the planned
            # path is a clean straight line toward the goal.
            ("QuinticPathPlanner.inputs:targetPosition", GOAL_POS),
            ("QuinticPathPlanner.inputs:targetOrientation", (0, 0, 0, 1)),
            # Transform node settings
            ("GetPrimLocalToWorldTransform.inputs:usePath", False),
            # Chassis prim for odometry and transform
            (graph_path + "/computeOdom.inputs:chassisPrim", [usdrt.Sdf.Path("/World/Carter/chassis_link")]),
            ("GetPrimLocalToWorldTransform.inputs:prim", [usdrt.Sdf.Path("/World/Carter/chassis_link")]),
            # Differential controller settings (from official example)
            ("DifferentialController.inputs:wheelRadius", 0.14),
            ("DifferentialController.inputs:wheelDistance", 0.4132),
            # Articulation controller settings
            ("ArticulationController.inputs:jointNames", ["joint_wheel_left", "joint_wheel_right"]),
            ("ArticulationController.inputs:robotPath", "/World/Carter"),
        ],
    },
)

log_print("OmniGraph created successfully")

# Reset controller (proven pattern from test_carter_navigation.py)
og.Controller.attribute(graph_path + "/DifferentialController.inputs:linearVelocity").set(0)
og.Controller.attribute(graph_path + "/DifferentialController.inputs:angularVelocity").set(0)

# Let the first plan run from the correct start pose
for _ in range(10):
    simulation_app.update()

# Re-target the planner to Carter's ACTUAL chassis orientation (read from
# the graph output - same proven og attribute get/set pattern the nav test
# uses for reachedGoal). This makes goal yaw == start yaw, so the planned
# path keeps a constant heading straight toward the goal.
chassis_quat = tuple(
    og.Controller.attribute(graph_path + "/GetRotationQuaternion.outputs:quaternion").get()
)
log_print(f"CHASSIS_ORIENTATION: {chassis_quat}")
og.Controller.attribute(graph_path + "/QuinticPathPlanner.inputs:targetOrientation").set(chassis_quat)

for _ in range(10):
    simulation_app.update()

# Get start position of the measured run
start_pos, start_rot = ar.get_world_pose()
log_print(f"INITIAL_POSITION: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
log_print(f"GOAL_POSITION: ({GOAL_POS[0]:.3f}, {GOAL_POS[1]:.3f}, {GOAL_POS[2]:.3f})")
log_print(f"SAFETY_DISTANCE: {SAFETY_DISTANCE:.4f}")

# ============================================================
# STEP 9: Closed-loop run: navigate, read LiDAR, STOP on threshold
# ============================================================
# Forward-cone constants (verified in Milestone 5): nose = sensor +X =
# azimuth 0 deg; floor rays at elevation -3 deg hit at >= 7m so they never
# beat the obstacle (3.8m -> 1.5m) inside the cone.
FORWARD_CONE_DEG = 30.0
ELEV_CONE_DEG = 3.0
SELF_HIT_MAX = 0.5  # min-range clamp on Carter's own body only
# NOTE: unlike milestone 5 there is NO 2.0m "vehicle body" exclusion here -
# the obstacle legitimately enters that range during the approach.

MAX_STEPS = 900        # 15 s hard cap at 60 Hz
HOLD_AFTER_TRIGGER = 240  # 4 s of remaining-stopped evidence after STOP


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


def build_forward_masks(frame, depth_shape):
    """Build the forward-cone mask from the first frame's azimuth/zenith
    (same convention handling as Milestone 5's verified analysis)."""
    n_az, n_ze = depth_shape
    azimuth_ref = to_numpy(frame.get("azimuth"))
    zenith_ref = to_numpy(frame.get("zenith"))

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
            zenith_deg = np.degrees(zenith_deg)

    # elevation above horizon (deg): zenith measured from +Z -> elev = 90 - zen
    if np.nanmedian(zenith_deg) > 45.0:
        elevation_deg = 90.0 - zenith_deg
    else:
        elevation_deg = zenith_deg
    log_print(
        f"ELEVATION_RANGE: [{np.nanmin(elevation_deg):.2f}, "
        f"{np.nanmax(elevation_deg):.2f}] deg"
    )

    mask = (np.abs(wrap_deg(azimuth_deg)) <= FORWARD_CONE_DEG)[:, None] & (
        np.abs(elevation_deg) <= ELEV_CONE_DEG
    )[None, :]
    return mask


stop_triggered = False
stop_step = -1
stop_distance = float("nan")
min_forward_distance = float("inf")
min_clearance_center = float("inf")
lidar_data_available = False
obstacle_detected = False
fwd_mask = None
frame_keys_logged = False
first_frame_analyzed = False
first_frame_valid_rays = 0
first_frame_forward_rays = 0
near_expected_rays = 0

position_history = []
speeds = []
prev_pos = np.array(start_pos, dtype=np.float64)

log_print(f"Running closed-loop avoidance (max {MAX_STEPS} steps, "
          f"hold {HOLD_AFTER_TRIGGER} steps after trigger)")

for step in range(MAX_STEPS):
    simulation_app.update()

    # --- Carter state ---
    pos, rot = ar.get_world_pose()
    pos = np.array(pos, dtype=np.float64)
    speed = float(np.linalg.norm(pos - prev_pos)) / physics_dt
    prev_pos = pos.copy()
    speeds.append(speed)
    position_history.append(pos.copy())

    # --- continuous LiDAR read (every physics step) ---
    frame = lidar.get_current_frame()
    d = to_numpy(frame.get("linear_depth")) if frame else None
    forward_dist = None

    if d is not None and d.size > 0:
        lidar_data_available = True
        if not frame_keys_logged:
            log_print(f"FRAME_KEYS: {list(frame.keys())}")
            frame_keys_logged = True
        if fwd_mask is None:
            fwd_mask = build_forward_masks(frame, d.shape)
            log_print(f"SCAN_SHAPE: {d.shape}")
            log_print(f"FORWARD_RAYS: {int(np.count_nonzero(fwd_mask))}")

        valid = (d > 0.0) & (d < LIDAR_MAX_RANGE)
        sel = fwd_mask & valid & (d > SELF_HIT_MAX)
        if np.any(sel):
            forward_dist = float(np.min(d[sel]))
            min_forward_distance = min(min_forward_distance, forward_dist)

            if not first_frame_analyzed:
                # Milestone-5 proven detection: a cluster of forward rays at
                # the physically expected obstacle distance.
                first_frame_analyzed = True
                first_frame_valid_rays = int(np.count_nonzero(valid))
                first_frame_forward_rays = int(np.count_nonzero(sel))
                near_expected_rays = int(
                    np.count_nonzero(
                        np.abs(d[sel] - EXPECTED_OBSTACLE_DISTANCE) <= 0.5
                    )
                )
                obstacle_detected = near_expected_rays >= 5
                log_print(f"VALID_RAYS: {first_frame_valid_rays}")
                log_print(f"FORWARD_MIN_DISTANCE_FIRST_FRAME: {forward_dist:.4f}")
                log_print(f"FORWARD_RAYS_NEAR_EXPECTED: {near_expected_rays}")

    # --- geometric separation from the obstacle (safety check) ---
    dx = max(abs(pos[0] - obstacle_pos[0]) - OBS_HALF[0], 0.0)
    dy = max(abs(pos[1] - obstacle_pos[1]) - OBS_HALF[1], 0.0)
    clearance = float(np.hypot(dx, dy))
    min_clearance_center = min(min_clearance_center, clearance)

    if step % 50 == 0:
        d2goal = float(np.linalg.norm(pos[:2] - np.array(GOAL_POS[:2])))
        fd_str = f"{forward_dist:.4f}" if forward_dist is not None else "nan"
        log_print(
            f"STEP {step}: POSITION: ({pos[0]:.3f}, {pos[1]:.3f}), "
            f"SPEED: {speed:.3f}, FORWARD_DIST: {fd_str}, "
            f"DIST_TO_GOAL: {d2goal:.3f}, STOPPED: {stop_triggered}"
        )

    # --- safety threshold: forward obstacle distance <= SAFETY_DISTANCE ---
    if (
        not stop_triggered
        and forward_dist is not None
        and forward_dist <= SAFETY_DISTANCE
    ):
        # STOP: disconnect the Stanley velocity commands from the proven
        # DifferentialController and command zero velocity. Verified
        # OmniGraph keys.DISCONNECT API (same spec strings as the CONNECT
        # above). Nothing re-connects them -> Carter remains stopped.
        og.Controller.edit(
            graph_path,
            {
                keys.DISCONNECT: [
                    (graph_path + "/StanleyControlPID.outputs:angularVelocity",
                     graph_path + "/DifferentialController.inputs:angularVelocity"),
                    (graph_path + "/StanleyControlPID.outputs:linearVelocity",
                     graph_path + "/DifferentialController.inputs:linearVelocity"),
                ],
            },
        )
        og.Controller.attribute(
            graph_path + "/DifferentialController.inputs:linearVelocity"
        ).set(0)
        og.Controller.attribute(
            graph_path + "/DifferentialController.inputs:angularVelocity"
        ).set(0)

        stop_triggered = True
        stop_step = step
        stop_distance = forward_dist
        log_print(
            f"STOP_TRIGGERED_STEP: {step}, STOP_DISTANCE: {stop_distance:.4f}, "
            f"POSITION_AT_STOP: ({pos[0]:.3f}, {pos[1]:.3f}), "
            f"SPEED_AT_STOP: {speed:.3f}"
        )

    # --- once stopped: hold for the remainder of the test ---
    if stop_triggered and (step - stop_step) >= HOLD_AFTER_TRIGGER:
        log_print(f"Stop hold complete at step {step}")
        break

log_print(f"Simulated {len(position_history)} steps")

# ============================================================
# STEP 10: Records and verification
# ============================================================
final_pos, final_rot = ar.get_world_pose()
final_pos = np.array(final_pos, dtype=np.float64)

# Did Carter move toward the goal?
disp = float(np.linalg.norm(final_pos[:2] - np.array(start_pos[:2])))
toward_goal = float(
    np.dot(final_pos[:2] - np.array(start_pos[:2]), np.array(GOAL_POS[:2]) - np.array(start_pos[:2]))
)
carter_moved = disp > 0.3 and toward_goal > 0

# Final speed (mean of last 10 steps, m/s)
final_speed = float(np.mean(speeds[-10:])) if speeds else float("nan")

# Did Carter remain stopped after the trigger? (drift over the last 60 steps)
hist = np.array(position_history)
if len(hist) >= 61:
    recent_drift = float(np.linalg.norm(hist[-1, :2] - hist[-61, :2]))
else:
    recent_drift = float(np.linalg.norm(hist[-1, :2] - hist[0, :2]))
carter_stopped = stop_triggered and final_speed < 0.05 and recent_drift < 0.05

# Safety check (req 19): obstacle must stay physically separated from
# Carter at ALL times - both geometrically and via the LiDAR reading.
MIN_FORWARD_ALLOWED = 1.0     # lidar-based clearance floor
MIN_CLEARANCE_CENTER = 0.7    # Carter center -> obstacle box face (nose ~0.55m)
separation_ok = (
    lidar_data_available
    and min_forward_distance >= MIN_FORWARD_ALLOWED
    and min_clearance_center >= MIN_CLEARANCE_CENTER
)

log_print("---- RECORDS ----")
log_print(f"INITIAL_POSITION: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
log_print(f"GOAL_POSITION: ({GOAL_POS[0]:.3f}, {GOAL_POS[1]:.3f}, {GOAL_POS[2]:.3f})")
log_print(f"MIN_FORWARD_DISTANCE: {min_forward_distance:.4f}")
log_print(f"STOP_STEP: {stop_step}")
log_print(f"MOVED_DISTANCE: {disp:.4f}")
log_print(f"MIN_CLEARANCE_CENTER: {min_clearance_center:.4f}")
log_print(f"DRIFT_AFTER_STOP: {recent_drift:.4f}")
log_print(
    f"SEPARATION_SAFETY: {'PASS' if separation_ok else 'FAIL'} "
    f"(min forward {min_forward_distance:.4f} >= {MIN_FORWARD_ALLOWED}, "
    f"min center clearance {min_clearance_center:.4f} >= {MIN_CLEARANCE_CENTER})"
)

log_print("---- VERIFICATION MARKERS ----")
log_print(f"LIDAR_INITIALIZED: {'YES' if lidar_initialized else 'NO'}")
log_print(f"LIDAR_DATA_AVAILABLE: {'YES' if lidar_data_available else 'NO'}")
log_print(f"CARTER_MOVED: {'YES' if carter_moved else 'NO'}")
log_print(f"OBSTACLE_DETECTED: {'YES' if obstacle_detected else 'NO'}")
log_print(f"STOP_TRIGGERED: {'YES' if stop_triggered else 'NO'}")
log_print(f"STOP_DISTANCE: {stop_distance:.4f}")
log_print(
    f"FINAL_POSITION: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})"
)
log_print(f"FINAL_SPEED: {final_speed:.4f}")
log_print(f"CARTER_STOPPED: {'YES' if carter_stopped else 'NO'}")

# ============================================================
# STEP 11: Determine test result
# ============================================================
test_passed = (
    lidar_initialized
    and lidar_data_available
    and carter_moved           # Carter actually moved toward the goal
    and obstacle_detected      # forward obstacle detected (Milestone 5 logic)
    and stop_triggered         # stop commanded when distance <= 1.5 m
    and final_speed < 0.05     # final speed effectively zero
    and carter_stopped         # remained stopped after the trigger
    and separation_ok          # never dangerously close / no collision
)

log_print(f"OBSTACLE_AVOIDANCE_TEST: {'PASS' if test_passed else 'FAIL'}")

# ============================================================
# STEP 12: Cleanup
# ============================================================
my_world.stop()
simulation_app.update()
simulation_app.close()

log_print("TEST_COMPLETED")

sys.exit(0 if test_passed else 1)
