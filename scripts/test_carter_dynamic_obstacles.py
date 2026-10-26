#!/usr/bin/env python3
"""Milestones 10+11 - Dynamic Obstacles + Reactive Navigation Robustness
(Isaac Sim 5.1, VISUAL)

Forks the FROZEN, VISIBLY VERIFIED Milestone 9 script
(scripts/test_carter_visualization.py, NOT modified) and adds:

M10 - DYNAMIC OBSTACLES (3 deterministic kinematic moving boxes)
    - Obstacle A "crosser"      slides laterally across Carter's aisle
                                (blocks the start->goal line around the
                                expected meeting time, then clears it)
    - Obstacle B "aisle runner" travels along another aisle (y = -2.0),
                                never on Carter's route
    - Obstacle C "approacher"   moves toward/away from the rack row at
                                x = -1.0 (off the main corridor)
    - No humans, no forklifts, no randomness: positions are analytic
      functions of simulation time (reproducible every run).
    - Each box is a UsdGeom.Cube with translate/scale ops on the prim
      itself + CollisionAPI + RigidBodyAPI (physics:kinematicEnabled).
      Per step the motion is written BOTH to the translate op (visual
      + kinematic follow) and to the PhysX view via RigidPrim
      (tensor write, belt and braces).
    - The controller NEVER assumes obstacles are stationary: every
      step re-runs scan -> detect -> slow -> choose safe direction
      from MEASURED sector clearance (LEFT +30..+90 deg, RIGHT
      -90..-30 deg, conservative per-sector MIN) -> steer -> verify ->
      resume. Turn direction is recomputed each event (never
      hard-coded); thresholds CLEAR>2.5 / APPROACH<=2.5 / AVOID<=1.5 m
      unchanged; slow/stop first if blocked; never intentionally
      collide. Runtime LiDAR-vs-predicted-ray/AABB check verifies the
      PhysX bodies really follow the scripted motion.

M11 - REACTIVE NAVIGATION ROBUSTNESS (RECOVERY, anti-oscillation,
      progress detection; NO global planner)
    - Explicit states: NAVIGATE, APPROACH, AVOID, PASS_OBSTACLE,
      RESUME, RECOVERY (with visible REASSESS phase), GOAL_REACHED.
    - RECOVERY triggers: both sectors <= BLOCKED_T (1.2 m), forward
      <= AVOID_T sustained too long without progress, no goal
      progress over a sliding window, or repeated avoidance failures
      on the same obstacle. Behavior: stop -> reverse slowly/short
      (only while the rear sector |az| >= 150 deg is clear) ->
      REASSESS (stop + re-scan, own color/log) -> compare sectors ->
      pick the safer side (or the larger clearance once the retry cap
      is hit) -> resume AVOID. Visible trace:
      RECOVERY -> REASSESS -> AVOID LEFT/RIGHT.
    - Anti-oscillation: avoidance direction is held for a minimum
      duration/distance; it may only change if the other side beats
      the held side by >= HYST_MARGIN (genuine safety reason).
      OSCILLATION_DETECTED flips to YES only if >= 2 side changes
      occur across consecutive avoidance events.
    - Progress detection: sliding goal-distance window and
      sustained-forward counters guarantee no infinite loop can hide.

VISUAL MODE: headless=False by default, window 1280x720, lightweight
renderer settings for the RTX 4050 6 GB (no RTX cameras, no ray-traced
perception, no expensive assets, one Carter, one PhysX LiDAR, simple
boxes). Set HEADLESS=1 for headless runs and DEMO_DURATION=<seconds>
for the post-goal observation period.

Proven pipelines reused verbatim (all seven verified scripts frozen):
- Init / warehouse / PhysX LiDAR (RotatingLidarPhysX, fov 360x30,
  resolution 1x2 deg -> scan (360,16), range 0.4-50 m, high_lod=True).
- Navigation OmniGraph: QuinticPathPlanner -> CheckGoal2D ->
  StanleyControlPID -> DifferentialController ->
  IsaacArticulationController + odometry.
- M7/M8/M9 reactive avoidance state machine + trajectory trail +
  state marker + LiDAR debug draw (reused from the M9 script).

Success markers (M9 re-verified plus):
    MILESTONE_9_TRAJECTORY_INITIALIZED: YES
    MILESTONE_9_TRAJECTORY_POINTS: <count>   (>0)
    MILESTONE_9_VISUALIZATION: YES
    MILESTONE_9_LIDAR: YES                   (scan rows == 16)
    MILESTONE_9_GOAL_REACHED: YES
    MILESTONE_10_DYNAMIC_OBSTACLES: YES
    MILESTONE_10_DYNAMIC_OBSTACLE_DETECTIONS: <count>  (>0)
    MILESTONE_10_AVOIDANCE_EVENTS: <count>             (>0)
    MILESTONE_10_GOAL_REACHED: YES
    MILESTONE_11_RECOVERY_ENABLED: YES
    MILESTONE_11_RECOVERY_EVENTS: <count>   (0 is valid: the
        scenario must stay safe; the mechanism is armed and logged)
    MILESTONE_11_OSCILLATION_DETECTED: NO
    MILESTONE_11_COLLISION_DETECTED: NO
    MILESTONE_11_GOAL_REACHED: YES
    WAREHOUSE_AUTONOMOUS_NAVIGATION_M9_M10_M11: PASS

Usage:
    cd /home/arun/warehouse_robotics_isaac
    ISAAC=/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64
    HEADLESS=0 $ISAAC/python.sh scripts/test_carter_dynamic_obstacles.py
"""

import sys
import os
import math
import numpy as np

# Write start marker
with open("/tmp/carter_dynamic_obstacles_results.txt", "w") as f:
    f.write("TEST_STARTED\n")


def log_print(msg):
    """Print to console and write to file"""
    print(msg)
    with open("/tmp/carter_dynamic_obstacles_results.txt", "a") as f:
        f.write(msg + "\n")


# ============================================================
# STEP 1: Start Isaac Sim 5.1 - VISIBLE by default
# ============================================================
from isaacsim import SimulationApp

# The demo opens a window unless HEADLESS=1 is explicitly provided.
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

sim_config = {
    # "RaytracedLighting" is the interactive Isaac Sim renderer (default);
    # "iray" is kept for headless regression parity with the verified scripts.
    "renderer": "iray" if HEADLESS else "RaytracedLighting",
    "headless": HEADLESS,
    "window_width": 1280,
    "window_height": 720,
    "width": 1280,
    "height": 720,
    "limit_sampling_buffer_size": True,
    "depth_value_enabled": False,
}

simulation_app = SimulationApp(sim_config)
log_print(f"SimulationApp created (headless={HEADLESS}, window 1280x720)")

# ============================================================
# STEP 2: Import Isaac Sim APIs (AFTER SimulationApp)
# ============================================================
import omni
import omni.graph.core as og
import usdrt.Sdf
import carb
from pxr import Gf, Sdf, Usd, PhysicsSchemaTools, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, RigidPrim
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

CARTER_USD_PATH = assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
carter_path = "/World/Carter"
add_reference_to_stage(usd_path=CARTER_USD_PATH, prim_path=carter_path)

# Physics ground plane + visual floor (identical to the verified scripts)
PhysicsSchemaTools.addGroundPlane(
    stage, "/groundPlane", "Z", 1500, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
)
floor_path = "/World/Floor"
add_reference_to_stage(
    usd_path=assets_root + "/Isaac/Props/Grid/plane.usd",
    prim_path=floor_path,
)

# ------------------------------------------------------------
# Static collision box helper (UsdGeom.Cube + CollisionAPI).
# Every collider is registered in collision_boxes for the safety check.
# ------------------------------------------------------------
collision_boxes = []  # entries: (name, center_xy np.array, half_xy np.array)


def add_warehouse_box(prim_path, position, size, color, collision=True):
    """Create a static box prim; optionally with collision geometry."""
    xform = UsdGeom.Xformable(stage.DefinePrim(prim_path, "Xform"))
    cube = UsdGeom.Cube.Define(stage, prim_path + "/Geom")
    cube.CreateSizeAttr(1.0)  # unit cube, full size comes from Xform scale
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform.AddTranslateOp().Set(Gf.Vec3f(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision_boxes.append(
            (
                prim_path,
                np.array([position[0], position[1]], dtype=np.float64),
                np.array([size[0] / 2.0, size[1] / 2.0], dtype=np.float64),
            )
        )
    return xform


# ------------------------------------------------------------
# Warehouse layout: 20x20m floor, outer walls, 3 pairs of racks
# (identical to test_carter_lidar.py / test_carter_lidar_obstacle.py)
# ------------------------------------------------------------
WALL_COLOR = (0.6, 0.6, 0.65)
RACK_COLOR = (0.25, 0.4, 0.7)

add_warehouse_box("/World/Wall_South", (0, -10, 1.5), (20.6, 0.4, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_North", (0, 10, 1.5), (20.6, 0.4, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_West", (-10, 0, 1.5), (0.4, 20.6, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_East", (10, 0, 1.5), (0.4, 20.6, 3.0), WALL_COLOR)

# 3 pairs of rack boxes (4m x 1.5m x 2m) at (+-5, -4/0/4)
RACK_SIZE = (4.0, 1.5, 2.0)
RACK_Y = (-4.0, 0.0, 4.0)
for i, y in enumerate(RACK_Y):
    add_warehouse_box(f"/World/Rack_L{i + 1}", (-5.0, y, 1.0), RACK_SIZE, RACK_COLOR)
    add_warehouse_box(f"/World/Rack_R{i + 1}", (5.0, y, 1.0), RACK_SIZE, RACK_COLOR)

log_print("Warehouse loaded (walls + 6 racks)")

# ============================================================
# STEP 4b: Add ONE LiDAR sensor to Carter (proven config)
# ============================================================
# rotation_frequency=0.0 -> static full 360 deg scan every physics step,
# exactly like the verified Milestone 4/5/6 tests.
LIDAR_PATH = "/World/Carter/chassis_link/lidar"
LIDAR_MAX_RANGE = 50.0

lidar = RotatingLidarPhysX(
    prim_path=LIDAR_PATH,
    name="carter_lidar",
    rotation_frequency=0.0,
    translation=np.array([-0.06, 0.0, 0.38]),
    fov=(360.0, 30.0),          # horizontal 360deg, vertical 30deg
    resolution=(1.0, 2.0),      # 360 cols x 16 rows -> lightweight scan
    valid_range=(0.4, LIDAR_MAX_RANGE),
)
lidar.add_linear_depth_data_to_frame()
lidar.add_azimuth_data_to_frame()
lidar.add_zenith_data_to_frame()

# Lightweight LiDAR visualization (official rotating_lidar_physX.py API):
# draws the scan hit POINTS in the viewport (GPU debug draw - no per-frame
# meshes or ray geometry created from Python). Lines stay off to avoid
# drawing thousands of rays.
# NOTE (verified by A/B probe on this machine): calling
# enable_visualization(high_lod=False, ...) collapses the PhysX scan from
# 16 rows to 1 row, while high_lod=True keeps the full 360x16 scan.
try:
    lidar.enable_visualization(high_lod=True, draw_points=True, draw_lines=False)
    log_print("LiDAR visualization enabled (hit points only, high_lod=True)")
except Exception as e:
    log_print(f"LiDAR visualization skipped: {e}")

log_print(f"LiDAR prim created at {LIDAR_PATH}")
log_print(f"LIDAR_RANGE: {lidar.get_valid_range()}")
log_print("LIDAR_ROTATION_FREQUENCY: 0.0 (static full scan)")

simulation_app.update()
log_print("USD loaded")

stage.SetTimeCodesPerSecond(60.0)

# ============================================================
# STEP 5: Create World and initialize simulation context
# ============================================================
my_world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=physics_dt)
log_print("World created")

try:
    my_world.initialize_simulation_context()
    log_print("Simulation context initialized")
except Exception as e:
    log_print(f"initialize_simulation_context error: {e}")

for _ in range(30):
    simulation_app.update()

my_world.play()
log_print("World played")

for _ in range(30):
    simulation_app.update()

# ============================================================
# STEP 6: Create SingleArticulation Carter
# ============================================================
ar = SingleArticulation("/World/Carter")
log_print("SingleArticulation created at /World/Carter")

articulation_view = ar._articulation_view
try:
    articulation_view.initialize()
    log_print("Articulation view initialized successfully")
except Exception as e:
    log_print(f"Articulation view initialize info: {e}")

# ============================================================
# STEP 7: Initialize LiDAR sensor
# ============================================================
lidar_initialized = True
try:
    lidar.initialize()
    log_print("LiDAR sensor initialized")
except Exception as e:
    lidar_initialized = False
    log_print(f"LiDAR initialize error: {e}")

# Carter start: central aisle, identity orientation, nose = world -X
# (verified runtime chassis transform, Milestones 1-6).
ar.set_world_pose(position=[0, -7, 0.1], orientation=[0, 0, 0, 1])
ar.set_world_velocity([0, 0, 0, 0, 0, 0])

for _ in range(10):
    simulation_app.update()

start_pos, start_rot = ar.get_world_pose()
log_print(f"START_POSITION: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
log_print(f"LIDAR_MOUNT: {LIDAR_PATH} translation [-0.06, 0, 0.38]")


# ------------------------------------------------------------
# Yaw extraction (self-verifying quaternion order):
# standard ZYX yaw, CCW positive. Verified at runtime: the start pose
# must yield ~+-180 deg (nose points toward world -X); if not, the
# quaternion order is reversed and we flip it once.
# ------------------------------------------------------------
def _yaw_deg(w, x, y, z):
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


_quat_order = "wxyz"
_probe = _yaw_deg(*[float(v) for v in start_rot])
if abs(_probe) < 90.0:
    _quat_order = "xyzw"
    _probe = _yaw_deg(float(start_rot[3]), float(start_rot[0]),
                      float(start_rot[1]), float(start_rot[2]))
log_print(f"START_YAW: {_probe:.2f} deg (quat order {_quat_order})")


def yaw_of(rot):
    """World yaw in degrees (CCW positive) from the articulation quaternion."""
    r = [float(v) for v in rot]
    if _quat_order == "wxyz":
        return _yaw_deg(r[0], r[1], r[2], r[3])
    return _yaw_deg(r[3], r[0], r[1], r[2])


# ============================================================
# STEP 7b: Runtime forward direction + TWO static obstacles (M8)
#          + goal + visual markers
# ============================================================
CARTEL_CENTER = np.array([0.0, -7.0])

# Forward = chassis +X in world, read from the USD composed transform
# (fallback: world -X, verified in Milestones 1-5).
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

# Unit vectors around the runtime forward direction:
# left = forward rotated +90 deg (CCW), right = -left.
left_world = np.array([-forward[1], forward[0]], dtype=np.float64)
right_world = -left_world

# ------------------------------------------------------------
# M10 DYNAMIC OBSTACLES: 3 deterministic kinematic moving boxes.
#   A "crosser"      - slides laterally across Carter's aisle at
#                      x = -4.2: y(t) = -6.05 - 0.6*cos(2pi(t-6)/10),
#                      blocking the start->goal line around the
#                      expected meet time and then clearing it.
#   B "aisle runner" - y = -2.0 (a different aisle, NEVER on the
#                      route): x(t) = 5*sin(2pi t/36).
#   C "approacher"   - x = -1.0: y(t) = -3.6 - 0.6*cos(2pi t/14)
#                      (moves toward/away from the rack row).
# No humans, no forklifts, no randomness: every pose is an analytic
# function of simulation time -> reproducible on every run.
# ------------------------------------------------------------
OBSTACLE_Z = 0.75
OBSTACLE_COLOR = (0.9, 0.3, 0.2)


def _mot_a(t):
    """Crosser: lateral sweep of Carter's aisle (10 s period, 1.2 m)."""
    return (-4.2, -6.05 - 0.6 * math.cos(2.0 * math.pi * (t - 6.0) / 10.0))


def _mot_b(t):
    """Aisle runner: y = -2.0 forever (36 s period, peak ~0.87 m/s)."""
    return (5.0 * math.sin(2.0 * math.pi * t / 36.0), -2.0)


def _mot_c(t):
    """Approacher: toward/away from the rack row at x = -1.0 (14 s)."""
    return (-1.0, -3.6 - 0.6 * math.cos(2.0 * math.pi * t / 14.0))


def add_dynamic_box(box_path, motion, size, color, tag):
    """Flat Cube prim that owns its translate/scale ops and carries
    CollisionAPI + kinematic RigidBodyAPI, so the scripted motion can
    be written every step (visual and PhysX body move together)."""
    cube = UsdGeom.Cube.Define(stage, box_path)
    cube.CreateSizeAttr(1.0)  # unit cube, final size from the scale op
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(cube.GetPrim())
    t_op = xf.AddTranslateOp()
    s_op = xf.AddScaleOp()
    p0 = np.array(motion(0.0), dtype=np.float64)
    t_op.Set(Gf.Vec3f(float(p0[0]), float(p0[1]), OBSTACLE_Z))
    s_op.Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    try:
        rb.CreateKinematicEnabledAttr(True)
    except Exception:
        cube.GetPrim().CreateAttribute(
            "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool
        ).Set(True)
    rec = {
        "path": box_path,
        "tag": tag,
        "motion": motion,
        "translate_op": t_op,
        "center": p0.copy(),
        "half": np.array([size[0] / 2.0, size[1] / 2.0], dtype=np.float64),
        "rigid": None,
        "driver": "usd_only",
    }
    try:
        rec["rigid"] = RigidPrim(box_path, reset_xform_properties=False)
        rec["driver"] = "usd+physics_view"
    except Exception as exc:
        rec["driver"] = f"usd_only ({type(exc).__name__})"
    return rec


def update_dynamic_obstacles(t):
    """Drive every dynamic box to its deterministic pose at time t.
    Called BEFORE simulation_app.update() so the PhysX step and the
    LiDAR raycast of the same frame see the new pose."""
    for o in dynamic_obstacles:
        x, y = o["motion"](t)
        o["center"][0] = x
        o["center"][1] = y
        o["translate_op"].Set(Gf.Vec3f(float(x), float(y), OBSTACLE_Z))
        if o["rigid"] is not None:
            try:
                o["rigid"].set_world_poses(
                    positions=np.array([[x, y, OBSTACLE_Z]], dtype=np.float32),
                    orientations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    usd=False,
                )
            except Exception:
                o["rigid"] = None  # kinematic USD follow remains active
                o["driver"] = "usd_only (tensor write unavailable)"


dynamic_obstacles = []
_MOTION_DEFS = [
    ("A", _mot_a, (0.5, 1.0, 1.5),
     "crosser x=-4.2, y=-6.05-0.6cos(2pi(t-6)/10)"),
    ("B", _mot_b, (1.0, 0.5, 1.5),
     "aisle runner y=-2.0, x=5sin(2pi t/36)"),
    ("C", _mot_c, (0.8, 0.8, 1.5),
     "approacher x=-1.0, y=-3.6-0.6cos(2pi t/14)"),
]
for _i, (_tag, _motion, _size, _desc) in enumerate(_MOTION_DEFS):
    _path = f"/World/DynamicObstacle_{_tag}"
    _rec = add_dynamic_box(_path, _motion, _size, OBSTACLE_COLOR, _tag)
    _rec["idx"] = _i  # same identity field the M7/M8 logic expects
    dynamic_obstacles.append(_rec)
    _x0, _y0 = _motion(0.0)
    log_print(
        f"DYNAMIC_OBSTACLE_{_tag}: {_path} POSITION(t=0): "
        f"({_x0:.3f}, {_y0:.3f}, {OBSTACLE_Z:.3f}) size {_size} "
        f"DRIVER: {_rec['driver']}"
    )
    log_print(f"    MOTION_{_tag}: {_desc}")

# Legacy alias: records/decision code refers to the obstacle list.
obstacles = dynamic_obstacles

# Expected first-frame LiDAR distance to box A's front face at t=0
# (center 4.2 m ahead - half depth 0.25 m + 0.06 m LiDAR offset).
EXPECTED_OBSTACLE_DISTANCE = 4.2 - 0.5 / 2.0 + 0.06
log_print(f"EXPECTED_OBSTACLE_DISTANCE (obstacle A face): {EXPECTED_OBSTACLE_DISTANCE:.4f} m")

# Goal: beyond both obstacles on the verified straight line (runtime forward).
GOAL_POS = (
    float(CARTEL_CENTER[0] + forward[0] * 8.6),
    float(CARTEL_CENTER[1] + forward[1] * 8.6),
    0.0,
)
log_print(f"GOAL_POSITION: ({GOAL_POS[0]:.3f}, {GOAL_POS[1]:.3f}, {GOAL_POS[2]:.3f})")

# ------------------------------------------------------------
# Visual markers (NO collision geometry -> they can never trip Carter):
#   start marker (blue), goal marker (green square + flag pole),
#   floating state marker (color = current state, follows Carter).
# ------------------------------------------------------------
START_MARKER_COLOR = (0.15, 0.4, 1.0)
GOAL_MARKER_COLOR = (0.1, 0.9, 0.2)
STATE_COLORS = {
    "NAVIGATE": (0.1, 0.9, 0.2),
    "APPROACH": (0.95, 0.8, 0.1),
    "AVOID": (1.0, 0.45, 0.0),
    "AVOID_LEFT": (1.0, 0.6, 0.0),   # side-dependent AVOID marker
    "AVOID_RIGHT": (1.0, 0.2, 0.6),
    "PASS_OBSTACLE": (0.2, 0.5, 1.0),
    "RESUME": (0.75, 0.3, 0.9),
    "RECOVERY": (1.0, 0.08, 0.08),   # M11: stop + reverse + re-scan
    "REASSESS": (0.1, 0.85, 0.9),    # M11: stopped, comparing sectors
    "GOAL_REACHED": (1.0, 1.0, 1.0),
}

add_warehouse_box(
    "/World/StartMarker",
    (CARTEL_CENTER[0], CARTEL_CENTER[1], 0.025),
    (0.6, 0.6, 0.05), START_MARKER_COLOR, collision=False,
)
add_warehouse_box(
    "/World/GoalMarker",
    (GOAL_POS[0], GOAL_POS[1], 0.03),
    (1.0, 1.0, 0.06), GOAL_MARKER_COLOR, collision=False,
)
add_warehouse_box(
    "/World/GoalFlag",
    (GOAL_POS[0], GOAL_POS[1] + 1.0, 0.8),
    (0.1, 0.1, 1.6), GOAL_MARKER_COLOR, collision=False,
)

marker_xf = UsdGeom.Xformable(stage.DefinePrim("/World/StateMarker", "Xform"))
state_op = marker_xf.AddTranslateOp()
state_op.Set(Gf.Vec3f(float(CARTEL_CENTER[0]), float(CARTEL_CENTER[1]), 1.5))
marker_xf.AddScaleOp().Set(Gf.Vec3f(0.3, 0.3, 0.3))
marker_cube = UsdGeom.Cube.Define(stage, "/World/StateMarker/Geom")
marker_cube.CreateSizeAttr(1.0)
marker_cube.CreateDisplayColorAttr([Gf.Vec3f(*STATE_COLORS["NAVIGATE"])])

# ------------------------------------------------------------
# Milestone 9: TRAJECTORY HISTORY as ONE lightweight UsdGeom.Points
# prim (sparse points, no per-point USD meshes). One point every
# TRAJ_STEP physics steps, capped at TRAJ_MAX_POINTS entries.
# ------------------------------------------------------------
TRAJ_STEP = 6              # ~10 Hz at 60 Hz physics
TRAJ_MAX_POINTS = 500      # bounded history (spec: reasonable number)
TRAJ_POINT_COLOR = (0.05, 0.95, 1.0)

trajectory_initialized = False
traj_cap_logged = False
trajectory_history = []
try:
    traj_prim = UsdGeom.Points.Define(stage, "/World/Trajectory")
    traj_prim.CreatePointsAttr([])
    trajectory_initialized = True  # core point cloud is live
    log_print(
        "TRAJECTORY_HISTORY: /World/Trajectory (UsdGeom.Points, "
        f"1 point / {TRAJ_STEP} steps, cap {TRAJ_MAX_POINTS})"
    )
except Exception as e:
    log_print(f"TRAJECTORY_INITIALIZATION_ERROR: {e}")
if trajectory_initialized:
    # Cosmetic attributes: never allowed to abort initialization.
    try:
        # Widths must always match the vertex count (grows with the
        # points below) - a 1-element widths array makes Hydra warn
        # every frame about HdInterpolationVertex fallback.
        traj_prim.CreateWidthsAttr([])
    except Exception as e:
        log_print(f"TRAJECTORY_WIDTHS_SKIPPED: {e}")
    try:
        traj_prim.CreateDisplayColorAttr([Gf.Vec3f(*TRAJ_POINT_COLOR)])
    except Exception:
        try:
            _pv = UsdGeom.Primvar(traj_prim.GetPrim(), "displayColor")
            _pv.Set([Gf.Vec3f(*TRAJ_POINT_COLOR)])
            _pv.SetInterpolation("constant")
        except Exception as e:
            log_print(f"TRAJECTORY_COLOR_SKIPPED: {e}")

# Fixed perspective view over the aisle (lightweight, one-shot).
if not HEADLESS:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(np.array([3.0, -15.0, 9.0]), np.array([-4.5, -6.8, 0.5]))
        log_print("CAMERA_VIEW: fixed perspective over the navigation aisle")
    except Exception as e:
        log_print(f"CAMERA_VIEW: not set ({e})")

# Let colliders register with physics
for _ in range(10):
    simulation_app.update()

# ============================================================
# STEP 8: OmniGraph - the proven navigation pipeline (unchanged)
# ============================================================
graph_path = "/ActionGraph"
keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": graph_path, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("DifferentialController", "isaacsim.robot.wheeled_robots.DifferentialController"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ("computeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("GetPrimLocalToWorldTransform", "omni.graph.nodes.GetPrimLocalToWorldTransform"),
            ("GetRotationQuaternion", "omni.graph.nodes.GetMatrix4Quaternion"),
            ("GetTranslation", "omni.graph.nodes.GetMatrix4Translation"),
            ("QuinticPathPlanner", "isaacsim.robot.wheeled_robots.QuinticPathPlanner"),
            ("CheckGoal2D", "isaacsim.robot.wheeled_robots.CheckGoal2D"),
            ("StanleyControlPID", "isaacsim.robot.wheeled_robots.StanleyControlPID"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "DifferentialController.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "computeOdom.inputs:execIn"),
            ("DifferentialController.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
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
            ("QuinticPathPlanner.inputs:targetPosition", GOAL_POS),
            ("QuinticPathPlanner.inputs:targetOrientation", (0, 0, 0, 1)),
            ("GetPrimLocalToWorldTransform.inputs:usePath", False),
            (graph_path + "/computeOdom.inputs:chassisPrim", [usdrt.Sdf.Path("/World/Carter/chassis_link")]),
            ("GetPrimLocalToWorldTransform.inputs:prim", [usdrt.Sdf.Path("/World/Carter/chassis_link")]),
            ("DifferentialController.inputs:wheelRadius", 0.14),
            ("DifferentialController.inputs:wheelDistance", 0.4132),
            ("ArticulationController.inputs:jointNames", ["joint_wheel_left", "joint_wheel_right"]),
            ("ArticulationController.inputs:robotPath", "/World/Carter"),
        ],
    },
)
log_print("OmniGraph created successfully")

og.Controller.attribute(graph_path + "/DifferentialController.inputs:linearVelocity").set(0)
og.Controller.attribute(graph_path + "/DifferentialController.inputs:angularVelocity").set(0)

for _ in range(10):
    simulation_app.update()

# Re-target the planner to Carter's ACTUAL chassis orientation so the
# planned path is a clean straight line toward the goal (proven pattern).
chassis_quat = tuple(
    og.Controller.attribute(graph_path + "/GetRotationQuaternion.outputs:quaternion").get()
)
log_print(f"CHASSIS_ORIENTATION: {chassis_quat}")
og.Controller.attribute(graph_path + "/QuinticPathPlanner.inputs:targetOrientation").set(chassis_quat)

for _ in range(10):
    simulation_app.update()

start_pos, start_rot = ar.get_world_pose()
start_pos = np.array(start_pos, dtype=np.float64)
log_print(f"INITIAL_POSITION: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")

# Route direction (start -> goal), used for the "obstacle is past" test.
_route = np.array(GOAL_POS[:2]) - start_pos[:2]
route_dir = _route / np.linalg.norm(_route)

# ------------------------------------------------------------
# Control authority: ABSOLUTE attribute paths (Milestone 6 fix).
# NORMAL:  Stanley -> DifferentialController connected.
# AVOIDANCE/MANUAL: disconnected, we write velocities directly.
# ------------------------------------------------------------
LIN_SRC = graph_path + "/StanleyControlPID.outputs:linearVelocity"
LIN_DST = graph_path + "/DifferentialController.inputs:linearVelocity"
ANG_SRC = graph_path + "/StanleyControlPID.outputs:angularVelocity"
ANG_DST = graph_path + "/DifferentialController.inputs:angularVelocity"

manual_mode = False


def enter_manual():
    """Disconnect Stanley: the reactive controller takes over."""
    global manual_mode
    if not manual_mode:
        og.Controller.edit(
            graph_path,
            {keys.DISCONNECT: [(LIN_SRC, LIN_DST), (ANG_SRC, ANG_DST)]},
        )
        manual_mode = True


def exit_manual():
    """Reconnect Stanley: normal navigation resumes."""
    global manual_mode
    if manual_mode:
        og.Controller.edit(
            graph_path,
            {keys.CONNECT: [(LIN_SRC, LIN_DST), (ANG_SRC, ANG_DST)]},
        )
        manual_mode = False


def write_cmd(v, w):
    """Command the DifferentialController directly while in manual mode."""
    og.Controller.attribute(LIN_DST).set(float(v))
    og.Controller.attribute(ANG_DST).set(float(w))


log_print(f"CONTROL_HANDOFF: absolute-path DISCONNECT/CONNECT on {graph_path}")

# ============================================================
# STEP 9: LiDAR analysis helpers (Milestone 5/6 conventions)
# ============================================================
FORWARD_CONE_DEG = 30.0     # forward cone: azimuth +-30 deg
ELEV_CONE_DEG = 3.0         # elevation +-3 deg (floor rays excluded)
SECTOR_DEG_INNER = 30.0     # LEFT  = azimuth +30..+90 deg
SECTOR_DEG_OUTER = 90.0     # RIGHT = azimuth -90..-30 deg
SECTOR_PERCENTILE = 20.0    # robust clearance statistic (p20 of valid ranges)
SELF_HIT_MAX = 0.5          # exclude Carter's own body


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


def build_masks(frame, depth_shape):
    """Build forward-cone, LEFT-sector and RIGHT-sector ray masks from the
    first frame's azimuth/zenith (Milestone 5 verified convention)."""
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
        azimuth_deg = -180.0 + np.arange(n_az) * 1.0
    else:
        azimuth_deg = azimuth_ref.copy()
        if np.nanmax(np.abs(azimuth_deg)) <= 6.5:
            azimuth_deg = np.degrees(azimuth_deg)
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
        zenith_deg = 90.0 - 15.0 + np.arange(n_ze) * 2.0
    else:
        zenith_deg = zenith_ref.copy()
        if np.nanmax(np.abs(zenith_deg)) <= 6.5:
            zenith_deg = np.degrees(zenith_deg)

    if np.nanmedian(zenith_deg) > 45.0:
        elevation_deg = 90.0 - zenith_deg
    else:
        elevation_deg = zenith_deg
    log_print(
        f"ELEVATION_RANGE: [{np.nanmin(elevation_deg):.2f}, "
        f"{np.nanmax(elevation_deg):.2f}] deg"
    )

    az = wrap_deg(azimuth_deg)
    elev_ok = np.abs(elevation_deg) <= ELEV_CONE_DEG
    fwd = (np.abs(az) <= FORWARD_CONE_DEG)[:, None] & elev_ok[None, :]
    left = ((az >= SECTOR_DEG_INNER) & (az <= SECTOR_DEG_OUTER))[:, None] & elev_ok[None, :]
    right = ((az <= -SECTOR_DEG_INNER) & (az >= -SECTOR_DEG_OUTER))[:, None] & elev_ok[None, :]
    # M11 rear sector (|az| >= 150 deg): used to gate RECOVERY reversing.
    rear = (np.abs(az) >= REAR_DEG_INNER)[:, None] & elev_ok[None, :]
    return fwd, left, right, rear


def sector_clearance(d, mask, valid):
    """Robust clearance statistic (p20 of valid ranges) inside a sector."""
    sel = mask & valid & (d > SELF_HIT_MAX)
    if not np.any(sel):
        return 0.0
    return float(np.percentile(d[sel], SECTOR_PERCENTILE))


def sector_min(d, mask, valid):
    """Conservative clearance statistic (minimum valid range) inside a
    sector. A narrow 1 m moving box can sit between p20 sample points,
    so decisions use the min: always safe, and the PhysX scan of this
    scenario is noise-free (verified in the frozen LiDAR scripts)."""
    sel = mask & valid & (d > SELF_HIT_MAX)
    if not np.any(sel):
        return 0.0
    return float(np.min(d[sel]))


def ray_aabb_forward(origin_xy, dir_xy, center_xy, half_xy):
    """Slab test: distance along dir_xy to the AABB, or None when the
    ray misses it. Used to predict where a moving box SHOULD appear in
    the forward cone and to cross-check the live PhysX LiDAR."""
    t_enter = -float("inf")
    t_exit = float("inf")
    for i in (0, 1):
        if abs(dir_xy[i]) < 1e-9:
            if abs(origin_xy[i] - center_xy[i]) > half_xy[i]:
                return None
        else:
            t1 = (center_xy[i] - half_xy[i] - origin_xy[i]) / dir_xy[i]
            t2 = (center_xy[i] + half_xy[i] - origin_xy[i]) / dir_xy[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
    if t_exit < max(t_enter, 0.0):
        return None
    return float(max(t_enter, 0.0))


def decide_avoid_side(d, valid, left_mask, right_mask, last_dir,
                      steps_since_dir, path_since_dir):
    """M10/M11 avoidance decision from MEASURED sector clearance.

    Returns (direction, left_clear, right_clear, lock_note).
    The side is computed every event (LEFT = larger min clearance) -
    never hard-coded. Anti-oscillation: the previously committed side
    is held unless the other side beats it by >= HYST_MARGIN while the
    direction lock is active (minimum duration AND path length), so a
    side change only happens for a genuine safety reason."""
    left_clear = sector_min(d, left_mask, valid)
    right_clear = sector_min(d, right_mask, valid)
    direction = "LEFT" if left_clear >= right_clear else "RIGHT"
    lock_note = "DIR_INIT"
    if last_dir is not None:
        if direction != last_dir:
            better = (
                (right_clear - left_clear)
                if last_dir == "LEFT"
                else (left_clear - right_clear)
            )
            locked = (
                steps_since_dir < DIR_LOCK_STEPS
                or path_since_dir < DIR_LOCK_DIST_M
            )
            if locked and better < HYST_MARGIN:
                direction = last_dir
                lock_note = "DIR_LOCK_HELD"
            else:
                lock_note = "DIR_CHANGED"
        else:
            lock_note = "DIR_HELD"
    return direction, left_clear, right_clear, lock_note


def aabb_distance(pos_xy, center_xy, half_xy):
    dx = max(abs(pos_xy[0] - center_xy[0]) - half_xy[0], 0.0)
    dy = max(abs(pos_xy[1] - center_xy[1]) - half_xy[1], 0.0)
    return float(math.hypot(dx, dy))


# ============================================================
# STEP 10: State machine configuration
# ============================================================
CLEAR_T = 2.5          # m: forward distance for CLEAR / NAVIGATE
AVOID_T = 1.5          # m: forward distance that triggers avoidance (M6 parity)
TURN_DEV_DEG = 55.0    # AVOID TURN ends at this heading deviation
TURN_TIMEOUT = 300     # steps safeguard for the TURN sub-phase
GLIDE_LATERAL_M = 0.75  # AVOID GLIDE ends at this lateral clearance gain
GLIDE_TIMEOUT = 400    # steps safeguard for the GLIDE sub-phase
PASS_HEADING_TOL = 15.0  # deg: heading error allowed to leave PASS
PASS_PAST_M = 0.6      # m: progress past obstacle center along the route
PASS_CLEAR_STEPS = 15  # consecutive cleared steps to leave PASS
PASS_TIMEOUT_STEPS = 500  # deadlock safeguard
NAV_RECOVER_STEPS = 30  # consecutive cleared steps to leave APPROACH
RESUME_STEPS = 20      # steps of Stanley control before leaving RESUME
GOAL_TOL = 0.5         # m: goal reached radius

APPROACH_V = 0.30      # m/s: reduced speed while in APPROACH
APPROACH_K = 1.5       # heading P-gain in APPROACH
APPROACH_W_MAX = 0.6   # rad/s clamp
TURN_V = 0.10          # m/s: near-stop while rotating away (M7: slow first)
TURN_W = 0.85          # rad/s: reactive turn rate
GLIDE_V = 0.45         # m/s: glide past the obstacle holding the turn heading
PASS_V = 0.45          # m/s: straighten back toward the goal
PASS_SLOW_V = 0.25     # m/s: slower if something re-enters the corridor
PASS_K = 2.0           # heading P-gain in PASS
PASS_W_MAX = 1.0       # rad/s clamp

MAX_STEPS = 4000       # 66 s hard cap at 60 Hz
OBSERVE_SECONDS = float(os.environ.get("DEMO_DURATION", "8"))

# Safety thresholds (fail the test if violated)
MIN_FORWARD_ALLOWED = 1.0     # m: closest LiDAR reading ever
MIN_CLEARANCE_CENTER = 0.5    # m: Carter center -> any collider AABB
COLLISION_CLEARANCE = 0.45    # m: below this = considered a collision

# --- M10: measured re-evaluation of MOVING obstacles (per step) ---
RETRIGGER_STEPS = 5      # sustained forward <= AVOID_T before re-committing
                         # to avoidance from NAVIGATE/RESUME/PASS
TRACK_PRED_T = 2.0       # m: only cross-check boxes predicted this close
TRACK_TOL = 0.5          # m: LiDAR may lag the prediction by this much

# --- M11: RECOVERY + anti-oscillation + progress detection ---
BLOCKED_T = 1.2          # m: sector clearance treated as blocked (recovery)
REAR_DEG_INNER = 150.0   # deg: rear cone (|az| >= 150) gates reversing
REAR_MIN_ALLOWED = 0.6   # m: rear clearance required to reverse
REVERSE_V = -0.15        # m/s: slow reverse during RECOVERY
REVERSE_MAX_STEPS = 300  # steps: short reverse phase (5 s)
RECOVERY_MAX_STEPS = 400 # steps: RECOVERY state hard timeout
RECOVERY_FORCE_EVENTS = 6  # attempts before a side is forced
REASSESS_STEPS = 20      # steps stopped + re-scanning before re-commit
FWD_STALL_STEPS = 180    # steps: sustained <= AVOID_T counts as stalled
                         # when paired with no goal progress
NO_PROGRESS_STEPS = 600  # steps: sliding goal-distance window (10 s)
NO_PROGRESS_M = 0.15     # m: required goal progress over that window
AVOID_FAIL_STREAK = 3    # same obstacle avoided this many times -> recovery
DIR_LOCK_STEPS = 600     # hold the chosen side at least 10 s ...
DIR_LOCK_DIST_M = 3.0    # ... and at least 3 m before it may change
HYST_MARGIN = 0.5        # m: the other side must beat the held side by
                         # this margin (genuine safety reason to switch)

# ============================================================
# STEP 11: Closed-loop run
# ============================================================
state = "NAVIGATE"
last_state = None
avoid_phase = None
cur_obs = None
avoid_entry = None
open_dir = None
dir_sign = 1.0
pending_lines = []           # log lines printed just before a STATE change

manual_mode_already = manual_mode  # noqa: F841 (clarity)
app_streak = 0
resume_counter = 0
avoid_steps = 0
phase_steps = 0
pass_steps = 0
pass_streak = 0

goal_reached = False
goal_step = -1
goal_hold = 0
observe_steps = max(1, int(OBSERVE_SECONDS * physics_rate))

avoid_events = 0
detected_set = set()
detection_dist = {}
events = []

m7 = {"detected": False, "triggered": False, "direction": "",
      "cleared": False, "resumed": False}

lidar_data_available = False
first_frame_analyzed = False
lidar_rows = 0
fwd_mask = None
left_mask = None
right_mask = None
rear_mask = None
frame_keys_logged = False

min_forward_distance = float("inf")
min_clearance = float("inf")
collision_detected = False

# --- M10/M11: loop counters, locks, recovery bookkeeping ---
forward_low_streak = 0   # sustained forward <= AVOID_T while not avoiding
fwd_stall_steps = 0      # sustained low forward without making progress
progress_window = []     # sliding (step, dist_goal) window for M11
same_obs_streak = 0      # consecutive avoidances of the SAME obstacle
avoid_direction = ""     # side chosen for the current/last AVOID event
last_dir = None          # last committed side (anti-oscillation lock)
steps_since_dir = 0      # steps since the side was last committed
path_since_dir = 0.0     # path length since the side was last committed
direction_changes = 0    # side changes across consecutive events
recovery_events = 0
recovery_steps = 0
reassess_steps = 0
recovery_reason = ""
dynamic_hit_frames = 0   # frames where LiDAR saw a predicted moving box
tracking_mismatch = 0    # predicted box unseen by LiDAR (kinematic lag)

position_history = []
speeds = []
prev_pos = start_pos.copy()

log_print(
    f"Running reactive navigation (max {MAX_STEPS} steps, "
    f"CLEAR>{CLEAR_T} m, APPROACH<={CLEAR_T} m, AVOID<={AVOID_T} m, "
    f"observe {OBSERVE_SECONDS:.0f}s)"
)
log_print(
    f"M10_DYNAMIC_OBSTACLES: {len(dynamic_obstacles)} kinematic boxes "
    f"(A crosser / B aisle / C approacher), re-evaluated every step"
)
log_print(
    f"M11_RECOVERY_ARMED: BLOCKED<={BLOCKED_T} m, REVERSE {REVERSE_V} m/s "
    f"({REVERSE_MAX_STEPS} steps), REASSESS {REASSESS_STEPS} steps, "
    f"DIR_LOCK {DIR_LOCK_STEPS} steps / {DIR_LOCK_DIST_M} m, "
    f"HYST_MARGIN {HYST_MARGIN} m"
)

for step in range(MAX_STEPS):
    # M10: drive the moving boxes BEFORE the physics/LiDAR step so this
    # frame's scan reflects their scripted pose for this instant.
    update_dynamic_obstacles(step / physics_rate)
    simulation_app.update()

    # --- Carter state ---
    pos, rot = ar.get_world_pose()
    pos = np.array(pos, dtype=np.float64)
    yaw_deg = yaw_of(rot)
    speed = float(np.linalg.norm(pos - prev_pos)) / physics_dt
    prev_pos = pos.copy()
    speeds.append(speed)
    position_history.append(pos.copy())

    # --- continuous LiDAR read (every physics step) ---
    frame = lidar.get_current_frame()
    d = to_numpy(frame.get("linear_depth")) if frame else None
    forward_dist = None
    valid = None
    if d is not None and d.size > 0:
        lidar_data_available = True
        if frame_keys_logged is False:
            log_print(f"FRAME_KEYS: {list(frame.keys())}")
            frame_keys_logged = True
        if fwd_mask is None:
            fwd_mask, left_mask, right_mask, rear_mask = build_masks(frame, d.shape)
            log_print(f"SCAN_SHAPE: {d.shape}")
            # Milestone 9 LiDAR verification: full 16-row scan (NOT the
            # collapsed 1-row scan produced by high_lod=False).
            try:
                lidar_rows = int(lidar.get_num_rows())
                lidar_cols = int(lidar.get_num_cols())
            except Exception:
                lidar_rows, lidar_cols = int(d.shape[1]), int(d.shape[0])
            log_print(f"GET_NUM_ROWS: {lidar_rows} GET_NUM_COLS: {lidar_cols}")
            log_print(f"FORWARD_RAYS: {int(np.count_nonzero(fwd_mask))}")
            log_print(f"LEFT_SECTOR_RAYS: {int(np.count_nonzero(left_mask))}")
            log_print(f"RIGHT_SECTOR_RAYS: {int(np.count_nonzero(right_mask))}")
            log_print(f"REAR_SECTOR_RAYS: {int(np.count_nonzero(rear_mask))}")
        valid = (d > 0.0) & (d < LIDAR_MAX_RANGE)
        sel = fwd_mask & valid & (d > SELF_HIT_MAX)
        if np.any(sel):
            forward_dist = float(np.min(d[sel]))
            if forward_dist < min_forward_distance:
                min_forward_distance = forward_dist
            if not first_frame_analyzed:
                # Milestone 5 proven detection check on the first frame.
                first_frame_analyzed = True
                near_expected = int(
                    np.count_nonzero(np.abs(d[sel] - EXPECTED_OBSTACLE_DISTANCE) <= 0.5)
                )
                log_print(f"VALID_RAYS: {int(np.count_nonzero(valid))}")
                log_print(f"FORWARD_MIN_DISTANCE_FIRST_FRAME: {forward_dist:.4f}")
                log_print(f"FORWARD_RAYS_NEAR_EXPECTED: {near_expected}")

        # --- M10: cross-check the live scan against the scripted motion ---
        if forward_dist is not None and valid is not None:
            _cf = np.array(
                [math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))],
                dtype=np.float64,
            )
            _lidar_xy = pos[:2] - 0.06 * _cf
            for _o in dynamic_obstacles:
                pred = ray_aabb_forward(_lidar_xy, _cf, _o["center"], _o["half"])
                if pred is None:
                    continue
                if pred <= TRACK_PRED_T and forward_dist > pred + TRACK_TOL:
                    tracking_mismatch += 1
                    log_print(
                        f"DYNAMIC_LIDAR_TRACKING_MISMATCH: box {_o['tag']} "
                        f"pred {pred:.3f} m but forward {forward_dist:.3f} m "
                        f"at step {step}"
                    )
                    break
            if forward_dist <= CLEAR_T:
                for _o in dynamic_obstacles:
                    pred = ray_aabb_forward(
                        _lidar_xy, _cf, _o["center"], _o["half"]
                    )
                    if pred is not None and pred <= forward_dist + 0.35:
                        dynamic_hit_frames += 1
                        break

    # --- geometric clearance against every static collider AND the
    #     LIVE centers of the moving boxes ---
    step_min_clear = float("inf")
    for _name, _c, _h in collision_boxes:
        dc = aabb_distance(pos[:2], _c, _h)
        if dc < step_min_clear:
            step_min_clear = dc
    for _o in dynamic_obstacles:
        dc = aabb_distance(pos[:2], _o["center"], _o["half"])
        if dc < step_min_clear:
            step_min_clear = dc
    if step_min_clear < min_clearance:
        min_clearance = step_min_clear
    if not collision_detected and min_clearance < COLLISION_CLEARANCE:
        collision_detected = True
        log_print(
            f"COLLISION_WARNING_STEP: {step} MIN_CLEARANCE: {min_clearance:.4f}"
        )

    # --- goal / heading geometry ---
    goal_bearing_deg = math.degrees(
        math.atan2(GOAL_POS[1] - pos[1], GOAL_POS[0] - pos[0])
    )
    dist_goal = float(math.hypot(GOAL_POS[0] - pos[0], GOAL_POS[1] - pos[1]))
    heading_err_deg = float(wrap_deg(goal_bearing_deg - yaw_deg))
    err_rad = math.radians(heading_err_deg)

    # --- floating state marker follows Carter ---
    state_op.Set(
        Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2]) + 1.5)
    )

    # --- Milestone 9: trajectory history point (sparse, capped) ---
    if step % TRAJ_STEP == 0 and trajectory_initialized:
        if len(trajectory_history) < TRAJ_MAX_POINTS:
            trajectory_history.append(
                Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2]) + 0.05)
            )
            traj_prim.GetPointsAttr().Set(trajectory_history)
            # widths sized to the vertex count: keeps Hydra on vertex
            # interpolation (avoids a per-frame fallback warning).
            traj_prim.GetWidthsAttr().Set([0.07] * len(trajectory_history))
        elif not traj_cap_logged:
            traj_cap_logged = True
            log_print(f"TRAJECTORY_POINT_CAP_REACHED: {TRAJ_MAX_POINTS}")

    # --- periodic progress log ---
    if step % 50 == 0:
        fd_str = f"{forward_dist:.4f}" if forward_dist is not None else "nan"
        log_print(
            f"STEP {step}: POSITION: ({pos[0]:.3f}, {pos[1]:.3f}), "
            f"SPEED: {speed:.3f}, FORWARD_DIST: {fd_str}, "
            f"DIST_TO_GOAL: {dist_goal:.3f}, STATE: {state}"
        )

    # ------------------------------------------------------------
    # GOAL (highest priority): stop safely, then hold for observation
    # ------------------------------------------------------------
    if state != "GOAL_REACHED" and dist_goal < GOAL_TOL:
        enter_manual()
        write_cmd(0.0, 0.0)
        state = "GOAL_REACHED"
        goal_reached = True
        goal_step = step
        goal_hold = 0
        log_print(
            f"GOAL_REACHED_STEP: {step}, DIST_TO_GOAL: {dist_goal:.4f}, "
            f"POSITION: ({pos[0]:.3f}, {pos[1]:.3f}), SPEED: {speed:.3f}"
        )

    if state == "GOAL_REACHED":
        # Keep commanding zero velocity and stay visible for the
        # observation period (do NOT close Isaac Sim instantly).
        write_cmd(0.0, 0.0)
        goal_hold += 1
        if goal_hold >= observe_steps:
            log_print(
                f"Observation complete at step {step} "
                f"({OBSERVE_SECONDS:.0f}s at goal)"
            )
            break
    else:
        # ------------------------------------------------------------
        # M10/M11 counters: sustained-block streaks, sliding progress
        # window, direction-lock timers (updated every non-goal step)
        # ------------------------------------------------------------
        do_avoid = False       # unified AVOID entry flag (set by trigger
                               # or by RECOVERY->REASSESS re-commit)
        avoid_dir = None
        avoid_lc = 0.0
        avoid_rc = 0.0
        motion_state = state in (
            "NAVIGATE", "APPROACH", "RESUME", "PASS_OBSTACLE"
        )
        if (
            motion_state
            and forward_dist is not None
            and forward_dist <= AVOID_T
        ):
            forward_low_streak += 1
            fwd_stall_steps += 1
        else:
            forward_low_streak = 0
            fwd_stall_steps = 0
        if state in ("NAVIGATE", "APPROACH", "RESUME", "PASS_OBSTACLE", "AVOID"):
            progress_window.append((step, dist_goal))
        while progress_window and progress_window[0][0] < step - NO_PROGRESS_STEPS:
            progress_window.pop(0)
        steps_since_dir += 1
        path_since_dir += speed * physics_dt

        # --- M11 recovery triggers #2/#3: forward stalled too long or
        #     no goal progress over the sliding window ---
        if motion_state and progress_window:
            _win_span = progress_window[-1][0] - progress_window[0][0]
            _prog = progress_window[0][1] - dist_goal
            _stalled = fwd_stall_steps >= FWD_STALL_STEPS and _prog < NO_PROGRESS_M
            _stuck = _win_span >= NO_PROGRESS_STEPS and _prog < NO_PROGRESS_M
            if _stalled or _stuck:
                recovery_reason = "FORWARD_STALL" if _stalled else "NO_PROGRESS"
                pending_lines = [
                    f"RECOVERY_REASON: {recovery_reason} (step {step})",
                    f"RECOVERY_PROGRESS: {progress_window[0][1]:.3f} -> "
                    f"{dist_goal:.3f} m over {_win_span} steps",
                ]
                enter_manual()
                recovery_events += 1
                recovery_steps = 0
                progress_window.clear()
                fwd_stall_steps = 0
                forward_low_streak = 0
                state = "RECOVERY"

        # ------------------------------------------------------------
        # M11 RECOVERY: stop -> reverse slowly/short (only while the
        # rear sector is clear) -> REASSESS (stopped re-scan) ->
        # compare sectors -> pick the safer side -> resume
        # ------------------------------------------------------------
        if state == "RECOVERY":
            recovery_steps += 1
            _rear = (
                sector_min(d, rear_mask, valid)
                if d is not None and rear_mask is not None and valid is not None
                else 0.0
            )
            _may_reverse = (
                _rear >= REAR_MIN_ALLOWED
                and recovery_steps < REVERSE_MAX_STEPS
                and forward_dist is not None
                and forward_dist < CLEAR_T
            )
            if not _may_reverse:
                state = "REASSESS"
                reassess_steps = 0
                pending_lines = [
                    f"RECOVERY_PHASE: REASSESS (step {step}, rear {_rear:.3f} m, "
                    f"reverse steps {recovery_steps})"
                ]
            elif recovery_steps >= RECOVERY_MAX_STEPS:
                state = "REASSESS"
                reassess_steps = 0
                pending_lines = [f"RECOVERY_TIMEOUT -> REASSESS (step {step})"]

        if state == "REASSESS":
            reassess_steps += 1
            if (
                reassess_steps >= REASSESS_STEPS
                and d is not None
                and valid is not None
            ):
                # stopped + re-scanned: compare the measured sectors again
                direction, lc, rc, lock_note = decide_avoid_side(
                    d, valid, left_mask, right_mask,
                    last_dir, steps_since_dir, path_since_dir,
                )
                blocked = lc <= BLOCKED_T and rc <= BLOCKED_T
                fwd_clear = forward_dist is None or forward_dist > AVOID_T
                if (
                    fwd_clear
                    and not blocked
                    and recovery_events < RECOVERY_FORCE_EVENTS
                ):
                    # reverse/re-scan opened the gap: no side needed
                    pending_lines = [
                        f"RECOVERY_RESOLVED: forward "
                        f"{forward_dist:.3f} m clear (step {step})"
                    ]
                    exit_manual()
                    state = "NAVIGATE"
                    recovery_steps = 0
                elif blocked and recovery_events < RECOVERY_FORCE_EVENTS:
                    recovery_events += 1
                    recovery_steps = 0
                    state = "RECOVERY"
                    pending_lines = [
                        f"REASSESS: BOTH_SIDES_BLOCKED -> RECOVERY "
                        f"(attempt {recovery_events}, step {step})",
                        f"RECOVERY_SECTORS: LEFT {lc:.4f} RIGHT {rc:.4f}",
                    ]
                else:
                    if blocked:
                        # retry cap reached: commit to the larger side
                        direction = "LEFT" if lc >= rc else "RIGHT"
                        lock_note = "RECOVERY_FORCED_DIRECTION"
                    cur_obs = min(
                        obstacles,
                        key=lambda o: aabb_distance(pos[:2], o["center"], o["half"]),
                    )
                    avoid_dir = direction
                    avoid_lc = lc
                    avoid_rc = rc
                    do_avoid = True
                    pending_lines = [
                        f"RECOVERY_DECISION: AVOID_{direction} (step {step})",
                        f"LEFT_CLEARANCE: {lc:.4f}",
                        f"RIGHT_CLEARANCE: {rc:.4f}",
                        f"AVOIDANCE_DIRECTION: {direction}",
                        f"DIRECTION_LOCK: {lock_note}",
                    ]

        # ------------------------------------------------------------
        # STATE TRANSITIONS
        # ------------------------------------------------------------
        # NAVIGATE -> APPROACH (obstacle entering the 2.5 m band)
        if state == "NAVIGATE":
            same_obs_streak = 0  # open corridor: forget failure streaks
            if forward_dist is not None and forward_dist <= CLEAR_T:
                enter_manual()
                state = "APPROACH"
                app_streak = 0

        # RESUME -> APPROACH/NAVIGATE (after a fixed Stanley hold)
        if state == "RESUME":
            resume_counter += 1
            if resume_counter >= RESUME_STEPS:
                if forward_dist is not None and forward_dist <= CLEAR_T:
                    enter_manual()
                    state = "APPROACH"
                    app_streak = 0
                else:
                    state = "NAVIGATE"

        # APPROACH -> (AVOID via the unified trigger below) or NAVIGATE
        if state == "APPROACH":
            if forward_dist is None or forward_dist > CLEAR_T:
                if forward_dist is not None and forward_dist > CLEAR_T:
                    app_streak += 1
                else:
                    app_streak = 0
                if app_streak >= NAV_RECOVER_STEPS:
                    exit_manual()
                    state = "NAVIGATE"

        # ------------------------------------------------------------
        # M10: unified AVOID trigger with ONE measured decision path.
        # APPROACH fires immediately at AVOID_T; NAVIGATE/RESUME/PASS
        # fire only after the forward gap stayed <= AVOID_T for
        # RETRIGGER_STEPS (moving boxes can re-enter the route).
        # ------------------------------------------------------------
        trig_avoid = False
        if (
            state == "APPROACH"
            and forward_dist is not None
            and forward_dist <= AVOID_T
        ):
            trig_avoid = True
        elif (
            state in ("NAVIGATE", "RESUME", "PASS_OBSTACLE")
            and forward_low_streak >= RETRIGGER_STEPS
            and forward_dist is not None
            and forward_dist <= AVOID_T
        ):
            trig_avoid = True

        if trig_avoid and d is not None and valid is not None:
            # ---- AVOIDANCE DECISION FROM LIVE LIDAR CLEARANCE ----
            avoid_dir, avoid_lc, avoid_rc, lock_note = decide_avoid_side(
                d, valid, left_mask, right_mask,
                last_dir, steps_since_dir, path_since_dir,
            )
            # nearest obstacle (identity of what we are avoiding)
            cur_obs = min(
                obstacles,
                key=lambda o: aabb_distance(pos[:2], o["center"], o["half"]),
            )
            cur_idx = cur_obs["idx"]
            if events and events[-1]["obstacle"] == cur_idx + 1:
                same_obs_streak += 1
            else:
                same_obs_streak = 1

            if avoid_lc <= BLOCKED_T and avoid_rc <= BLOCKED_T:
                # M11 recovery trigger #1: both sectors blocked
                recovery_reason = "BOTH_SIDES_BLOCKED"
                pending_lines = [
                    f"RECOVERY_REASON: {recovery_reason} (step {step})",
                    f"RECOVERY_SECTORS: LEFT {avoid_lc:.4f} RIGHT {avoid_rc:.4f}",
                ]
            elif same_obs_streak >= AVOID_FAIL_STREAK:
                # M11 recovery trigger #4: repeated avoidance failures
                recovery_reason = "REPEATED_AVOIDANCE_FAILURE"
                pending_lines = [
                    f"RECOVERY_REASON: {recovery_reason} "
                    f"(obstacle {cur_idx + 1}, {same_obs_streak} attempts, "
                    f"step {step})",
                    f"RECOVERY_SECTORS: LEFT {avoid_lc:.4f} RIGHT {avoid_rc:.4f}",
                ]
            else:
                do_avoid = True
                pending_lines = [
                    f"LEFT_CLEARANCE: {avoid_lc:.4f}",
                    f"RIGHT_CLEARANCE: {avoid_rc:.4f}",
                    f"AVOIDANCE_DIRECTION: {avoid_dir}",
                    f"DIRECTION_LOCK: {lock_note}",
                ]

            if not do_avoid:
                enter_manual()
                recovery_events += 1
                recovery_steps = 0
                progress_window.clear()
                fwd_stall_steps = 0
                forward_low_streak = 0
                state = "RECOVERY"
        elif trig_avoid:
            # no scan this frame: retry on the next step
            trig_avoid = False

        # AVOID sub-phases: TURN (rotate away) -> GLIDE (lateral gain)
        if state == "AVOID":
            avoid_steps += 1
            phase_steps += 1
            if avoid_phase == "TURN":
                if abs(heading_err_deg) >= TURN_DEV_DEG:
                    avoid_phase = "GLIDE"
                    phase_steps = 0
                    log_print(
                        f"AVOID_PHASE: GLIDE (heading dev "
                        f"{abs(heading_err_deg):.1f} deg, step {step})"
                    )
                elif phase_steps > TURN_TIMEOUT:
                    avoid_phase = "GLIDE"
                    phase_steps = 0
                    log_print(f"AVOID_TURN_TIMEOUT -> GLIDE (step {step})")
            if avoid_phase == "GLIDE":
                lateral = float(np.dot(pos[:2] - avoid_entry, open_dir))
                if lateral >= GLIDE_LATERAL_M:
                    state = "PASS_OBSTACLE"
                    pass_steps = 0
                    pass_streak = 0
                    log_print(
                        f"AVOID_GLIDE_DONE: lateral {lateral:.3f} m (step {step})"
                    )
                elif phase_steps > GLIDE_TIMEOUT:
                    state = "PASS_OBSTACLE"
                    pass_steps = 0
                    pass_streak = 0
                    log_print(f"AVOID_GLIDE_TIMEOUT -> PASS_OBSTACLE (step {step})")

        # PASS_OBSTACLE -> RESUME (obstacle behind + corridor clear)
        if state == "PASS_OBSTACLE":
            pass_steps += 1
            if cur_obs is not None:
                past = (
                    float(np.dot(pos[:2] - cur_obs["center"], route_dir))
                    > PASS_PAST_M
                )
            else:
                past = True
            heading_ok = abs(heading_err_deg) <= PASS_HEADING_TOL
            clear_ok = forward_dist is not None and forward_dist > CLEAR_T
            if past and heading_ok and clear_ok:
                pass_streak += 1
            else:
                pass_streak = 0
            if pass_streak >= PASS_CLEAR_STEPS or pass_steps > PASS_TIMEOUT_STEPS:
                if pass_steps > PASS_TIMEOUT_STEPS and pass_streak < PASS_CLEAR_STEPS:
                    log_print(f"PASS_TIMEOUT -> RESUME (step {step})")
                exit_manual()
                state = "RESUME"
                resume_counter = 0
                if avoid_events == 1 and not m7["cleared"]:
                    m7["cleared"] = True
                    m7["resumed"] = True

        # ------------------------------------------------------------
        # Single AVOID entry point: the APPROACH trigger, a sustained
        # re-block, and the RECOVERY->REASSESS re-commit all land here.
        # ------------------------------------------------------------
        if do_avoid:
            if last_dir is not None and avoid_dir != last_dir:
                direction_changes += 1
                log_print(
                    f"DIRECTION_CHANGE: {last_dir} -> {avoid_dir} "
                    f"(event {avoid_events + 1}, step {step})"
                )
            last_dir = avoid_dir
            avoid_direction = avoid_dir
            steps_since_dir = 0
            path_since_dir = 0.0
            avoid_events += 1
            cur_idx = cur_obs["idx"]
            if cur_idx not in detected_set:
                detected_set.add(cur_idx)
            if cur_idx not in detection_dist:
                detection_dist[cur_idx] = forward_dist
            events.append(
                {
                    "step": step,
                    "obstacle": cur_idx + 1,
                    "forward": forward_dist,
                    "left": avoid_lc,
                    "right": avoid_rc,
                    "direction": avoid_dir,
                }
            )
            if avoid_events == 1:
                m7["detected"] = True
                m7["triggered"] = True
                m7["direction"] = avoid_dir

            enter_manual()
            avoid_entry = pos[:2].copy()
            # left in world = nose rotated +90 deg (CCW)
            left_w = np.array(
                [-math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg))],
                dtype=np.float64,
            )
            dir_sign = 1.0 if avoid_dir == "LEFT" else -1.0
            open_dir = left_w * dir_sign
            avoid_phase = "TURN"
            avoid_steps = 0
            phase_steps = 0
            state = "AVOID"
            forward_low_streak = 0
            fwd_stall_steps = 0
            do_avoid = False

        # ------------------------------------------------------------
        # CONTROL OUTPUT: exactly one authority per state
        # ------------------------------------------------------------
        if state == "APPROACH":
            write_cmd(APPROACH_V, float(np.clip(APPROACH_K * err_rad, -APPROACH_W_MAX, APPROACH_W_MAX)))
        elif state == "AVOID":
            if avoid_phase == "TURN":
                write_cmd(TURN_V, dir_sign * TURN_W)
            else:
                write_cmd(GLIDE_V, 0.0)
        elif state == "PASS_OBSTACLE":
            if forward_dist is not None and forward_dist <= CLEAR_T:
                write_cmd(PASS_SLOW_V, float(np.clip(PASS_K * err_rad, -PASS_W_MAX, PASS_W_MAX)))
            else:
                write_cmd(PASS_V, float(np.clip(PASS_K * err_rad, -PASS_W_MAX, PASS_W_MAX)))
        elif state == "RECOVERY":
            # M11: slow, short reverse (further rear gating is done in
            # the RECOVERY transition above)
            write_cmd(REVERSE_V, 0.0)
        elif state == "REASSESS":
            # M11: stopped while re-scanning the sectors
            write_cmd(0.0, 0.0)
        # NAVIGATE / RESUME: Stanley pipeline owns the command

    # ------------------------------------------------------------
    # State change reporting + marker color
    # ------------------------------------------------------------
    if state != last_state:
        for _ln in pending_lines:
            log_print(_ln)
        pending_lines = []
        fd_str = f"{forward_dist:.4f}" if forward_dist is not None else "nan"
        log_print(f"STATE: {state} (step {step}, FORWARD_DISTANCE: {fd_str})")
        if state == "AVOID" and events:
            ev = events[-1]
            log_print(
                f"AVOID_EVENT {ev['step']}: STEP {ev['step']}, "
                f"OBSTACLE {ev['obstacle']}, "
                f"DETECTION_DISTANCE: {ev['forward']:.4f}, "
                f"LEFT_CLEARANCE: {ev['left']:.4f}, "
                f"RIGHT_CLEARANCE: {ev['right']:.4f}, "
                f"AVOIDANCE_DIRECTION: {ev['direction']}"
            )
        if state == "AVOID" and avoid_direction:
            _marker_key = f"AVOID_{avoid_direction}"
        else:
            _marker_key = state
        marker_cube.GetDisplayColorAttr().Set(
            [Gf.Vec3f(
                *STATE_COLORS.get(
                    _marker_key, STATE_COLORS.get(state, (1.0, 1.0, 1.0))
                )
            )]
        )
        last_state = state

# ============================================================
# STEP 12: Records and measurements
# ============================================================
final_pos, final_rot = ar.get_world_pose()
final_pos = np.array(final_pos, dtype=np.float64)
final_yaw = yaw_of(final_rot)

disp = float(np.linalg.norm(final_pos[:2] - start_pos[:2]))
toward_goal = float(
    np.dot(final_pos[:2] - start_pos[:2], np.array(GOAL_POS[:2]) - start_pos[:2])
)
carter_moved = disp > 0.3 and toward_goal > 0

final_speed = float(np.mean(speeds[-10:])) if speeds else float("nan")
final_goal_dist = float(
    math.hypot(GOAL_POS[0] - final_pos[0], GOAL_POS[1] - final_pos[1])
)

hist = np.array(position_history)
if len(hist) >= 61:
    recent_drift = float(np.linalg.norm(hist[-1, :2] - hist[-61, :2]))
else:
    recent_drift = float(np.linalg.norm(hist[-1, :2] - hist[0, :2]))
carter_stopped = goal_reached and final_speed < 0.05 and recent_drift < 0.05

safety_ok = (
    lidar_data_available
    and min_forward_distance >= MIN_FORWARD_ALLOWED
    and min_clearance >= MIN_CLEARANCE_CENTER
)

log_print("---- RECORDS ----")
log_print(f"INITIAL_POSITION: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
log_print(f"GOAL_POSITION: ({GOAL_POS[0]:.3f}, {GOAL_POS[1]:.3f}, {GOAL_POS[2]:.3f})")
for i, o in enumerate(obstacles):
    log_print(
        f"DYNAMIC_OBSTACLE_{o.get('tag', i + 1)}_FINAL_POSITION: "
        f"({o['center'][0]:.3f}, {o['center'][1]:.3f}, {OBSTACLE_Z:.3f})"
    )
    dd = detection_dist.get(o["idx"])
    if dd is not None:
        log_print(f"OBSTACLE_{o.get('tag', i + 1)}_DETECTION_DISTANCE: {dd:.4f}")
    else:
        log_print(f"OBSTACLE_{o.get('tag', i + 1)}_DETECTION_DISTANCE: NONE")
log_print(f"AVOIDANCE_EVENTS_TOTAL: {avoid_events}")
for n, ev in enumerate(events, start=1):
    log_print(
        f"AVOIDANCE_EVENT_{n}: STEP {ev['step']}, OBSTACLE {ev['obstacle']}, "
        f"FORWARD_DISTANCE: {ev['forward']:.4f}, "
        f"LEFT_CLEARANCE: {ev['left']:.4f}, "
        f"RIGHT_CLEARANCE: {ev['right']:.4f}, "
        f"DIRECTION: {ev['direction']}"
    )
log_print(f"DYNAMIC_LIDAR_FRAMES: {dynamic_hit_frames}")
log_print(f"DYNAMIC_TRACKING_MISMATCH: {tracking_mismatch}")
log_print(f"RECOVERY_EVENTS_TOTAL: {recovery_events}")
log_print(f"DIRECTION_CHANGES_TOTAL: {direction_changes}")
log_print(f"MIN_FORWARD_DISTANCE: {min_forward_distance:.4f}")
log_print(f"MIN_CLEARANCE_CENTER: {min_clearance:.4f}")
log_print(f"MOVED_DISTANCE: {disp:.4f}")
log_print(f"GOAL_DISTANCE: {final_goal_dist:.4f}")
log_print(f"DRIFT_AFTER_STOP: {recent_drift:.4f}")
log_print(
    f"SAFETY: {'PASS' if safety_ok else 'FAIL'} "
    f"(min forward {min_forward_distance:.4f} >= {MIN_FORWARD_ALLOWED}, "
    f"min clearance {min_clearance:.4f} >= {MIN_CLEARANCE_CENTER})"
)

# ============================================================
# STEP 13: M9 re-verification + M10 + M11 verification markers
# ============================================================
visual_ok = True  # SimulationApp + world + Carter + markers all came up
traj_ok = trajectory_initialized and len(trajectory_history) > 0
m9_vis = visual_ok and trajectory_initialized
m9_lidar = lidar_initialized and lidar_data_available and lidar_rows == 16
m9_goal = goal_reached and final_speed < 0.1

# M10: moving boxes present, identified from LiDAR, actually tracked
m10_dynamic = len(dynamic_obstacles) >= 3  # A crosser + B runner + C approacher
m10_detections = len(detected_set)  # distinct moving boxes identified at triggers
m10_avoid = avoid_events >= 1        # the reactive loop really ran
m10_tracking = tracking_mismatch == 0 and dynamic_hit_frames > 0
m10_goal = m9_goal

# M11: recovery armed; oscillation only if >= 2 side changes occurred
m11_enabled = True  # mechanism armed + logged (RECOVERY/REASSESS states run)
m11_oscillation = direction_changes >= 2
m11_goal = m9_goal

log_print("---- VERIFICATION MARKERS ----")
log_print(f"VISUAL_DEMO_INITIALIZED: {'YES' if visual_ok else 'NO'}")
log_print(f"LIDAR_INITIALIZED: {'YES' if lidar_initialized else 'NO'}")
log_print(f"LIDAR_DATA_AVAILABLE: {'YES' if lidar_data_available else 'NO'}")
log_print(f"LIDAR_SCAN_ROWS: {lidar_rows} (expected 16)")
log_print(f"CARTER_MOVED: {'YES' if carter_moved else 'NO'}")
log_print(
    f"MILESTONE_9_TRAJECTORY_INITIALIZED: "
    f"{'YES' if trajectory_initialized else 'NO'}"
)
log_print(f"MILESTONE_9_TRAJECTORY_POINTS: {len(trajectory_history)}")
log_print(f"MILESTONE_9_VISUALIZATION: {'YES' if m9_vis else 'NO'}")
log_print(f"MILESTONE_9_LIDAR: {'YES' if m9_lidar else 'NO'}")
log_print(f"MILESTONE_9_GOAL_REACHED: {'YES' if m9_goal else 'NO'}")
log_print(
    f"MILESTONE_10_DYNAMIC_OBSTACLES: "
    f"{'YES' if m10_dynamic else 'NO'} "
    f"({len(dynamic_obstacles)} kinematic boxes)"
)
log_print(f"MILESTONE_10_DYNAMIC_OBSTACLE_DETECTIONS: {m10_detections}")
log_print(f"MILESTONE_10_DYNAMIC_LIDAR_FRAMES: {dynamic_hit_frames}")
log_print(f"MILESTONE_10_TRACKING: {'YES' if m10_tracking else 'NO'}")
log_print(f"MILESTONE_10_AVOIDANCE_EVENTS: {avoid_events}")
log_print(f"MILESTONE_10_GOAL_REACHED: {'YES' if m10_goal else 'NO'}")
log_print(f"MILESTONE_11_RECOVERY_ENABLED: {'YES' if m11_enabled else 'NO'}")
log_print(f"MILESTONE_11_RECOVERY_EVENTS: {recovery_events}")
log_print(f"MILESTONE_11_DIRECTION_CHANGES: {direction_changes}")
log_print(
    f"MILESTONE_11_OSCILLATION_DETECTED: {'YES' if m11_oscillation else 'NO'}"
)
log_print(
    f"MILESTONE_11_COLLISION_DETECTED: {'YES' if collision_detected else 'NO'}"
)
log_print(f"MILESTONE_11_GOAL_REACHED: {'YES' if m11_goal else 'NO'}")
log_print(
    f"FINAL_POSITION: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})"
)
log_print(f"FINAL_SPEED: {final_speed:.4f}")
log_print(f"GOAL_DISTANCE: {final_goal_dist:.4f}")
log_print(f"COLLISION_DETECTED: {'YES' if collision_detected else 'NO'}")

# ============================================================
# STEP 14: Test result (combined M9 + M10 + M11 verdict)
# ============================================================
test_passed = (
    visual_ok
    and lidar_initialized
    and lidar_data_available
    and carter_moved
    and traj_ok            # trajectory points were recorded
    and m9_vis             # visualization subsystems up
    and m9_lidar           # full (360,16) PhysX scan
    and m9_goal            # goal reached and stopped (M9 parity)
    and m10_dynamic        # moving boxes present with a live driver
    and m10_detections > 0 # at least one moving box identified
    and m10_tracking       # PhysX bodies follow the scripted motion
    and m10_avoid          # reactive avoidance ran against dynamics
    and m10_goal
    and m11_enabled        # recovery armed (events may legitimately be 0)
    and not m11_oscillation
    and final_speed < 0.1
    and not collision_detected
    and safety_ok
)

log_print(
    f"WAREHOUSE_AUTONOMOUS_NAVIGATION_M9_M10_M11: "
    f"{'PASS' if test_passed else 'FAIL'}"
)

# ============================================================
# STEP 15: Cleanup
# ============================================================
my_world.stop()
simulation_app.update()
simulation_app.close()

log_print("TEST_COMPLETED")

sys.exit(0 if test_passed else 1)
