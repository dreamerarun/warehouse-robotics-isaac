#!/usr/bin/env python3
"""Milestones 15+16+17 - Multi-Carter Fleet + Multi-Robot LiDAR Collision
Avoidance + Basic Task Allocation (Isaac Sim 5.1, VISUAL)

Forks the FROZEN, VISIBLY VERIFIED M12/M13/M14 script
(scripts/test_carter_realistic_warehouse.py, NOT modified) and generalizes
its single-Carter control stack into THREE independent Carter AMRs sharing
ONE warehouse. Carter init, PhysX LiDAR, scan interpretation, thresholds
and the reactive state machine are reused unchanged (one CarterAgent
instance per robot, no navigation rewrite, no external planner).

M15 - MULTIPLE CARTER AMRs (three robots, one shared warehouse)
    - Three NovaCarters from the proven USD
      /Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd at
      /World/Carter_1..3, each with its OWN articulation, LiDAR, pose,
      goal, state machine, trajectory trail and floating state marker.
    - ONE shared warehouse (walls, rack rows S/M/N, corridor, staging,
      loading, charging, restricted zone, pallets, forklift, human).
    - Each robot runs the same lightweight PhysX LiDAR (360 deg scan,
      30 deg vertical FOV, 1 deg azimuth, 2 deg vertical resolution,
      high_lod=True -> full 360x16 scan) with per-robot ray masks.
    - Suggested starts: Carter_1 (-1.5,-6.6), Carter_2 (2.0,-6.6),
      Carter_3 (-0.5,-9.0); each faces its first task goal and every
      pair starts with a clearly positive body-footprint gap.

M16 - MULTI-ROBOT LiDAR COLLISION AVOIDANCE (decentralized, reactive)
    - Routes create a DETERMINISTIC interaction: Carter_2 (east box ->
      staging) drives west through the corridor intersection at
      (x ~ 1.2, y ~ -7) exactly while Carter_3 (south box -> north
      aisle) crosses it northbound. Carter_2 detects Carter_3 with its
      LiDAR, slows (APPROACH, forward safety zone <= 2.5 m), stops
      (emergency zone <= 1.5 m measured clearance), Carter_3 continues,
      the intersection clears, Carter_2 resumes (NAVIGATE). A second
      detection happens when Carter_3 starts nose-to-nose with
      Carter_2 and yields the same way.
    - Every robot sees static obstacles, the forklift, the human and
      the other Carters through ITS OWN LiDAR (no cameras, no tracker,
      no privileged robot state: the yield decision uses only the
      measured forward-cone clearance and the measured sector
      clearances).
    - Yielding is NEVER hard-coded: which robot slows/stops is decided
      purely by which robot measures the other inside its forward cone.
      Avoidance side selection stays the proven measured
      LEFT/RIGHT sector comparison (decide_avoid_side).
    - Priority labeling only (no priority planner): HUMAN >
      FORKLIFT > OTHER CARTER > STATIC, printed with each event.
    - NO ORCA/RVO/MPC/RL/centralized solution: purely the proven
      per-robot reactive state machine.

M17 - BASIC TASK ALLOCATION (lightweight greedy task manager)
    - Tasks are dicts (id, pickup, delivery, status, assigned_robot);
      robots are dicts (id, position, state, current_task).
    - Greedy nearest-available-robot-to-pickup allocation in task
      order (NO Hungarian/RL/optimization/cloud):
      TASK_001 staging -> loading  => Carter_1
      TASK_002 east box -> staging  => Carter_2
      TASK_003 south box -> north aisle => Carter_3
      (TASK_001/TASK_002 follow the spec's example tasks exactly;
      TASK_003's example "charging" delivery point is replaced by a
      north-aisle delivery - the example task list is explicitly
      optional and the crossing route is what creates M16's required
      deterministic interaction. Markers only check ASSIGNED/
      COMPLETED.)
    - State flow per robot:
      IDLE -> TASK_ASSIGNED -> GO_TO_PICKUP -> PICKUP -> (dwell)
              -> GO_TO_DELIVERY -> DELIVERY -> (dwell)
              -> TASK_COMPLETE -> IDLE
      Pickup/delivery are logical (state-based, no arm manipulation).
    - Visible assignment: per-task colored pickup/delivery floor
      markers + floating per-robot state marker color (nav state or
      task phase) + log lines. No UI work.

VISUAL MODE: headless=False by default, window 1280x720, lightweight
renderer settings for the RTX 4050 6 GB (NO cameras, NO RTX LiDAR, NO
heavy meshes, NO crowds, NO ROS2/Nav2/SLAM/A*/RL/Orbit/extra
extensions: exactly 3 Carters, 3 PhysX LiDARs, 1 forklift, 1 human,
simple USD primitives). Set HEADLESS=1 for headless runs.

Proven pipelines reused verbatim (all ten verified scripts frozen):
- Init / floor / PhysX LiDAR (RotatingLidarPhysX, fov 360x30,
  resolution 1x2 deg -> scan (360,16), range 0.4-50 m, high_lod=True).
- Navigation OmniGraph per robot: QuinticPathPlanner -> CheckGoal2D ->
  StanleyControlPID -> DifferentialController ->
  IsaacArticulationController + odometry (one graph per robot,
  /ActionGraph_1..3).
- Reactive avoidance state machine (NAVIGATE/APPROACH/AVOID/
  PASS_OBSTACLE/RESUME/RECOVERY/REASSESS/GOAL_REACHED, anti-
  oscillation direction lock, progress detection) + trajectory trail
  + state marker + LiDAR debug draw.

Success markers:
    MILESTONE_15_MULTI_AMR: YES
    CARTER_1_INITIALIZED: YES
    CARTER_1_LIDAR: YES
    CARTER_1_MOVED: YES
    CARTER_2_INITIALIZED: YES
    CARTER_2_LIDAR: YES
    CARTER_2_MOVED: YES
    CARTER_3_INITIALIZED: YES
    CARTER_3_LIDAR: YES
    CARTER_3_MOVED: YES
    MILESTONE_16_MULTI_ROBOT_AVOIDANCE: YES
    MULTI_ROBOT_DETECTED: YES
    MULTI_ROBOT_REACTION: YES
    MULTI_ROBOT_CLEARANCE: YES
    CARTER_1_COLLISION: NO
    CARTER_2_COLLISION: NO
    CARTER_3_COLLISION: NO
    MULTI_ROBOT_COLLISION: NO
    MILESTONE_17_TASK_ALLOCATION: YES
    TASK_001_ASSIGNED: YES
    TASK_001_COMPLETED: YES
    TASK_002_ASSIGNED: YES
    TASK_002_COMPLETED: YES
    TASK_003_ASSIGNED: YES
    TASK_003_COMPLETED: YES
    ALL_TASKS_COMPLETED: YES
    FLEET_DEMO: PASS

Usage:
    cd /home/arun/warehouse_robotics_isaac
    ISAAC=/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64
    PYTHONUNBUFFERED=1 HEADLESS=0 $ISAAC/python.sh \
        scripts/test_multi_carter_fleet.py
"""

import sys
import os
import math
import random
import numpy as np

# Write start marker (single writer: the shell may tee stdout only)
with open("/tmp/multi_carter_fleet_results.txt", "w") as f:
    f.write("TEST_STARTED\n")


def log_print(msg):
    """Print to console and write to file"""
    print(msg)
    with open("/tmp/multi_carter_fleet_results.txt", "a") as f:
        f.write(msg + "\n")


# ============================================================
# STEP 1: Start Isaac Sim 5.1 - VISIBLE by default
# ============================================================
from isaacsim import SimulationApp

HEADLESS = os.environ.get("HEADLESS", "0") == "1"

sim_config = {
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
# STEP 2b: PHASE 3A - ROS 2 BRIDGE EXTENSION (Carter 1 only)
# ------------------------------------------------------------
# This file is the Phase-3 working copy of the Phase-2 PASS baseline
# (scripts/test_multi_carter_fleet.py, left byte-identical).  The
# bridge is enabled HERE - before the stage exists, before World() is
# built and before my_world.play() - so extension startup adds ZERO
# physics ticks to the verified Phase-2 timeline.
#
# Nothing in Phase 3A writes to the stage, the articulations, the
# LiDAR sensors, the collision boxes, the safety thresholds, the task
# coordinates or /ActionGraph_1..3.  Everything ROS 2 is READ-ONLY
# with respect to the fleet (see STEP 8b).
#
#   PHASE3_ROS2=0  -> disables the whole block (A/B control run)
#   PHASE3_ROS2_HOLD_S -> extra seconds with the timeline PLAYING
#                         after all markers, for external
#                         `ros2 topic echo` verification.
# ============================================================
import omni.kit.app

PHASE3_ROS2 = os.environ.get("PHASE3_ROS2", "1") == "1"
PHASE3_ROS2_HOLD_S = float(os.environ.get("PHASE3_ROS2_HOLD_S", "0"))

# ------------------------------------------------------------------
# PHASE 3C - three-Carter command handover gate
#   PHASE3C_HANDOVER=0 -> the Phase-2 control path stays active for ALL
#                         three Carters; the three ROS2SubscribeTwist
#                         nodes exist but never drive anything.
#   PHASE3C_HANDOVER=1 -> /carter{1,2,3}/cmd_vel own THAT Carter's
#                         DifferentialController input, independently.
# ROS2_HANDOVER_GRAPHS holds the OmniGraph paths (/ActionGraph_1..3)
# whose command input is owned by ROS 2.  It is empty for every robot
# when the gate is 0, so enter_manual()/exit_manual() behave exactly as
# they do in Phase 2.  A Carter is added to it ONLY when the first Twist
# addressed to THAT Carter arrives - PHASE3C_HANDOVER=1 alone arms the
# three wires and never engages any of them (the verified Phase-3B rule,
# now applied per robot).  Engaging one Carter never touches another
# Carter's graph, because every Carter has its own graph_path.
# ------------------------------------------------------------------
PHASE3C_HANDOVER = os.environ.get("PHASE3C_HANDOVER", "0") == "1"
ROS2_HANDOVER_GRAPHS = set()

ROS2_EXT_ID = "isaacsim.ros2.bridge"     # Isaac Sim 5.1 name (4.12.4)
ROS2_ROBOTS = ("carter1", "carter2", "carter3")   # Phase 3C: all three
ROS2_TAGS = ("CARTER_1", "CARTER_2", "CARTER_3")
# Exactly 12 required topics: 4 per Carter x 3 Carters.
ROS2_TOPIC_TYPES = tuple(
    (f"/{ns}/{sfx}", mtype, direction)
    for ns in ROS2_ROBOTS
    for sfx, mtype, direction in (
        ("cmd_vel", "geometry_msgs/msg/Twist", "subscribe"),
        ("odom", "nav_msgs/msg/Odometry", "publish"),
        ("scan", "sensor_msgs/msg/LaserScan", "publish"),
        ("tf", "tf2_msgs/msg/TFMessage", "publish"),
    )
)
ROS2_MONITORED = tuple(t for t, _, _ in ROS2_TOPIC_TYPES)
ROS2_SENSOR_TOPICS = tuple(
    f"/{ns}/{sfx}" for ns in ROS2_ROBOTS for sfx in ("odom", "scan", "tf")
)
ROS2_CMD_TOPICS = tuple(f"/{ns}/cmd_vel" for ns in ROS2_ROBOTS)

ros2 = {
    "requested": PHASE3_ROS2,
    "ext_ok": False,
    "graphs": [],         # [/ROS2ActionGraph_1, _2, _3] actually built
    "enabled": False,     # True only when ALL THREE graphs were built
    "ctx": None,          # private rclpy Context (the extension already
                          # used the default one once -> cannot re-init)
    "node": None,         # in-process monitor node (12 subscriptions)
    "executor": None,
    "counts": {},         # topic -> messages actually received
    "first": {},          # topic -> description of the first message
    "spin_errors": 0,
    "errors": [],
    # ------------------------------------------------------------------
    # PER-CARTER state.  Everything that can move a robot lives here and
    # is indexed by robot, so /carter2/cmd_vel can only ever reach
    # agents[2]'s DifferentialController and can never move Carter 1 or
    # Carter 3.  Built one entry per Carter; prims/graph filled in STEP 8b.
    # ------------------------------------------------------------------
    "bots": [
        {
            "i": i,
            "ns": ns,
            "tag": ROS2_TAGS[i],
            "robot": None,          # /World/Carter_{i+1}
            "chassis": None,        # <robot>/chassis_link
            "lidar_prim": None,     # <robot>/chassis_link/lidar
            "laser_frame": f"{ns}_laser",
            "graph": None,          # /ROS2ActionGraph_{i+1}
            "enabled": False,
            # --- OGN view of /<ns>/cmd_vel (ROS2SubscribeTwist) ---
            "cmd_attr": None,
            "cmd_ogn_last": (0.0, 0.0, 0.0),
            "cmd_ogn_changes": 0,
            # --- DDS view of /<ns>/cmd_vel (in-process monitor) ---
            "cmd_rx": 0,
            "cmd_last": None,
            # --- /<ns>/scan ring feeder (see _ros2_feed_scan) ---
            "scan_lin_out": None,   # ReadLidar.outputs:linearDepthData
            "scan_int_out": None,   # ReadLidar.outputs:intensitiesData
            "scan_lin_in": None,    # PublishScan.inputs:linearDepthData
            "scan_int_in": None,    # PublishScan.inputs:intensitiesData
            "lidar_cols_attr": None,
            "lidar_rows_attr": None,
            "scan_cols": None,      # resolved once (expect 360)
            "scan_rows": None,      # resolved once (expect 16)
            "scan_ring": None,      # ring whose elevation is closest to 0
            "scan_elev": None,      # that ring's elevation, degrees
            "scan_fed": 0,
            "scan_ring_tries": 0,
            # --- PER-CARTER command handover telemetry ---
            "handover_writes": 0,        # steps where ROS 2 drove it
            "handover_safety_holds": 0,  # steps Phase-2 safety kept it
            "handover_changes": 0,       # distinct Twist commands applied
            "handover_last": None,       # last applied (lin.x, ang.z)
            "handover_held": False,      # True while safety holds it
            "handover_applied_msgs": 0,  # cmd_vel msgs actually applied
            "handover_applied_msgs_seen": None,
            "handover_env_stops": 0,     # steps stopped by the envelope
            "handover_env_logged": False,
        }
        for i, ns in enumerate(ROS2_ROBOTS)
    ],
}

if PHASE3_ROS2:
    try:
        _ext_mgr = omni.kit.app.get_app().get_extension_manager()
        _ext_mgr.set_extension_enabled(ROS2_EXT_ID, True)
        for _ in range(10):        # no physics yet: the timeline is stopped
            simulation_app.update()
        ros2["ext_ok"] = bool(_ext_mgr.is_extension_enabled(ROS2_EXT_ID))
        log_print(f"PHASE3_ROS2_EXT: {ROS2_EXT_ID} enabled={ros2['ext_ok']}")
        if not ros2["ext_ok"]:
            # The extension disables itself when its own ROS 2 self-test
            # (bin/isaacsim.ros2.bridge.check) fails - see the lines above.
            ros2["errors"].append("extension self-test failed / not enabled")
            log_print(
                "PHASE3_ROS2_ERROR: bridge refused to start - check the "
                "'ROS2 Bridge startup failed' + 'export RMW_IMPLEMENTATION' "
                "lines above"
            )
    except Exception as e:
        ros2["errors"].append(f"enable_extension: {e}")
        log_print(f"PHASE3_ROS2_ERROR: enabling {ROS2_EXT_ID} failed: {e}")
else:
    log_print("PHASE3_ROS2: OFF (PHASE3_ROS2=0 control run)")

# ============================================================
# STEP 3: Configure physics rate (per official test)
# ============================================================
physics_rate = 60
carb.settings.get_settings().set_bool("/app/runLoops/main/rateLimitEnabled", True)
carb.settings.get_settings().set_int("/app/runLoops/main/rateLimitFrequency", int(physics_rate))
carb.settings.get_settings().set_int("/persistent/simulation/minFrameRate", int(physics_rate))
physics_dt = 1 / physics_rate

# ============================================================
# STEP 4: Create USD stage and load THREE Carters (M15)
# ============================================================
stage = get_current_stage()
assets_root = get_assets_root_path()

CARTER_USD_PATH = assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
ROBOT_SPECS = (
    # (prim path, start x, start y)
    # Starts keep clearly separated body footprints (>= 0.5 m pairwise
    # gap, measured at step 0) while preserving the greedy allocation
    # (CARTER_1 -> TASK_001, CARTER_2 -> TASK_002, CARTER_3 -> TASK_003)
    # and the M16 C2 x C3 corridor interaction.
    ("/World/Carter_1", -1.5, -6.6),
    ("/World/Carter_2", 2.0, -6.6),
    ("/World/Carter_3", -0.5, -9.0),
)
for _rp, _sx, _sy in ROBOT_SPECS:
    add_reference_to_stage(usd_path=CARTER_USD_PATH, prim_path=_rp)
log_print(f"Loaded {len(ROBOT_SPECS)} Carters from {CARTER_USD_PATH}")

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
# ONE shared M12 realistic warehouse (20x20 m floor, simple USD
# primitives) - built exactly once for the whole fleet.
#   rack row S (y=-3.6) with 4 racks, rows M (y=1.0) and N (y=5.6)
#   with 3 racks each; N-S aisles at x ~ -4.4 (forklift aisle),
#   x ~ 0.6 and x ~ 5.6; the main E-W corridor y in [-8.5,-5.5]
#   with the intersection at (-4.4, -7).
# ------------------------------------------------------------
WALL_COLOR = (0.6, 0.6, 0.65)
RACK_COLOR = (0.25, 0.4, 0.7)
PALLET_COLOR = (0.75, 0.55, 0.3)
CARTON_COLOR = (0.85, 0.6, 0.25)
PAINT_YELLOW = (0.95, 0.85, 0.1)
PAINT_WHITE = (0.95, 0.95, 0.95)

add_warehouse_box("/World/Wall_South", (0, -10, 1.5), (20.6, 0.4, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_North", (0, 10, 1.5), (20.6, 0.4, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_West", (-10, 0, 1.5), (0.4, 20.6, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_East", (10, 0, 1.5), (0.4, 20.6, 3.0), WALL_COLOR)

# --- pallet racks (blue), 3 rows; gaps form the N-S aisles ---
RACK_SIZE = (3.2, 1.4, 2.2)
RACK_ROWS = {
    "S": (-3.6, (-7.2, -2.0, 3.2, 8.0)),
    "M": (1.0, (-7.2, -2.0, 3.2)),
    "N": (5.6, (-7.2, -2.0, 3.2)),
}
rack_count = 0
for _row, (_ry, _rxs) in RACK_ROWS.items():
    for _ri, _rx in enumerate(_rxs):
        add_warehouse_box(
            f"/World/Rack_{_row}{_ri}", (_rx, _ry, 1.1), RACK_SIZE, RACK_COLOR
        )
        rack_count += 1

# --- pallets (wood) with cartons on top (brown) ---
PALLET_SIZE = (1.1, 1.1, 0.16)
CARTON_SIZE = (0.8, 0.8, 0.6)
PALLETS = ((3.2, -5.0), (-4.6, -1.3), (-8.6, -9.2), (2.6, -9.2))
pallet_count = 0
for _pi, (_px, _py) in enumerate(PALLETS):
    add_warehouse_box(
        f"/World/Pallet{_pi + 1}", (_px, _py, 0.08), PALLET_SIZE, PALLET_COLOR
    )
    add_warehouse_box(
        f"/World/Carton{_pi + 1}", (_px, _py, 0.46), CARTON_SIZE, CARTON_COLOR
    )
    pallet_count += 1


def add_floor_paint(tag, cx, cy, w, h, color, z=0.02):
    """Thin paint-only quad (no collision): floor markings/zones."""
    add_warehouse_box(
        f"/World/Floor_{tag}", (cx, cy, z), (w, h, 0.02), color, collision=False
    )


# --- floor markings / zones (paint only, never touch LiDAR) ---
add_floor_paint("Staging", -7.0, -8.6, 4.6, 2.0, (0.3, 0.4, 0.55))     # SW staging
add_floor_paint("Charging", 4.5, -8.6, 4.0, 2.0, (0.2, 0.7, 0.3))      # SE charging
add_floor_paint("Loading", -4.6, -1.3, 1.8, 3.0, (0.95, 0.5, 0.1))     # middle aisle
add_floor_paint("Restricted", 7.9, 3.3, 3.4, 3.0, (0.8, 0.12, 0.12))   # E restricted
add_floor_paint("AisleLineN", -4.5, -4.45, 10.6, 0.14, PAINT_YELLOW, z=0.03)
add_floor_paint("AisleLineS", -4.5, -9.55, 10.6, 0.14, PAINT_YELLOW, z=0.03)
for _zi, _zx in enumerate((-5.15, -4.75, -4.35, -3.95)):
    add_floor_paint(f"Zebra{_zi}", _zx, -7.0, 0.18, 2.6, PAINT_WHITE, z=0.04)

# --- zone props (collision) ---
add_warehouse_box("/World/Charger", (4.5, -9.3, 0.7), (1.2, 0.6, 1.4), (0.25, 0.3, 0.35))
add_warehouse_box(
    "/World/RestrictedBarrier", (6.3, 3.3, 0.25), (0.18, 2.6, 0.5), (0.9, 0.5, 0.1)
)

loading_area_built = True
charging_area_built = True
log_print(
    f"M15_BUILD: shared warehouse racks={rack_count}, pallets={pallet_count}, "
    "zones=STAGING+CHARGING+LOADING+RESTRICTED, markings=lines+zebra"
)
log_print("ONE shared warehouse loaded (walls + racks + pallets + cartons + zones)")

# ============================================================
# STEP 4a: NVIDIA SimReady VISUAL UPGRADE (RENDER ONLY)
# ============================================================
# Everything above is the FROZEN navigation / collision model.
# collision_boxes, the PhysX LiDAR masks, every threshold, the greedy
# allocation and all three state machines stay byte-for-byte identical
# to the verified run - this block only changes what the primitives
# LOOK like.  Each one gets a real NVIDIA USD asset resolved from the
# configured Isaac Sim asset root (get_assets_root_path()), which also
# performs the centimetre -> metre unit conversion through the Metrics
# Assembler (a raw Sdf reference would come in 100x too large).
#
# Contract, verified on THIS machine before the script was edited:
#   1. every asset URL is stat-checked with omni.client.stat() first;
#      a missing / unloadable asset is logged as NVIDIA_ASSET_MISSING
#      and skipped, so the primitive simply stays visible - a download
#      problem can never leave a hole in the scene;
#   2. a primitive is hidden ONLY AFTER its replacement composed and
#      produced a non-empty world bounding box;
#   3. UsdGeom.Imageable.MakeInvisible() is render only.  An isolated
#      sphere-drop test on this installation (5 boxes, rest height
#      1.7000 m = box top 1.500 + radius 0.200 in EVERY case) proved
#      the collider survives "hidden before the physics scene exists",
#      "hidden after world.play()", purpose=guide and
#      primvars:displayOpacity=0 - PhysX keeps building the shape from
#      UsdPhysics.CollisionAPI regardless of render visibility;
#   4. the added props and wall panels are placed outside every robot
#      route (x >= 4.5 m or y >= -2.5 m) so the PhysX scan,
#      collision_boxes and the robot paths are untouched;
#   5. some NVIDIA USDs ship their OWN CollisionAPI prims (SM_WallA_6M,
#      SM_floor02, SM_BarelPlastic_A_01, SM_PushcartA_02).  Every one of
#      them is disabled by _vis_disable_colliders() the moment the
#      reference lands, so the physics scene keeps EXACTLY the original
#      geometry: /groundPlane + the frozen UsdGeom.Cube colliders.
# ============================================================
import omni.client

VIS_STATS = {"rack": 0, "pallet": 0, "carton": 0, "wall": 0, "floor": 0,
             "prop": 0, "forklift": 0, "stock": 0, "human": 0,
             "skipped": 0, "col": 0}
_VIS_BBOX_CACHE = None
_VIS_JOBS = []
_VISUAL_UPGRADE = os.environ.get("VISUAL_UPGRADE", "1") == "1"


def _vis_cache():
    global _VIS_BBOX_CACHE
    if _VIS_BBOX_CACHE is None:
        _VIS_BBOX_CACHE = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
    return _VIS_BBOX_CACHE


def _vis_bbox(prim_path):
    """World-space AABB (min, max) of prim_path, or None."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    rng = _vis_cache().ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn, mx = rng.GetMin(), rng.GetMax()
    # express it in the prim's OWN frame: a wrapper parented under a
    # moving/moved ancestor (the forklift root drives its children from
    # its translate op) must be fitted in that ancestor's local frame,
    # which is exactly where its xformOp:translate will be applied.
    t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()).ExtractTranslation()
    return (Gf.Vec3d(mn[0] - t[0], mn[1] - t[1], mn[2] - t[2]),
            Gf.Vec3d(mx[0] - t[0], mx[1] - t[1], mx[2] - t[2]))


def _vis_disable_colliders(vis_path):
    """Turn off every PhysX collider the referenced asset brings with it.

    The visual layer must be render-only, but some NVIDIA USDs ship their
    own CollisionAPI prims: SM_WallA_6M (2 per panel), SM_floor02,
    SM_BarelPlastic_A_01 and SM_PushcartA_02.  Left active they would add
    physics geometry the original script never had - a SECOND ground
    plane under the robots' spawn drop, wall panels 2 cm proud of our
    cubes, and prop colliders that are absent from collision_boxes and
    from the LiDAR mask.  This run disabled all 35 of them.

    Authoring physics:collisionEnabled=False in the local layer overrides
    the value composed in from the referenced layer, so the mesh stays
    fully renderable while PhysX builds no shape for it.  Our frozen
    UsdGeom.Cube colliders and /groundPlane remain the only physics
    geometry in the scene.
    """
    root = stage.GetPrimAtPath(vis_path)
    if not root or not root.IsValid():
        return 0
    n = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            log_print(
                f"NVIDIA_ASSET_RIGID_BODY: {prim.GetPath()} "
                "(visual assets must stay static)"
            )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            n += 1
    return n


def vis_add(kind, rel, vis_path, target_size, target_pos, mode="uniform",
            anchor="floor", hide=(), rot_deg=0.0, prop_cap=2.5):
    """Queue one NVIDIA visual replacement (or visual-only addition).

    mode:
      uniform     - uniform scale, never leaves the primitive box
      uniform_xy  - uniform scale from the footprint, height may exceed
                    it (open air above, e.g. the forklift mast)
      exact       - non-uniform scale that exactly fills the box
      floor       - thin slab scaled in x/y only
      natural     - asset's own size (optionally shrunk to prop_cap)

    VISUAL_UPGRADE=0 disables the whole layer (primitives stay visible),
    which exists only to produce a before/after render comparison.
    """
    if not _VISUAL_UPGRADE:
        return
    _VIS_JOBS.append(dict(kind=kind, rel=rel, vis_path=vis_path,
                          size=tuple(target_size), pos=tuple(target_pos),
                          mode=mode, anchor=anchor, hide=list(hide),
                          rot=float(rot_deg), cap=float(prop_cap),
                          queued=True))


def vis_apply():
    """Verify -> reference -> auto-fit -> hide the primitive (last).

    Returns the list of vis_paths whose replacement measured OK, so a
    caller can hide a shared primitive only once all of its panels are
    in place.
    """
    pending = []
    for job in _VIS_JOBS:
        if not job.pop("queued", False):
            continue
        url = assets_root + "/" + job["rel"]
        try:
            code, _info = omni.client.stat(url)
            found = code == omni.client.Result.OK
        except Exception:
            found = False
        if not found:
            log_print(f"NVIDIA_ASSET_MISSING: {job['kind']} {job['rel']}")
            VIS_STATS["skipped"] += 1
            continue
        # wrapper Xform carries OUR transform ops; the reference goes on
        # a child so the asset's own xformOpOrder can never reorder them
        try:
            UsdGeom.Xform.Define(stage, job["vis_path"])
            add_reference_to_stage(usd_path=url,
                                   prim_path=job["vis_path"] + "/Asset")
        except Exception as exc:
            log_print(f"NVIDIA_ASSET_REF_ERROR: {job['rel']} -> {exc}")
            VIS_STATS["skipped"] += 1
            continue
        # strip the asset's own colliders BEFORE it can be measured or
        # ever seen by the physics scene: the visual layer is render-only
        _n = _vis_disable_colliders(job["vis_path"])
        if _n:
            VIS_STATS["col"] += _n
            log_print(
                f"NVIDIA_ASSET_COLLIDER_DISABLED: {job['kind']} "
                f"{job['rel']} ({_n} prim(s))"
            )
        pending.append(job)

    if pending:
        simulation_app.update()  # compose every reference once, then measure

    applied = []
    for job in pending:
        bb = _vis_bbox(job["vis_path"])
        nat = None if bb is None else [float(bb[1][i] - bb[0][i]) for i in range(3)]
        degenerate = (nat is None) or (
            (nat[0] <= 1e-6 or nat[1] <= 1e-6)
            or (job["mode"] != "floor" and min(nat) <= 1e-6)
        )
        if degenerate:
            log_print(
                f"NVIDIA_ASSET_EMPTY_BBOX: {job['kind']} {job['rel']} "
                f"size={None if nat is None else [round(v, 4) for v in nat]}"
            )
            try:
                stage.RemovePrim(job["vis_path"])
            except Exception:
                pass
            VIS_STATS["skipped"] += 1
            continue

        mn, mx = bb
        nat = [float(bb[1][i] - bb[0][i]) for i in range(3)]
        ctr = [float((bb[1][i] + bb[0][i]) * 0.5) for i in range(3)]
        rot = job["rot"] % 360.0
        if abs(rot - 90.0) < 1e-6 or abs(rot - 270.0) < 1e-6:
            # rotation about Z swaps the x/y footprint
            nat_f, ctr_f = [nat[1], nat[0], nat[2]], [ctr[1], ctr[0], ctr[2]]
            mn_f = [mn[1], mn[0], mn[2]]
            mx_f = [mx[1], mx[0], mx[2]]
        else:
            nat_f, ctr_f, mn_f, mx_f = list(nat), list(ctr), list(mn), list(mx)
        if abs(rot - 90.0) < 1e-6:
            ctr_f[0] = -ctr[1]
        elif abs(rot - 180.0) < 1e-6:
            ctr_f[0], ctr_f[1] = -ctr[0], -ctr[1]
        elif abs(rot - 270.0) < 1e-6:
            ctr_f[0], ctr_f[1] = ctr[1], -ctr[0]

        tgt = job["size"]
        if job["mode"] == "uniform":
            s = min(tgt[i] / nat_f[i] for i in range(3))
            scl = [s, s, s]
        elif job["mode"] == "uniform_xy":
            s = min(tgt[i] / nat_f[i] for i in range(2))
            scl = [s, s, s]
        elif job["mode"] == "exact":
            scl = [tgt[i] / nat_f[i] for i in range(3)]
        elif job["mode"] == "floor":
            s = min(tgt[i] / nat_f[i] for i in range(2))
            # keep the slab between the grid plane (z=0) and the paint (z=0.01)
            scl = [s, s, s if nat_f[2] <= 1e-6 else min(s, 0.010 / nat_f[2])]
        else:  # natural
            s = 1.0
            big = max(nat_f)
            if job["cap"] > 0.0 and big > job["cap"]:
                s = job["cap"] / big
            scl = [s, s, s]

        px, py, pz = job["pos"]
        tx = px - scl[0] * ctr_f[0]
        ty = py - scl[1] * ctr_f[1]
        if job["anchor"] == "top":
            tz = pz - scl[2] * mx_f[2]
        elif job["anchor"] == "center":
            tz = pz - scl[2] * ctr_f[2]
        else:  # floor
            tz = pz - scl[2] * mn_f[2]

        try:
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(job["vis_path"]))
            xf.AddTranslateOp().Set(Gf.Vec3f(tx, ty, tz))   # outermost
            xf.AddScaleOp().Set(Gf.Vec3f(scl[0], scl[1], scl[2]))
            if abs(rot) > 1e-6:
                xf.AddRotateZOp().Set(rot)                  # innermost
        except Exception as exc:
            log_print(f"NVIDIA_ASSET_XFORM_ERROR: {job['rel']} -> {exc}")
            VIS_STATS["skipped"] += 1
            continue

        hidden = 0
        for hp in job["hide"]:
            hprim = stage.GetPrimAtPath(hp)
            if hprim and hprim.IsValid():
                UsdGeom.Imageable(hprim).MakeInvisible()
                hidden += 1

        VIS_STATS[job["kind"]] = VIS_STATS.get(job["kind"], 0) + 1
        applied.append(job["vis_path"])
        log_print(
            f"NVIDIA_ASSET_USED: {job['kind']} {job['rel']} -> {job['vis_path']} "
            f"natural {nat[0]:.3f}x{nat[1]:.3f}x{nat[2]:.3f} m "
            f"fitted {scl[0] * nat_f[0]:.3f}x{scl[1] * nat_f[1]:.3f}"
            f"x{scl[2] * nat_f[2]:.3f} m "
            f"scale ({scl[0]:.3f},{scl[1]:.3f},{scl[2]:.3f}) rot {rot:.0f} "
            f"primitive_hidden {hidden}/{len(job['hide'])}"
        )

    log_print(f"NVIDIA_VISUAL_PASS: applied={len(applied)} "
              f"skipped_total={VIS_STATS['skipped']}")
    return applied


# --- NVIDIA USD assets (all stat-verified against the asset root) ---
RACK_USD = ("NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"
            "Shelving/Racks/Rack_A/Rack_A08_PR_NVD_01.usd")
PALLET_USD = ("NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"
              "Shipping/Pallets/Wood/Block_A/BlockPallet_A08_PR_NVD_01.usd")
CARTON_USDS = (
    "NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"
    "Shipping/Cardboard_Boxes/Cube_A/CubeBox_A10_92cm_PR_NVD_01.usd",
    "NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"
    "Shipping/Cardboard_Boxes/Cube_A/CubeBox_A09_77cm_PR_NVD_01.usd",
)
WALL_USD = "Isaac/Environments/Simple_Warehouse/Props/SM_WallA_6M.usd"
FLOOR_USD = "Isaac/Environments/Simple_Warehouse/Props/SM_floor02.usd"
FORKLIFT_USD = ("NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"
                "Equipment/Forklifts/Forklift_A/Forklift_A01_PR_V_NVD_01.usd")

# --- 10 racks: same centres, same footprint, real pallet racks ---
for _row, (_ry, _rxs) in RACK_ROWS.items():
    for _ri, _rx in enumerate(_rxs):
        vis_add("rack", RACK_USD, f"/World/RackVis_{_row}{_ri}",
                RACK_SIZE, (_rx, _ry, 0.0), mode="uniform", anchor="floor",
                hide=(f"/World/Rack_{_row}{_ri}/Geom",))

# --- 4 pallets + 4 cartons: SimReady block pallets / cardboard boxes ---
CARTON_Z0 = 0.46 - CARTON_SIZE[2] * 0.5   # carton bottom sits on the pallet
for _pi, (_px, _py) in enumerate(PALLETS):
    vis_add("pallet", PALLET_USD, f"/World/PalletVis{_pi + 1}",
            PALLET_SIZE, (_px, _py, 0.0), mode="exact", anchor="floor",
            hide=(f"/World/Pallet{_pi + 1}/Geom",))
    vis_add("carton", CARTON_USDS[_pi % len(CARTON_USDS)],
            f"/World/CartonVis{_pi + 1}", CARTON_SIZE,
            (_px, _py, CARTON_Z0), mode="exact", anchor="floor",
            hide=(f"/World/Carton{_pi + 1}/Geom",))

# --- SimReady inventory ON the rack shelves (render-only, no physics) ---
#
# Shelf geometry comes from Rack_A08 itself: after the exact vis_add()
# fit above, the rack's horizontal deck members top out at world
# z = 0.475 m and z = 1.155 m, leaving 0.655 m / 0.665 m of clear
# headroom under the next beam.  Every item is therefore queued with
# anchor="floor" at one of those two z values, so it sits ON a deck and
# INSIDE the rack footprint - never on the floor, never in an aisle.
# Usable deck is +-1.45 m in x (clear of the upright frames) and
# +-0.62 m in y (rack half depth 0.6565 m minus a margin).
_STOCK_HALF_X = 1.45
_STOCK_HALF_Y = 0.62
_STOCK_SPAN = 2.0 * _STOCK_HALF_X        # 2.90 m of usable deck length
_STOCK_GAP = 0.06                        # gap between neighbouring items
_STOCK_LIMIT = 56                        # hard ceiling (band is 30-60)
_SHELF_DECKS = (0.475, 1.155)            # two usable decks per rack
_STOCK_NCHOICE = (0, 2, 3, 3, 3, 4)      # >=1 shelf per rack stays empty
_DTW = "NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"
# (relative USD, natural x, natural y, natural z, selection weight).
# Heights: max 0.472 m < 0.655 m clear, so nothing can touch a beam.
_STOCK_ITEMS = (
    (_DTW + "Storage/Totes/Tote_A/Tote_A04_80x60x22cm_PR_V_NVD_01.usd",
     0.800, 0.600, 0.220, 4),
    (_DTW + "Storage/Cases/Case_A/Case_A03_65x48x30cm_PR_V_NVD_01.usd",
     0.656, 0.480, 0.305, 4),
    (_DTW + "Storage/Boxes/Box_A/Box_A01_60x40x32cm_PR_V_NVD_01.usd",
     0.600, 0.400, 0.330, 5),
    (_DTW + "Storage/Containers/Container_A/"
     "Container_A03_63x45x33cm_PR_V_NVD_01.usd",
     0.625, 0.445, 0.330, 4),
    (_DTW + "Storage/Bins/Bin_A/Bin_A06_30x51x19cm_PR_V_NVD_01.usd",
     0.508, 0.300, 0.187, 3),
    (_DTW + "Storage/Boxes/Box_A/Box_A05_40x30x32cm_PR_V_NVD_01.usd",
     0.400, 0.300, 0.330, 3),
    (_DTW + "Storage/Boxes/Box_A/Box_A11_40x30x23cm_PR_V_NVD_01.usd",
     0.401, 0.300, 0.130, 2),
    (_DTW + "Shipping/Cardboard_Boxes/Cube_A/"
     "CubeBox_A06_42cm_PR_NVD_01.usd",
     0.413, 0.414, 0.421, 3),
    (_DTW + "Shipping/Cardboard_Boxes/White_A/"
     "WhiteCorrugatedBox_A07_25x25x25cm_PR_NVD_01.usd",
     0.251, 0.254, 0.252, 1),
    (_DTW + "Storage/Drums/Plastic_A/PlasticDrum_A01_PR_V_NVD_01.usd",
     0.283, 0.284, 0.387, 2),
    (_DTW + "Shipping/Wood_Crates/PalletCollar_A/"
     "WoodPalletCollar_A08_80x120x30cm_PR_NV_01.usd",
     0.817, 1.215, 0.317, 3),
    (_DTW + "Shipping/Wood_Crates/Plywood_A/"
     "PlywoodCrate_A04_120x120x50cm_PR_NV_01.usd",
     1.215, 1.219, 0.472, 2),
)
_stock_planned = 0
_stock_len = 0.0
_stock_span = 0.0
_stock_kinds = set()
for _row, (_ry, _rxs) in RACK_ROWS.items():
    for _ri, _rx in enumerate(_rxs):
        for _si, _deck_z in enumerate(_SHELF_DECKS):
            # one deterministic stream per deck: no two racks come out alike
            _sr = random.Random(20261001 + ord(_row) * 97 + _ri * 17 + _si * 7)
            _cursor = -_STOCK_HALF_X
            for _k in range(_sr.choice(_STOCK_NCHOICE)):
                if _stock_planned >= _STOCK_LIMIT:
                    break
                _it = _STOCK_ITEMS[_sr.choices(
                    range(len(_STOCK_ITEMS)),
                    weights=[w[4] for w in _STOCK_ITEMS], k=1)[0]]
                _irot = _sr.choice((0.0, 90.0))
                # 90 deg swaps the footprint: pick whichever fits in y
                _dx = _it[2] if _irot else _it[1]
                _dy = _it[1] if _irot else _it[2]
                if _dy > 2.0 * _STOCK_HALF_Y:
                    continue
                if _cursor + _STOCK_GAP + _dx > _STOCK_HALF_X:
                    break                      # deck is full
                _cx = _cursor + _STOCK_GAP + _dx * 0.5
                _cursor += _STOCK_GAP + _dx
                _yroom = _STOCK_HALF_Y - _dy * 0.5
                _cy = _ry + (_sr.uniform(-_yroom, _yroom)
                             if _yroom > 0.02 else 0.0)
                vis_add("stock", _it[0],
                        f"/World/StockVis_{_row}{_ri}{_si}_{_k}",
                        (0.0, 0.0, 0.0), (_rx + _cx, _cy, _deck_z),
                        mode="natural", anchor="floor", rot_deg=_irot)
                _stock_planned += 1
                _stock_len += _dx
                _stock_kinds.add(_it[0].rsplit("/", 1)[-1])
            _stock_span += _STOCK_SPAN
_stock_fill = (100.0 * _stock_len / _stock_span) if _stock_span > 0.0 else 0.0
log_print(
    f"NVIDIA_STOCK_PLAN: planned={_stock_planned} "
    f"decks={len(_SHELF_DECKS) * 10} fill={_stock_fill:.1f}% "
    f"limit={_STOCK_LIMIT} kinds={len(_stock_kinds)} "
    f"deck_z={_SHELF_DECKS} (on the shelves inside the rack footprint)"
)

# --- 4 walls: 4 tiled SM_WallA_6M panels each, 2 cm proud of the cube
#     in thickness and height and exactly tiling its length, so the
#     grey cube ends up fully enclosed (and is then hidden) ---
WALL_PANELS = 4
WALL_SPEC = (("South", "/World/Wall_South", 0.0, -10.0, 90.0),
             ("North", "/World/Wall_North", 0.0, 10.0, 90.0),
             ("West", "/World/Wall_West", -10.0, 0.0, 0.0),
             ("East", "/World/Wall_East", 10.0, 0.0, 0.0))
_WALL_HIDE = []
for _wtag, _wpath, _wcx, _wcy, _wrot in WALL_SPEC:
    _panels = []
    for _wi in range(WALL_PANELS):
        _off = -10.3 + (20.6 / WALL_PANELS) * (_wi + 0.5)
        if _wrot == 0.0:          # length already runs along y
            _ppos = (_wcx, _off, 0.0)
            _psize = (0.44, 5.15, 3.05)
        else:                      # panel length must run along x
            _ppos = (_off, _wcy, 0.0)
            _psize = (5.15, 0.44, 3.05)
        _vp = f"/World/WallVis_{_wtag}{_wi}"
        vis_add("wall", WALL_USD, _vp, _psize, _ppos, mode="exact",
                anchor="floor", rot_deg=_wrot)
        _panels.append(_vp)
    _WALL_HIDE.append((_wpath, _panels))

# --- concrete warehouse floor over the simulation grid (paint stays on
#     top: the slab tops out at z=0.005, the markings start at z=0.01) ---
vis_add("floor", FLOOR_USD, "/World/FloorVis", (40.0, 40.0, 0.0),
        (0.0, 0.0, 0.005), mode="floor", anchor="top")

# --- visual-only props (NO CollisionAPI), all outside every robot route ---
VIS_PROPS = (
    ("NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Shipping/Wood_Crates/"
     "Standard_A/StandardWoodCrate_A19_80x80x80cm_PR_NV_01.usd",
     5.6, -5.8, "crateA"),
    ("NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Shipping/Wood_Crates/"
     "Standard_A/StandardWoodCrate_A19_80x80x80cm_PR_NV_01.usd",
     6.6, -8.7, "crateB"),
    ("NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Equipment/Hand_Trucks/"
     "Std_Alum_A/StandardAluminumHandTruck_A01_PR_NVD_01.usd",
     7.4, -6.4, "handtruck"),
    ("NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Facilities/Trash_Cans/"
     "TrashCanWithWheels_A/TrashCanWithWheels_A01_Yellow_01.usd",
     8.5, -7.4, "trashcan"),
    ("NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/"
     "TrafficCone_A04_71cm_PR_V_NVD_01.usd",
     5.6, -4.7, "cone"),
    ("NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Safety/Barriers/"
     "GuardRail_A/GuardRail_A01_01.usd",
     7.9, 2.6, "guardrail"),
    ("Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
     7.2, -8.9, "barrel"),
    ("Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd",
     0.8, 3.3, "pushcart"),
)
for _rel, _ppx, _ppy, _ptag in VIS_PROPS:
    vis_add("prop", _rel, f"/World/PropVis_{_ptag}", (0.0, 0.0, 0.0),
            (_ppx, _ppy, 0.0), mode="natural", anchor="floor")

# --- 5 render-only NVIDIA human workers (RL-rig characters, NO physics) --
# Every worker stands OUTSIDE the robot operating box (x in [-9, 3.6],
# y in [-9.6, -1.3]), i.e. y >= -0.4 or x >= 4.5, and clear of the
# STAGING / CHARGING / LOADING / RESTRICTED paint, all 4 pallets, the
# forklift lane (FORK_X = -4.4) and every VIS_PROPS position above.
# They are queued like any other visual asset, so vis_apply() calls
# _vis_disable_colliders() on them first: PhysX builds no shape, and
# RotatingLidarPhysX raycasts PHYSX geometry - LiDAR, clearance,
# collision_boxes, task allocation and every safety gate stay untouched.
_HUMAN_USDS = (
    "Isaac/People/Characters/male_adult_construction_01_new/"
    "male_adult_construction_01_new.usd",
    "Isaac/People/Characters/male_adult_construction_03/"
    "male_adult_construction_03.usd",
    "Isaac/People/Characters/male_adult_construction_05_new/"
    "male_adult_construction_05_new.usd",
    "Isaac/People/Characters/original_male_adult_construction_02/"
    "male_adult_construction_02.usd",
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
)
# (x, y, yaw, character index, static-pose index) - yaw is a multiple of
# 90 so vis_add()'s footprint swap stays exact.
WORKERS = (
    (-6.8, -0.4, 180.0, 0, 0),   # west aisle, facing north
    (-2.9, -0.4, 270.0, 1, 4),   # beside the LOADING lane, crouched
    (-2.6, 3.3, 90.0, 2, 1),     # middle aisle, facing east
    (5.4, -2.6, 180.0, 3, 3),    # east of the racks, facing north
    (-0.5, 7.2, 0.0, 4, 2),      # north aisle, facing south
)
for _wi, (_wx, _wy, _wyaw, _ci, _pose) in enumerate(WORKERS):
    vis_add("human", _HUMAN_USDS[_ci], f"/World/WorkerVis{_wi + 1}",
            (0.0, 0.0, 0.0), (_wx, _wy, 0.0),
            mode="natural", anchor="floor", rot_deg=_wyaw)

_vis_applied = vis_apply()

# a wall cube is hidden only once ALL of its panels measured OK, so a
# partial download can never leave a gap in the building shell
for _wpath, _wpanels in _WALL_HIDE:
    _ready = all(_p in _vis_applied for _p in _wpanels)
    if _ready:
        _wp = stage.GetPrimAtPath(_wpath)
        if _wp and _wp.IsValid():
            UsdGeom.Imageable(_wp).MakeInvisible()
            VIS_STATS["wall"] += 1
            log_print(f"NVIDIA_ASSET_USED: wall {_wpath} hidden behind "
                      f"{len(_wpanels)} x SM_WallA_6M panels")
    else:
        log_print(f"NVIDIA_WALL_PARTIAL: kept {_wpath} visible "
                  f"({sum(1 for _p in _wpanels if _p in _vis_applied)}"
                  f"/{len(_wpanels)} panels loaded)")

# --- static worker poses: ONE frame of the Reallusion animation library --
# The Isaac character USDs ship in a T-pose.  Their skeleton is the
# Reallusion RL_Bone rig (101 joints, RL_BoneRoot/...), which the stock
# Isaac/People/Animations clips CANNOT drive - those are an incompatible
# 81-joint 'Root/Pelvis' rig, so name-matched skinning maps nothing.
# Rivermark's pedestrian library is the SAME rig: verified joint for
# joint identical (101/101, same set, same order) apart from the root
# token '_Root'.  Each worker therefore gets ONE sampled frame authored
# as a constant-valued SkelAnimation - no time samples, so the pose is
# frozen: nothing animates, nothing depends on the timeline, and the
# skinned mesh is the ONLY thing that changes.  Binding goes on that
# character's SkelRoot, exactly like the stock UsdSkel setup.
_POSE_CLIPS = (
    # (relative USD, frame, tag)
    ("Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/"
     "common_assets/animals/pedestrians/Reallusion/DanceCard_DS/"
     "Relax_Idle_I_take_001.usd", 8.0, "idleA"),
    ("Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/"
     "common_assets/animals/pedestrians/Reallusion/DanceCard_DS/"
     "Relax_Idle_I_take_001.usd", 26.0, "idleB"),
    ("Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/"
     "common_assets/animals/pedestrians/Reallusion/DanceCard_DS/"
     "Relax_Idle_I_take_001.usd", 48.0, "idleC"),
    ("Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/"
     "common_assets/animals/pedestrians/Reallusion/DanceCard_DS/"
     "action_tie_shoes_standing_01.usda", 1.0, "stand"),
    ("Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/"
     "common_assets/animals/pedestrians/Reallusion/DanceCard_DS/"
     "action_picking_up_01.usda", 1.0, "lift"),
)


def _author_worker_poses(applied_paths):
    """Freeze one RL-rig pose onto each worker.  Render-only, never fatal."""
    from pxr import UsdSkel

    # the character's own joint list, taken from the first worker that loaded
    char_joints = None
    for _wi in range(len(WORKERS)):
        _ap = f"/World/WorkerVis{_wi + 1}"
        if _ap not in applied_paths:
            continue
        _root = stage.GetPrimAtPath(_ap + "/Asset")
        if not _root or not _root.IsValid():
            continue
        for _p in Usd.PrimRange(_root):
            if _p.IsA(UsdSkel.Skeleton):
                char_joints = list(
                    UsdSkel.Skeleton(_p).GetJointsAttr().Get() or [])
                break
        if char_joints:
            break
    if not char_joints:
        log_print("NVIDIA_WORKER_POSE: UNAVAILABLE (no worker skeleton)")
        return

    scratch = "/World/_WorkerClip"      # untyped: the reference must be able
    pose_of = {}                        # to author 'SkelAnimation' itself
    for _pi, (_rel, _frame, _tag) in enumerate(_POSE_CLIPS):
        url = assets_root + "/" + _rel
        try:
            code, _info = omni.client.stat(url)
            found = code == omni.client.Result.OK
        except Exception:
            found = False
        if not found:
            log_print(f"NVIDIA_WORKER_CLIP_MISSING: {_rel}")
            continue
        try:
            stage.DefinePrim(scratch)   # no typeName authored on purpose
            add_reference_to_stage(usd_path=url, prim_path=scratch)
            cp = stage.GetPrimAtPath(scratch)
            if not UsdSkel.IsSkelAnimationPrim(cp):
                log_print(f"NVIDIA_WORKER_CLIP_NOT_ANIM: {_rel}")
                stage.RemovePrim(scratch)
                continue
            src = UsdSkel.Animation(cp)
            sj = list(src.GetJointsAttr().Get() or [])
            norm = [("RL_BoneRoot" + j[5:]) if j.startswith("_Root") else j
                    for j in sj]
            if norm != char_joints:
                log_print(f"NVIDIA_WORKER_CLIP_RIG_MISMATCH: {_rel} "
                          f"joints={len(sj)} vs {len(char_joints)}")
                stage.RemovePrim(scratch)
                continue
            tc = Usd.TimeCode(_frame)
            tr, ro, sc = (src.GetTranslationsAttr(),
                          src.GetRotationsAttr(), src.GetScalesAttr())
            v_tr = tr.Get(tc) if tr is not None and tr.HasValue() else None
            v_ro = ro.Get(tc) if ro is not None and ro.HasValue() else None
            v_sc = sc.Get(tc) if sc is not None and sc.HasValue() else None
            if v_tr is None or v_ro is None:
                log_print(f"NVIDIA_WORKER_CLIP_NODATA: {_rel} frame={_frame}")
                stage.RemovePrim(scratch)
                continue
            # author a FRESH constant-valued SkelAnimation (no time samples)
            _pp = f"/World/WorkerPose{_pi}"
            pa = UsdSkel.Animation.Define(stage, _pp)   # returns the schema
            pa.CreateJointsAttr().Set(char_joints)
            pa.CreateTranslationsAttr().Set(v_tr)
            pa.CreateRotationsAttr().Set(v_ro)
            if v_sc is not None:
                pa.CreateScalesAttr().Set(v_sc)
            pose_of[_pi] = (_pp, _tag)
            stage.RemovePrim(scratch)
        except Exception as exc:
            log_print(f"NVIDIA_WORKER_CLIP_ERROR: {_rel} -> {exc}")
            try:
                stage.RemovePrim(scratch)
            except Exception:
                pass

    bound = 0
    for _wi, (_wx, _wy, _wyaw, _ci, _pose) in enumerate(WORKERS):
        _ap = f"/World/WorkerVis{_wi + 1}"
        if _ap not in applied_paths or _pose not in pose_of:
            continue
        try:
            _root = None
            for _p in Usd.PrimRange(stage.GetPrimAtPath(_ap + "/Asset")):
                if _p.IsA(UsdSkel.Root):
                    _root = _p
                    break
            if _root is None:
                log_print(f"NVIDIA_WORKER_POSE_NO_ROOT: {_ap}")
                continue
            UsdSkel.BindingAPI.Apply(_root)
            UsdSkel.BindingAPI(_root).CreateAnimationSourceRel().SetTargets(
                [pose_of[_pose][0]])
            bound += 1
            log_print(f"NVIDIA_WORKER_POSE_USED: {_ap} <- {pose_of[_pose][1]} "
                      f"({pose_of[_pose][0]})")
        except Exception as exc:
            log_print(f"NVIDIA_WORKER_POSE_ERROR: {_ap} -> {exc}")
    log_print(f"NVIDIA_WORKER_POSE: bound={bound}/{len(WORKERS)}")


if _VISUAL_UPGRADE:
    _author_worker_poses(_vis_applied)

# ============================================================
# STEP 4b: Add ONE PhysX LiDAR per Carter (proven config, x3)
# ============================================================
LIDAR_MAX_RANGE = 50.0
lidars = []
for _i, (_rp, _sx, _sy) in enumerate(ROBOT_SPECS):
    _lp = f"{_rp}/chassis_link/lidar"
    _lidar = RotatingLidarPhysX(
        prim_path=_lp,
        name=f"carter_lidar_{_i + 1}",
        rotation_frequency=0.0,
        translation=np.array([-0.06, 0.0, 0.38]),
        fov=(360.0, 30.0),          # horizontal 360deg, vertical 30deg
        resolution=(1.0, 2.0),      # 360 cols x 16 rows -> lightweight scan
        valid_range=(0.4, LIDAR_MAX_RANGE),
    )
    _lidar.add_linear_depth_data_to_frame()
    _lidar.add_azimuth_data_to_frame()
    _lidar.add_zenith_data_to_frame()
    # Lightweight LiDAR visualization: hit POINTS only, high_lod=True
    # keeps the full 360x16 scan (verified A/B probe on this machine).
    try:
        _lidar.enable_visualization(high_lod=True, draw_points=True, draw_lines=False)
        log_print(f"CARTER_{_i + 1}_LIDAR_VIZ: enabled (hit points, high_lod=True)")
    except Exception as e:
        log_print(f"CARTER_{_i + 1}_LIDAR_VIZ: skipped ({e})")
    log_print(f"CARTER_{_i + 1}_LIDAR_PRIM: {_lp}")
    lidars.append(_lidar)

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
# STEP 6: Create ONE SingleArticulation per Carter (M15)
# ============================================================
articulations = []
for _i, (_rp, _sx, _sy) in enumerate(ROBOT_SPECS):
    _ar = SingleArticulation(_rp)
    log_print(f"SingleArticulation created at {_rp}")
    try:
        _ar._articulation_view.initialize()
        log_print(f"CARTER_{_i + 1}_ART_VIEW: initialized")
    except Exception as e:
        log_print(f"CARTER_{_i + 1}_ART_VIEW: initialize info: {e}")
    articulations.append(_ar)

# ============================================================
# STEP 7: Initialize the three LiDAR sensors
# ============================================================
lidar_init_flags = []
for _i, _lidar in enumerate(lidars):
    try:
        _lidar.initialize()
        lidar_init_flags.append(True)
        log_print(f"CARTER_{_i + 1}_LIDAR_SENSOR: initialized")
    except Exception as e:
        lidar_init_flags.append(False)
        log_print(f"CARTER_{_i + 1}_LIDAR_INITIALIZE_ERROR: {e}")

# Temporarily pose every robot at identity (world yaw ~180 deg) so the
# quaternion order can be probed exactly like the frozen script and the
# body AABB can be measured in a well-conditioned orientation.
for _i, (_rp, _sx, _sy) in enumerate(ROBOT_SPECS):
    articulations[_i].set_world_pose(position=[_sx, _sy, 0.1], orientation=[0, 0, 0, 1])
    articulations[_i].set_world_velocity([0, 0, 0, 0, 0, 0])

for _ in range(10):
    simulation_app.update()

_probe_rot = articulations[0].get_world_pose()[1]


# ------------------------------------------------------------
# Yaw extraction (self-verifying quaternion order): standard ZYX
# yaw, CCW positive. The probe pose must yield ~+-180 deg.
# ------------------------------------------------------------
def _yaw_deg(w, x, y, z):
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


_quat_order = "wxyz"
_probe = _yaw_deg(*[float(v) for v in _probe_rot])
if abs(_probe) < 90.0:
    _quat_order = "xyzw"
    _probe = _yaw_deg(float(_probe_rot[3]), float(_probe_rot[0]),
                      float(_probe_rot[1]), float(_probe_rot[2]))
log_print(f"START_YAW_PROBE: {_probe:.2f} deg (quat order {_quat_order})")


def yaw_of(rot):
    """World yaw in degrees (CCW positive) from the articulation quaternion."""
    r = [float(v) for v in rot]
    if _quat_order == "wxyz":
        return _yaw_deg(r[0], r[1], r[2], r[3])
    return _yaw_deg(r[3], r[0], r[1], r[2])


def yaw_quat(yaw_deg):
    """wxyz (or xyzw) quaternion that points the Carter nose at world yaw."""
    half = math.radians(yaw_deg) / 2.0
    w, s = math.cos(half), math.sin(half)
    if _quat_order == "wxyz":
        return [w, 0.0, 0.0, s]
    return [0.0, 0.0, s, w]


# ------------------------------------------------------------
# Measure each robot's body AABB (yaw ~180 deg, well conditioned)
# -> half extents used for the robot-vs-robot rectangle gap check.
# Fallback: NovaCarter-sized box (verified close to the real mesh).
# ------------------------------------------------------------
def measure_half_extents(ar, yaw_deg):
    """Half (length, width) of the robot's world AABB de-rotated to the
    robot frame; falls back to (0.5, 0.375) on any failure."""
    L_fallback, W_fallback = 1.0, 0.75
    try:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        # ComputeWorldBound takes the Usd.Prim (passing the Xformable
        # raised a Python/C++ argument mismatch and silently fell back
        # to the oversized placeholder box).
        cache_prim = get_prim_at_path(ar.prim_path)
        bound = cache.ComputeWorldBound(cache_prim)
        rng = bound.ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        dx = float(mx[0] - mn[0])
        dy = float(mx[1] - mn[1])
        th = math.radians(yaw_deg)
        c, s = abs(math.cos(th)), abs(math.sin(th))
        det = c * c - s * s
        if abs(det) < 0.2:
            raise RuntimeError(f"ill-conditioned yaw {yaw_deg:.1f}")
        L = (c * dx - s * dy) / det
        W = (c * dy - s * dx) / det
        # Plausibility window for the Carter body.  The measured body box is
        # L=0.707 W=0.477 on all three robots (half-diagonal 0.43 m, which is
        # consistent with the proven SELF_HIT_MAX=0.5 lidar self-mask).  The
        # old window (W >= 0.5) rejected that real measurement and silently
        # fell back to a placeholder 2.0 x 1.5 m box, which is 3x oversized:
        # it produced phantom rect gaps < 0.25 m at 3 m robot separation and
        # inflated attribute_forward_entity()'s predicted distance so that
        # lidar attribution failed (swerving instead of yielding).
        if not (0.5 <= L <= 2.0 and 0.3 <= W <= 1.5):
            raise RuntimeError(f"implausible extents L={L:.3f} W={W:.3f}")
        return np.array([L / 2.0, W / 2.0], dtype=np.float64), (dx, dy, L, W)
    except Exception as e:
        log_print(f"AABB_MEASURE_FALLBACK: {e}")
        return np.array([L_fallback, W_fallback], dtype=np.float64), None


robot_halves = []
for _i, _ar in enumerate(articulations):
    _rot = _ar.get_world_pose()[1]
    _y = yaw_of(_rot)
    _half, _info = measure_half_extents(_ar, _y)
    robot_halves.append(_half)
    if _info is not None:
        log_print(
            f"CARTER_{_i + 1}_AABB: world dx={_info[0]:.3f} dy={_info[1]:.3f} "
            f"-> L={_info[2]:.3f} W={_info[3]:.3f} half={_half.tolist()} "
            f"(yaw {_y:.1f})"
        )
    else:
        log_print(f"CARTER_{_i + 1}_AABB: fallback half={_half.tolist()}")

# ============================================================
# STEP 7b: M13/M14 TRAFFIC ENTITIES (kinematic, deterministic).
# One shared forklift + one shared human (frozen schedules).
# ============================================================
FORK_X = -4.4
HUMAN_X = -8.6


def _forklift_y(t):
    """Deterministic forklift schedule (32 s period, <= 0.5 m/s)."""
    tt = t % 32.0
    if tt < 5.0:
        return -5.0 - 2.4 * (tt / 5.0)           # -5.0 -> -7.4 (0.48 m/s)
    if tt < 9.0:
        return -7.4 - 0.4 * ((tt - 5.0) / 4.0)   # -7.4 -> -7.8 (crawl)
    if tt < 11.0:
        return -7.8 - 0.5 * ((tt - 9.0) / 2.0)   # -7.8 -> -8.3
    if tt < 20.0:
        return -8.3                              # parked south bay
    if tt < 28.0:
        return -8.3 + 3.3 * ((tt - 20.0) / 8.0)  # -> -5.0 (0.41 m/s)
    return -5.0                                  # parked rack gap


def _human_y(t):
    """Deterministic worker walk: staging -> across the aisle -> rack."""
    if t < 14.0:
        return -8.3
    y = -8.3 + 0.447 * (t - 14.0)
    return y if y < -5.5 else -5.5


def _child_box(parent_path, name, t, s, color):
    """Unit-cube child prim with its own translate/scale ops (local frame)."""
    p = f"{parent_path}/{name}"
    xform = UsdGeom.Xformable(stage.DefinePrim(p, "Xform"))
    xform.AddTranslateOp().Set(Gf.Vec3f(*t))
    xform.AddScaleOp().Set(Gf.Vec3f(*s))
    cube = UsdGeom.Cube.Define(stage, p + "/Geom")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cube


def add_traffic_entity(root_path, motion, half, tag, kind):
    """Xform root (translate op + kinematic RigidBodyAPI) with primitive
    child geometry. One child carries CollisionAPI so the scripted
    motion moves the visuals and the PhysX body together."""
    root = UsdGeom.Xformable(stage.DefinePrim(root_path, "Xform"))
    t_op = root.AddTranslateOp()
    p0 = np.array(motion(0.0), dtype=np.float64)
    t_op.Set(Gf.Vec3f(float(p0[0]), float(p0[1]), 0.0))
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    try:
        rb.CreateKinematicEnabledAttr(True)
    except Exception:
        root.GetPrim().CreateAttribute(
            "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool
        ).Set(True)

    if kind == "forklift":
        body = _child_box(
            root_path, "Body", (0.0, 0.0, 0.45), (0.9, 1.6, 0.7), (0.95, 0.6, 0.05)
        )
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        _child_box(root_path, "MastL", (-0.3, -0.85, 0.85), (0.09, 0.09, 1.5),
                   (0.25, 0.25, 0.3))
        _child_box(root_path, "MastR", (0.3, -0.85, 0.85), (0.09, 0.09, 1.5),
                   (0.25, 0.25, 0.3))
        _child_box(root_path, "MastBar", (0.0, -0.85, 1.55), (0.69, 0.09, 0.12),
                   (0.25, 0.25, 0.3))
        _child_box(root_path, "ForkL", (-0.24, -1.1, 0.07), (0.16, 0.6, 0.1),
                   (0.7, 0.7, 0.75))
        _child_box(root_path, "ForkR", (0.24, -1.1, 0.07), (0.16, 0.6, 0.1),
                   (0.7, 0.7, 0.75))
        for _wi, (_wx, _wy) in enumerate(
            ((-0.5, 0.55), (0.5, 0.55), (-0.5, -0.55), (0.5, -0.55))
        ):
            w = UsdGeom.Cylinder.Define(stage, f"{root_path}/Wheel{_wi}")
            w.CreateRadiusAttr(0.28)
            w.CreateHeightAttr(0.16)
            w.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.08)])
            wxf = UsdGeom.Xformable(w.GetPrim())
            wxf.AddTranslateOp().Set(Gf.Vec3f(_wx, _wy, 0.28))
            wxf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
    else:
        torso = _child_box(
            root_path, "Torso", (0.0, 0.0, 1.15), (0.45, 0.3, 0.6), (1.0, 0.55, 0.05)
        )
        UsdPhysics.CollisionAPI.Apply(torso.GetPrim())
        head = UsdGeom.Sphere.Define(stage, root_path + "/Head")
        head.CreateRadiusAttr(0.16)
        head.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.65, 0.5)])
        UsdGeom.Xformable(head.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3f(0.0, 0.0, 1.62)
        )
        for _li, _lx in enumerate((-0.12, 0.12)):
            leg = UsdGeom.Cylinder.Define(stage, f"{root_path}/Leg{_li}")
            leg.CreateRadiusAttr(0.09)
            leg.CreateHeightAttr(0.85)
            leg.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.15, 0.25)])
            UsdGeom.Xformable(leg.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3f(_lx, 0.0, 0.425)
            )
            # The forward/sector cones only use rays within +-3 deg of
            # the z=0.38 mount - the legs are what the scan hits.
            UsdPhysics.CollisionAPI.Apply(leg.GetPrim())
        for _ai, _ax in enumerate((-0.27, 0.27)):
            arm = UsdGeom.Cylinder.Define(stage, f"{root_path}/Arm{_ai}")
            arm.CreateRadiusAttr(0.06)
            arm.CreateHeightAttr(0.55)
            arm.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.55, 0.05)])
            UsdGeom.Xformable(arm.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3f(_ax, 0.0, 1.2)
            )

    rec = {
        "path": root_path,
        "tag": tag,
        "kind": kind,
        "motion": motion,
        "translate_op": t_op,
        "center": p0.copy(),
        "half": np.array(half, dtype=np.float64),
        "rigid": None,
        "driver": "usd_only",
        "detected": False,
        "reacted": False,
        "cleared": False,
        "motion_len": 0.0,
    }
    try:
        rec["rigid"] = RigidPrim(root_path, reset_xform_properties=False)
        rec["driver"] = "usd+physics_view"
    except Exception as exc:
        rec["driver"] = f"usd_only ({type(exc).__name__})"
    return rec


def update_traffic(t):
    """Drive both traffic entities to their deterministic pose at time t.
    Called BEFORE simulation_app.update() so this frame's scan sees them."""
    for o in traffic_entities:
        x, y = o["motion"](t)
        o["motion_len"] += math.hypot(x - o["center"][0], y - o["center"][1])
        o["center"][0] = x
        o["center"][1] = y
        o["translate_op"].Set(Gf.Vec3f(float(x), float(y), 0.0))
        if o["rigid"] is not None:
            try:
                o["rigid"].set_world_poses(
                    positions=np.array([[x, y, 0.0]], dtype=np.float32),
                    orientations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                )
            except Exception:
                o["rigid"] = None  # kinematic USD follow remains active


traffic_entities = []
_fork = add_traffic_entity(
    "/World/Forklift", lambda t: (FORK_X, _forklift_y(t)), (0.45, 0.8),
    "FORKLIFT", "forklift",
)
_fork["idx"] = 0
traffic_entities.append(_fork)
_hum = add_traffic_entity(
    "/World/Pedestrian", lambda t: (HUMAN_X, _human_y(t)), (0.225, 0.15),
    "HUMAN", "human",
)
_hum["idx"] = 1
traffic_entities.append(_hum)

for _o in traffic_entities:
    _x0, _y0 = _o["motion"](0.0)
    log_print(
        f"TRAFFIC_{_o['tag']}: {_o['path']} POSITION(t=0): "
        f"({_x0:.3f}, {_y0:.3f}, 0.000) AABB half {_o['half'].tolist()} "
        f"DRIVER: {_o['driver']}"
    )
log_print(
    "TRAFFIC_SCHEDULE: forklift x=-4.4 descend/crawl/park/return (32 s cycle); "
    "pedestrian x=-8.6 walk 14.0s -> 20.3s then park at y=-5.5 (0.447 m/s)"
)

# --- forklift visual -> NVIDIA SimReady Forklift_A01 ---------------
# The kinematic root, its translate op, the Body collision box, the
# traffic schedule and the LiDAR scan are untouched; only the primitive
# render geometry (body / masts / forks / wheels) is swapped for the
# real forklift USD - and only after that USD verified + measured, so a
# failed download leaves the original primitive forklift on screen.
# Fit target = the primitive's own visual envelope:
#   x +/-0.580, y -1.400..+0.830, z 0..1.610  ->  centre (0, -0.285)
vis_add(
    "forklift", FORKLIFT_USD, "/World/Forklift/Vis",
    (1.16, 2.23, 1.61), (0.0, -0.285, 0.0),
    mode="uniform_xy", anchor="floor",
    hide=(
        "/World/Forklift/Body/Geom",
        "/World/Forklift/MastL/Geom",
        "/World/Forklift/MastR/Geom",
        "/World/Forklift/MastBar/Geom",
        "/World/Forklift/ForkL/Geom",
        "/World/Forklift/ForkR/Geom",
        "/World/Forklift/Wheel0",
        "/World/Forklift/Wheel1",
        "/World/Forklift/Wheel2",
        "/World/Forklift/Wheel3",
    ),
)
vis_apply()
log_print(
    "NVIDIA_VISUAL_UPGRADE: "
    f"racks={VIS_STATS['rack']} pallets={VIS_STATS['pallet']} "
    f"cartons={VIS_STATS['carton']} walls={VIS_STATS['wall']} "
    f"floor={VIS_STATS['floor']} props={VIS_STATS['prop']} "
    f"forklift={VIS_STATS['forklift']} stock={VIS_STATS['stock']} "
    f"workers={VIS_STATS['human']} "
    f"skipped={VIS_STATS['skipped']} "
    f"colliders_disabled={VIS_STATS['col']} "
    "(render only - collision_boxes, LiDAR masks and every threshold unchanged)"
)

# ============================================================
# STEP 7c: M17 TASKS + greedy nearest-available-robot allocation
# ============================================================
PICKUP_DWELL_S = 3.0     # visible logical pickup dwell
DELIVERY_DWELL_S = 1.5   # visible logical delivery dwell

TASKS = [
    {
        "id": "TASK_001",
        "pickup": (-6.0, -8.0),      # staging area (SW)
        "delivery": (-4.6, -2.9),    # loading zone (middle aisle)
        "status": "PENDING",
        "assigned_robot": None,
    },
    {
        "id": "TASK_002",
        "pickup": (3.2, -6.8),       # east box (corridor east end)
        "delivery": (-6.5, -8.2),    # staging area (SW)
        "status": "PENDING",
        "assigned_robot": None,
    },
    {
        "id": "TASK_003",
        "pickup": (1.2, -7.6),       # south box
        "delivery": (0.6, -1.5),     # north aisle (through row-S gap)
        "status": "PENDING",
        "assigned_robot": None,
    },
]

# Greedy nearest-available-robot-to-pickup, tasks processed in order.
robot_start_xy = [(sx, sy) for (_rp, sx, sy) in ROBOT_SPECS]
robot_busy = [False] * len(ROBOT_SPECS)
for task in TASKS:
    best_i, best_d = None, float("inf")
    for i, (rx, ry) in enumerate(robot_start_xy):
        if robot_busy[i]:
            continue
        dd = math.hypot(task["pickup"][0] - rx, task["pickup"][1] - ry)
        if dd < best_d:
            best_i, best_d = i, dd
    task["assigned_robot"] = best_i
    task["pickup_distance"] = best_d
    robot_busy[best_i] = True
    log_print(
        f"TASK_ALLOCATION: {task['id']} pickup "
        f"({task['pickup'][0]:.3f},{task['pickup'][1]:.3f}) -> "
        f"CARTER_{best_i + 1} (nearest available, dist {best_d:.3f} m)"
    )

# Per-robot first goal = its assigned pickup; face the nose at it.
first_goals = []
for i, task in enumerate(TASKS):
    ti = next(t for t in TASKS if t["assigned_robot"] == i)
    sx, sy = robot_start_xy[i]
    gx, gy = ti["pickup"]
    yaw_deg = math.degrees(math.atan2(gy - sy, gx - sx))
    first_goals.append((gx, gy, yaw_deg))
    articulations[i].set_world_pose(
        position=[sx, sy, 0.1], orientation=yaw_quat(yaw_deg)
    )
    articulations[i].set_world_velocity([0, 0, 0, 0, 0, 0])

for _ in range(10):
    simulation_app.update()

# Per-robot runtime forward direction (chassis +X in world) + start logs.
for i, (_rp, sx, sy) in enumerate(ROBOT_SPECS):
    _p, _r = articulations[i].get_world_pose()
    _y = yaw_of(_r)
    log_print(
        f"CARTER_{i + 1}_START: {_rp} POSITION: ({sx:.3f}, {sy:.3f}, 0.100) "
        f"YAW: {_y:.2f} deg (faces its first goal)"
    )
    try:
        _ch = get_prim_at_path(f"{_rp}/chassis_link")
        _m = UsdGeom.Xformable(_ch).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        _f3 = _m.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        _fv = np.array([float(_f3[0]), float(_f3[1])], dtype=np.float64)
        _n = np.linalg.norm(_fv)
        _fv = _fv / _n if _n > 1e-6 else np.array([-1.0, 0.0])
    except Exception as e:
        log_print(f"CARTER_{i + 1}_FORWARD_FALLBACK ({e}): using (-1, 0)")
        _fv = np.array([-1.0, 0.0])
    _want = first_goals[i][2]
    _diff = float((_y - _want + 180.0) % 360.0 - 180.0)
    log_print(
        f"CARTER_{i + 1}_FORWARD_WORLD_DIR: ({_fv[0]:.4f}, {_fv[1]:.4f}) "
        f"heading_error_vs_first_goal: {_diff:.2f} deg"
    )

# ============================================================
# STEP 7d: Visual markers (NO collision geometry)
#   per-robot start marker + state marker + trajectory trail,
#   per-task pickup/delivery markers (visible assignment)
# ============================================================
START_MARKER_COLOR = (0.15, 0.4, 1.0)
STATE_COLORS = {
    "NAVIGATE": (0.1, 0.9, 0.2),
    "APPROACH": (0.95, 0.8, 0.1),
    "AVOID": (1.0, 0.45, 0.0),
    "AVOID_LEFT": (1.0, 0.6, 0.0),
    "AVOID_RIGHT": (1.0, 0.2, 0.6),
    "PASS_OBSTACLE": (0.2, 0.5, 1.0),
    "RESUME": (0.75, 0.3, 0.9),
    "RECOVERY": (1.0, 0.08, 0.08),
    "REASSESS": (0.1, 0.85, 0.9),
    "GOAL_REACHED": (1.0, 1.0, 1.0),
}
# M17 task-phase colors (floating marker + pickup/delivery squares)
TASK_PHASE_COLORS = {
    "IDLE": (0.75, 0.75, 0.75),
    "TASK_ASSIGNED": (0.6, 0.2, 0.9),
    "GO_TO_PICKUP": (0.1, 0.7, 1.0),
    "PICKUP": (1.0, 0.55, 0.0),
    "GO_TO_DELIVERY": (0.5, 0.9, 0.3),
    "DELIVERY": (0.1, 0.9, 0.4),
    "TASK_COMPLETE": (1.0, 1.0, 1.0),
}
TASK_MARKER_COLORS = {
    "TASK_001": (0.9, 0.25, 0.25),
    "TASK_002": (0.25, 0.45, 0.95),
    "TASK_003": (0.7, 0.25, 0.9),
}
TRAJ_COLORS = ((0.05, 0.95, 1.0), (1.0, 0.85, 0.1), (1.0, 0.25, 0.9))

for i, (_rp, sx, sy) in enumerate(ROBOT_SPECS):
    add_warehouse_box(
        f"/World/StartMarker_{i + 1}",
        (sx, sy, 0.025), (0.6, 0.6, 0.05), START_MARKER_COLOR, collision=False,
    )

# Task pickup (solid) + delivery (square + flag pole) markers
for task in TASKS:
    col = TASK_MARKER_COLORS[task["id"]]
    add_warehouse_box(
        f"/World/{task['id']}_PickupMarker",
        (task["pickup"][0], task["pickup"][1], 0.03),
        (0.9, 0.9, 0.06), col, collision=False,
    )
    add_warehouse_box(
        f"/World/{task['id']}_DeliveryMarker",
        (task["delivery"][0], task["delivery"][1], 0.03),
        (1.0, 1.0, 0.06), col, collision=False,
    )
    add_warehouse_box(
        f"/World/{task['id']}_DeliveryFlag",
        (task["delivery"][0] + 0.7, task["delivery"][1] + 0.7, 0.6),
        (0.1, 0.1, 1.2), col, collision=False,
    )
log_print(
    "TASK_MARKERS: pickup/delivery squares colored per task "
    "(red=TASK_001, blue=TASK_002, purple=TASK_003)"
)

# ============================================================
# STEP 8: ONE navigation OmniGraph per Carter (proven pipeline)
# ============================================================
keys = og.Controller.Keys
graphs = []
for i, (_rp, sx, sy) in enumerate(ROBOT_SPECS):
    graph_path = f"/ActionGraph_{i + 1}"
    gx, gy, _gyaw = first_goals[i]
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
                ("QuinticPathPlanner.inputs:targetPosition", (float(gx), float(gy), 0.0)),
                ("QuinticPathPlanner.inputs:targetOrientation", (0, 0, 0, 1)),
                ("GetPrimLocalToWorldTransform.inputs:usePath", False),
                (graph_path + "/computeOdom.inputs:chassisPrim", [usdrt.Sdf.Path(f"{_rp}/chassis_link")]),
                ("GetPrimLocalToWorldTransform.inputs:prim", [usdrt.Sdf.Path(f"{_rp}/chassis_link")]),
                ("DifferentialController.inputs:wheelRadius", 0.14),
                ("DifferentialController.inputs:wheelDistance", 0.4132),
                ("ArticulationController.inputs:jointNames", ["joint_wheel_left", "joint_wheel_right"]),
                ("ArticulationController.inputs:robotPath", _rp),
            ],
        },
    )
    log_print(f"OmniGraph created successfully: {graph_path} (robot {_rp})")
    og.Controller.attribute(graph_path + "/DifferentialController.inputs:linearVelocity").set(0)
    og.Controller.attribute(graph_path + "/DifferentialController.inputs:angularVelocity").set(0)
    for _ in range(10):
        simulation_app.update()
    # Re-target the planner to this Carter's ACTUAL chassis orientation.
    chassis_quat = tuple(
        og.Controller.attribute(graph_path + "/GetRotationQuaternion.outputs:quaternion").get()
    )
    log_print(f"CARTER_{i + 1}_CHASSIS_ORIENTATION: {chassis_quat}")
    og.Controller.attribute(
        graph_path + "/QuinticPathPlanner.inputs:targetOrientation"
    ).set(chassis_quat)
    for _ in range(10):
        simulation_app.update()
    graphs.append(graph_path)

for _ in range(10):
    simulation_app.update()

# ============================================================
# STEP 8b: PHASE 3A/3C - ONE ROS 2 ACTION GRAPH PER CARTER
# ------------------------------------------------------------
# Target architecture (Phase 3C - comms + per-Carter handover, NO Nav2,
# NO SLAM):
#
#   Isaac Sim
#      |
#      +-- Carter_1   (/World/Carter_1, own articulation + PhysX LiDAR)
#      +-- Carter_2   (/World/Carter_2, own articulation + PhysX LiDAR)
#      +-- Carter_3   (/World/Carter_3, own articulation + PhysX LiDAR)
#      |
#      +-- ROS 2 bridge  (isaacsim.ros2.bridge OmniGraph nodes)
#             |
#             +-- /ROS2ActionGraph_1 -> /carter1/{cmd_vel,odom,scan,tf}
#             +-- /ROS2ActionGraph_2 -> /carter2/{cmd_vel,odom,scan,tf}
#             +-- /ROS2ActionGraph_3 -> /carter3/{cmd_vel,odom,scan,tf}
#
# Exactly 12 topics, 4 per Carter:
#   cmd_vel  geometry_msgs/msg/Twist      subscribe
#   odom     nav_msgs/msg/Odometry        publish
#   scan     sensor_msgs/msg/LaserScan    publish
#   tf       tf2_msgs/msg/TFMessage       publish
#
# Three SEPARATE ROS 2 action graphs are used on purpose:
# /ActionGraph_1..3 - the proven Stanley + QuinticPathPlanner +
# DifferentialController pipeline that drives all three Carters - is not
# touched at all until (and unless) that Carter's own first Twist
# arrives, so Phase-2 navigation, M16 avoidance, safety thresholds, task
# allocation and the collision boxes cannot drift.
#
# Each graph is the byte-for-byte Phase-3A/3B 9-node architecture, only
# re-pointed at that Carter's chassis/lidar prim and namespace:
#   OnPlaybackTick, IsaacReadSimulationTime, IsaacComputeOdometry,
#   ROS2PublishOdometry, IsaacReadLidarBeams, ROS2PublishLaserScan,
#   ROS2PublishRawTransformTree (odom->base_link),
#   ROS2PublishRawTransformTree (base_link-><ns>_laser),
#   ROS2SubscribeTwist
#
# Reads (identical pattern per Carter):
#   odom  <- IsaacComputeOdometry on /World/Carter_N/chassis_link
#            (the same node type and chassis prim /ActionGraph_N uses)
#   scan  <- IsaacReadLidarBeams on /World/Carter_N/chassis_link/lidar
#            (the SAME RotatingLidarPhysX sensor the fleet already uses;
#             reading it a second time does not modify it - the fleet
#             keeps reading get_current_frame() exactly as before).
#            LaserScan is 2D and ROS2PublishLaserScan rejects numRows
#            != 1, so _ros2_feed_scan() slices out that Carter's single
#            ring whose elevation is closest to 0 deg (plus its
#            intensities ring) and feeds it to the publisher each step.
#            Verified numbers are unchanged: 360 samples,
#            angle_min=-3.1416, angle_max=3.1241,
#            angle_increment=0.017453, ring 7/16 at elevation ~-1.0 deg.
#   tf    <- odom -> base_link from the odometry sample, plus the fixed
#            base_link -> <ns>_laser mount (-0.06, 0, 0.38)
#
# Frame ids stay the Phase-3A/3B convention (odom / base_link / <ns>_laser)
# because every Carter publishes onto its OWN namespaced tf topic, so the
# three trees never share a channel.
# ============================================================
ROS2_LIDAR_MOUNT = [-0.06, 0.0, 0.38]                     # RotatingLidarPhysX
ROS2_GRAPHS = []

if ros2["ext_ok"]:
    _k = og.Controller.Keys
    for _i, (_rp, _sx, _sy) in enumerate(ROBOT_SPECS):
        _ns = ROS2_ROBOTS[_i]
        _bot = ros2["bots"][_i]
        _chassis = f"{_rp}/chassis_link"
        _lidar = f"{_chassis}/lidar"
        _gpath = f"/ROS2ActionGraph_{_i + 1}"
        _laser_frame = _bot["laser_frame"]
        _bot["robot"] = _rp
        _bot["chassis"] = _chassis
        _bot["lidar_prim"] = _lidar
        _bot["graph"] = _gpath
        try:
            og.Controller.edit(
                {"graph_path": _gpath, "evaluator_name": "execution"},
                {
                    _k.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                        ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                        ("ReadLidar", "isaacsim.sensors.physx.IsaacReadLidarBeams"),
                        ("PublishScan", "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
                        ("TfOdomBase", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                        ("TfBaseLaser", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                        ("SubscribeCmdVel", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                    ],
                    _k.SET_VALUES: [
                        # --- /<ns>/odom ---
                        ("ComputeOdom.inputs:chassisPrim",
                         [usdrt.Sdf.Path(_chassis)]),
                        ("PublishOdom.inputs:topicName", f"{_ns}/odom"),
                        ("PublishOdom.inputs:odomFrameId", "odom"),
                        ("PublishOdom.inputs:chassisFrameId", "base_link"),
                        ("PublishOdom.inputs:publishRawVelocities", False),
                        # --- /<ns>/scan ---
                        # ROS2PublishLaserScan is a 2D sensor_msgs/LaserScan
                        # publisher and validates numRows == 1 on every tick:
                        #   "Number of rows must be equal to 1. High LOD not
                        #    supported for LaserScan, only 2D Lidar Supported
                        #    for LaserScan."
                        # The fleet's RotatingLidarPhysX is 360 x 16 (Phase-2
                        # high LOD), so linearDepthData / intensitiesData /
                        # numRows are deliberately NOT connected here -
                        # _ros2_feed_scan() hands the node this Carter's
                        # single elevation~0 ring each step instead.
                        ("ReadLidar.inputs:lidarPrim",
                         [usdrt.Sdf.Path(_lidar)]),
                        ("PublishScan.inputs:topicName", f"{_ns}/scan"),
                        ("PublishScan.inputs:frameId", _laser_frame),
                        ("PublishScan.inputs:numRows", 1),
                        # --- /<ns>/tf ---
                        ("TfOdomBase.inputs:topicName", f"{_ns}/tf"),
                        ("TfOdomBase.inputs:parentFrameId", "odom"),
                        ("TfOdomBase.inputs:childFrameId", "base_link"),
                        ("TfBaseLaser.inputs:topicName", f"{_ns}/tf"),
                        ("TfBaseLaser.inputs:parentFrameId", "base_link"),
                        ("TfBaseLaser.inputs:childFrameId", _laser_frame),
                        ("TfBaseLaser.inputs:translation", list(ROS2_LIDAR_MOUNT)),
                        ("TfBaseLaser.inputs:rotation", [0.0, 0.0, 0.0, 1.0]),
                        # --- /<ns>/cmd_vel (observe only, see below) ---
                        ("SubscribeCmdVel.inputs:topicName", f"{_ns}/cmd_vel"),
                    ],
                    _k.CONNECT: [
                        # odometry chain
                        ("OnTick.outputs:tick", "ComputeOdom.inputs:execIn"),
                        ("ComputeOdom.outputs:execOut", "PublishOdom.inputs:execIn"),
                        ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
                        ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
                        ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
                        ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
                        ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
                        # laser scan chain
                        ("OnTick.outputs:tick", "ReadLidar.inputs:execIn"),
                        ("ReadLidar.outputs:execOut", "PublishScan.inputs:execIn"),
                        ("ReadLidar.outputs:azimuthRange", "PublishScan.inputs:azimuthRange"),
                        ("ReadLidar.outputs:depthRange", "PublishScan.inputs:depthRange"),
                        ("ReadLidar.outputs:horizontalFov", "PublishScan.inputs:horizontalFov"),
                        ("ReadLidar.outputs:horizontalResolution", "PublishScan.inputs:horizontalResolution"),
                        ("ReadLidar.outputs:numCols", "PublishScan.inputs:numCols"),
                        ("ReadLidar.outputs:rotationRate", "PublishScan.inputs:rotationRate"),
                        # NOT connected on purpose (fed by _ros2_feed_scan):
                        #   ReadLidar.outputs:linearDepthData   -> numRows is 16
                        #   ReadLidar.outputs:intensitiesData   -> same reason
                        #   ReadLidar.outputs:numRows           -> pinned to 1
                        ("ReadSimTime.outputs:simulationTime", "PublishScan.inputs:timeStamp"),
                        # tf: odom -> base_link (same sample as the odometry)
                        ("ComputeOdom.outputs:execOut", "TfOdomBase.inputs:execIn"),
                        ("ComputeOdom.outputs:position", "TfOdomBase.inputs:translation"),
                        ("ComputeOdom.outputs:orientation", "TfOdomBase.inputs:rotation"),
                        ("ReadSimTime.outputs:simulationTime", "TfOdomBase.inputs:timeStamp"),
                        # tf: base_link -> <ns>_laser (fixed mount)
                        ("ComputeOdom.outputs:execOut", "TfBaseLaser.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "TfBaseLaser.inputs:timeStamp"),
                        # cmd_vel in (terminal -> sim); outputs are polled only
                        ("OnTick.outputs:tick", "SubscribeCmdVel.inputs:execIn"),
                    ],
                },
            )
            _bot["enabled"] = True
            ROS2_GRAPHS.append(_gpath)
            log_print(
                f"PHASE3_ROS2_GRAPH: {_gpath} created (9 nodes) ns=/{_ns}"
            )
            log_print(
                f"PHASE3_ROS2_TARGET: robot={_rp} chassis={_chassis} "
                f"lidar={_lidar}"
            )
            # handles used by _ros2_step() for cheap per-step observation
            _bot["cmd_attr"] = og.Controller.attribute(
                _gpath + "/SubscribeCmdVel.outputs:linearVelocity"
            )
            _bot["lidar_cols_attr"] = og.Controller.attribute(
                _gpath + "/ReadLidar.outputs:numCols"
            )
            _bot["lidar_rows_attr"] = og.Controller.attribute(
                _gpath + "/ReadLidar.outputs:numRows"
            )
            _bot["scan_lin_out"] = og.Controller.attribute(
                _gpath + "/ReadLidar.outputs:linearDepthData"
            )
            _bot["scan_int_out"] = og.Controller.attribute(
                _gpath + "/ReadLidar.outputs:intensitiesData"
            )
            _bot["scan_lin_in"] = og.Controller.attribute(
                _gpath + "/PublishScan.inputs:linearDepthData"
            )
            _bot["scan_int_in"] = og.Controller.attribute(
                _gpath + "/PublishScan.inputs:intensitiesData"
            )
        except Exception as e:
            ros2["errors"].append(f"action graph {_gpath}: {e}")
            log_print(f"PHASE3_ROS2_ERROR: action graph creation failed: {e}")

    ros2["graphs"] = list(ROS2_GRAPHS)
    ros2["enabled"] = all(b["enabled"] for b in ros2["bots"])
    if ros2["errors"]:
        # all-or-nothing: one broken graph means the 12-topic contract is
        # not met, so the bridge is treated as failed rather than partial.
        ros2["enabled"] = False
        log_print("PHASE3_ROS2_ERRORS: " + " | ".join(ros2["errors"]))
    log_print(
        "PHASE3_ROS2_TOPICS: "
        + " ".join(f"{t}({m},{d})" for t, m, d in ROS2_TOPIC_TYPES)
    )
    log_print(
        "PHASE3_ROS2_CMD_VEL: all three subscribed for verification only - "
        "NOT wired into DifferentialController; the Phase-2 drive stack "
        "stays intact until a Carter's own first Twist arrives"
    )


def _teleop_envelope(a0):
    """PHASE 3B/3C: additive safety envelope for a ROS-2-commanded Carter.

    Called once per ENGAGED Carter (Phase 3C) / for Carter 1 (Phase 3B)
    with that robot's own agent, so it is inherently per robot.

    Returns ``None`` when the commanded motion is allowed, otherwise a
    short reason string and the caller forces a stop.

    The three numbers enforced here are EXACTLY the thresholds Phase 2
    already grades the whole run on, so teleoperation can never make the
    Phase-2 safety verdict fail:

      * forward LiDAR cone   >= MIN_FORWARD_ALLOWED (1.0 m)
      * static/traffic AABB  >= MIN_CLEARANCE_CENTER (0.5 m)
      * robot-robot rect gap >= 0.5 m  (the MIN_PAIR_GAP requirement)

    This only ever ADDS a stop; it never relaxes, bypasses or replaces a
    Phase-2 check, and it uses the same helpers (aabb_distance / rect_gap)
    and the same live LiDAR frame the rest of the script uses.
    """
    if a0._d is None:
        return "no LiDAR frame"
    if a0.forward_dist is None or a0.forward_dist < MIN_FORWARD_ALLOWED:
        return (f"forward {a0.forward_dist if a0.forward_dist is not None else -1.0:.4f}"
                f" < {MIN_FORWARD_ALLOWED}")
    step_min = float("inf")
    for _n, _c, _h in collision_boxes:
        dc = aabb_distance(a0.pos[:2], _c, _h)
        if dc < step_min:
            step_min = dc
    for _o in traffic_entities:
        dc = aabb_distance(a0.pos[:2], _o["center"], _o["half"])
        if dc < step_min:
            step_min = dc
    if step_min < MIN_CLEARANCE_CENTER:
        return f"clearance {step_min:.4f} < {MIN_CLEARANCE_CENTER}"
    for _aj in agents:
        if _aj is a0:
            continue
        g = rect_gap(a0.pos, a0.yaw, a0.half, _aj.pos, _aj.yaw, _aj.half)
        if g < MIN_CLEARANCE_CENTER:
            return f"robot gap {g:.4f} < {MIN_CLEARANCE_CENTER}"
    return None


def _ros2_handover_engage(step):
    """PHASE 3C: for EACH Carter independently, detach that Carter's
    Stanley -> DifferentialController edge, ONCE, the first time a Twist
    arrives on THAT Carter's /carterN/cmd_vel.

    Until this fires for a given Carter nothing about that robot differs
    from Phase 2 - same graph, same connections, same planner output - so
    /carter2/cmd_vel can never move Carter 1 and /carter3/cmd_vel can
    never move Carter 2.  Reaching here needs BOTH gates:
    PHASE3C_HANDOVER=1 (arming) AND at least one actual received Twist
    for this specific robot.
    """
    if not (PHASE3C_HANDOVER and ros2.get("enabled")):
        return
    for i, a in enumerate(agents):
        if i >= len(ros2["bots"]):
            break
        bot = ros2["bots"][i]
        if not bot["enabled"]:
            continue
        if bot["cmd_rx"] == 0:
            continue                      # THIS robot has not been commanded
        if a.graph_path in ROS2_HANDOVER_GRAPHS:
            continue                      # already engaged, once only
        og.Controller.edit(
            a.graph_path,
            {keys.DISCONNECT: [(a.LIN_SRC, a.LIN_DST),
                               (a.ANG_SRC, a.ANG_DST)]},
        )
        ROS2_HANDOVER_GRAPHS.add(a.graph_path)
        _others = ", ".join(
            agents[j].tag for j in range(len(agents)) if j != i
        )
        log_print(
            f"PHASE3C_HANDOVER: ENGAGED at step {step} robot={a.tag} "
            f"ns=/{bot['ns']} after {bot['cmd_rx']} /{bot['ns']}/cmd_vel "
            f"message(s) - Stanley->DifferentialController detached on "
            f"{a.graph_path}; /{bot['ns']}/cmd_vel owns THIS Carter's "
            f"command from here ({_others} still on their Phase-2 control "
            f"path, QuinticPathPlanner + Stanley still running, "
            f"manual_mode safety latch unchanged)"
        )


def _ros2_handover_apply(step, a, bot):
    """PHASE 3C: write the last /carterN/cmd_vel into ONE Carter's
    DifferentialController - the exact slot StanleyControlPID drove.

    Priority order (highest first):

    1. Phase-2 safety.  ``manual_mode`` is the same latch the reactive
       machine already uses, so ROS 2 stands down in APPROACH / AVOID /
       PASS_OBSTACLE / RECOVERY / REASSESS, on a LiDAR-frame gap and on
       the M16 robot-to-robot emergency stop - i.e. in exactly the states
       where Phase 2 takes the command away from Stanley.  The one
       exception is GOAL_REACHED, which is Phase-2's *task* stop for a
       parked robot, not a safety hold, and only while a real LiDAR frame
       exists so the frame-unavailable hold still wins.
    2. The additive teleop envelope in ``_teleop_envelope`` (1.0 m forward,
       0.5 m clearance, 0.5 m robot gap).

    Because this runs AFTER navigate(), a safety command issued this step
    is already in place and is never overwritten by the operator.
    Nothing here disables, weakens or bypasses a check, and no threshold,
    task coordinate or LiDAR setting is touched.  ``bot`` carries this
    robot's OWN last Twist and OWN telemetry, so two Carters can never
    read each other's command.
    """
    cl = bot.get("cmd_last")            # (lin.x, lin.y, ang.z) or None
    v = float(cl[0]) if cl else 0.0     # no Twist seen yet -> hold still
    w = float(cl[2]) if cl else 0.0

    # --- 1. Phase-2 safety has absolute priority ---
    _task_park = (a.state == "GOAL_REACHED" and a._d is not None)
    if a.manual_mode and not _task_park:
        bot["handover_safety_holds"] += 1
        if not bot["handover_held"]:
            bot["handover_held"] = True
            log_print(
                f"PHASE3C_HANDOVER_SAFETY_TAKEOVER: step={step} "
                f"robot={a.tag} state={a.state} - Phase-2 machine holds "
                f"the command, ROS 2 stands down"
            )
        return

    if bot["handover_held"]:
        bot["handover_held"] = False
        log_print(
            f"PHASE3C_HANDOVER_SAFETY_RELEASE: step={step} "
            f"robot={a.tag} state={a.state} - ROS 2 has the command again"
        )

    # --- 2. additive teleop safety envelope ---
    env = None
    if v != 0.0 or w != 0.0:
        env = _teleop_envelope(a)
    if env is not None:
        v, w = 0.0, 0.0
        bot["handover_env_stops"] += 1
        if not bot.get("handover_env_logged"):
            bot["handover_env_logged"] = True
            log_print(
                f"PHASE3C_HANDOVER_ENVELOPE: step={step} robot={a.tag} "
                f"forcing STOP - {env} (1.0 m forward / 0.5 m clearance / "
                f"0.5 m robot gap - the Phase-2 safety numbers)"
            )

    a.write_cmd(v, w)
    bot["handover_writes"] += 1
    # count how many RECEIVED cmd_vel messages actually reached the wheels
    if bot["cmd_rx"] != bot.get("handover_applied_msgs_seen"):
        bot["handover_applied_msgs"] += 1
        bot["handover_applied_msgs_seen"] = bot["cmd_rx"]

    key = (round(v, 6), round(w, 6))
    if key != bot["handover_last"]:
        bot["handover_last"] = key
        bot["handover_changes"] += 1
        if bot["handover_changes"] <= 12:
            log_print(
                f"PHASE3C_HANDOVER_CMD: #{bot['handover_changes']} "
                f"robot={a.tag} ns=/{bot['ns']} lin.x={v:.3f} "
                f"angular.z={w:.3f} applied at step {step} "
                f"(state={a.state})"
            )

    # --- per-second telemetry: authority, command, ACTUAL measured motion ---
    if bot["handover_writes"] % 60 == 1:
        log_print(
            f"PHASE3C_HANDOVER_TELE: step={step} robot={a.tag} "
            f"state={a.state} authority=ROS2 "
            f"cmd_lin.x={v:.3f} cmd_ang.z={w:.3f} "
            f"measured_speed={a.speed:.3f} "
            f"pos=({a.pos[0]:.3f},{a.pos[1]:.3f}) "
            f"forward={a.forward_dist if a.forward_dist is not None else -1.0:.4f} "
            f"cmd_vel_rx={bot['cmd_rx']}"
        )


def _ros2_handover_step(step):
    """PHASE 3C: apply every ENGAGED Carter's own last Twist - and only
    that Carter's.  A robot whose graph is not in ROS2_HANDOVER_GRAPHS is
    not touched here at all, so one Carter's handover can never influence
    another Carter.
    """
    if not (PHASE3C_HANDOVER and ros2.get("enabled")):
        return
    if not ROS2_HANDOVER_GRAPHS:
        return
    for i, a in enumerate(agents):
        if a.graph_path not in ROS2_HANDOVER_GRAPHS:
            continue
        if i >= len(ros2["bots"]):
            break
        _ros2_handover_apply(step, a, ros2["bots"][i])


def _ros2_monitor_start():
    """Start an independent in-process ROS 2 subscriber node.

    This is a SECOND participant on the DDS bus: it receives the very
    messages the bridge publishes and records what really crossed the
    wire, so 'the topic name exists' can never be mistaken for 'the
    topic works'.  It subscribes only - it never publishes and never
    touches the simulation.
    """
    if not ros2["enabled"]:
        return
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import (
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from sensor_msgs.msg import LaserScan
        from tf2_msgs.msg import TFMessage
    except Exception as e:
        ros2["errors"].append(f"rclpy import: {e}")
        log_print(f"PHASE3_ROS2_MONITOR_ERROR: rclpy import failed: {e}")
        return

    try:
        # The bridge extension already ran rclpy.init()/shutdown() on the
        # DEFAULT context once, and rclpy forbids re-initialising a context
        # ("Context.init() must only be called once").  A private context
        # sidesteps that and keeps the monitor off the default signal
        # handlers, which belong to Isaac Sim.
        ctx = rclpy.Context()
        rclpy.init(context=ctx)
        node = rclpy.create_node("fleet_ros2_monitor", context=ctx)
        executor = SingleThreadedExecutor(context=ctx)
        executor.add_node(node)
        # BEST_EFFORT is accepted by every publisher (reliable or not),
        # so this subscription matches whatever QoS each node picked.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # 12 subscriptions: odom + scan + tf + cmd_vel for each of the
        # three Carters.  The cmd_vel callback routes the Twist to that
        # Carter's OWN bot entry, so /carter3/cmd_vel can only ever bump
        # agents[2]'s cmd_rx/cmd_last.
        msg_types = {}
        for _b in ros2["bots"]:
            _ns = _b["ns"]
            msg_types[f"/{_ns}/odom"] = Odometry
            msg_types[f"/{_ns}/scan"] = LaserScan
            msg_types[f"/{_ns}/tf"] = TFMessage
            msg_types[f"/{_ns}/cmd_vel"] = Twist
        _bot_by_topic = {f"/{_b['ns']}/cmd_vel": _b for _b in ros2["bots"]}

        def _make_cb(topic):
            _target = _bot_by_topic.get(topic)

            def _cb(msg):
                ros2["counts"][topic] = ros2["counts"].get(topic, 0) + 1
                if topic not in ros2["first"]:
                    ros2["first"][topic] = _ros2_describe(topic, msg)
                if _target is not None:
                    _target["cmd_rx"] += 1
                    _target["cmd_last"] = (
                        msg.linear.x, msg.linear.y, msg.angular.z
                    )
            return _cb

        for topic, mtype in msg_types.items():
            node.create_subscription(mtype, topic, _make_cb(topic), qos)
        ros2["ctx"] = ctx
        ros2["node"] = node
        ros2["executor"] = executor
        log_print(
            "PHASE3_ROS2_MONITOR: node=/fleet_ros2_monitor subs="
            + ",".join(sorted(msg_types))
        )
    except Exception as e:
        ros2["errors"].append(f"monitor: {e}")
        log_print(f"PHASE3_ROS2_MONITOR_ERROR: {e}")


def _ros2_describe(topic, msg):
    """Compact human-checkable summary of the first message per topic."""
    try:
        if topic.endswith("/odom"):
            p, t = msg.pose.pose, msg.twist.twist
            return (
                f"header.frame_id={msg.header.frame_id} "
                f"child_frame_id={msg.child_frame_id} "
                f"pos=({p.position.x:.3f},{p.position.y:.3f},"
                f"{p.position.z:.3f}) "
                f"lin=({t.linear.x:.3f},{t.linear.y:.3f},{t.linear.z:.3f}) "
                f"ang.z={t.angular.z:.3f}"
            )
        if topic.endswith("/scan"):
            finite = sum(1 for r in msg.ranges if np.isfinite(r))
            return (
                f"header.frame_id={msg.header.frame_id} "
                f"len(ranges)={len(msg.ranges)} finite={finite} "
                f"angle=[{msg.angle_min:.4f},{msg.angle_max:.4f}] "
                f"increment={msg.angle_increment:.5f} "
                f"range=[{msg.range_min:.2f},{msg.range_max:.2f}]"
            )
        if topic.endswith("/tf"):
            pairs = [
                f"{tr.header.frame_id}->{tr.child_frame_id}"
                for tr in msg.transforms
            ]
            return "transforms=" + (" ".join(pairs) if pairs else "(none)")
        return f"linear.x={msg.linear.x:.3f} angular.z={msg.angular.z:.3f}"
    except Exception as e:
        return f"<describe error: {e}>"


def _ros2_feed_scan(step, bot):
    """Hand ONE Carter's ROS2PublishLaserScan its horizontal ring.

    ``bot`` is that Carter's entry in ``ros2["bots"]``: its own
    IsaacReadLidarBeams / ROS2PublishLaserScan handles, its own ring
    selection, its own telemetry.  Each Carter reads its OWN
    RotatingLidarPhysX, so the three scans are independent real returns.

    ROS2PublishLaserScan is a *2D* sensor_msgs/LaserScan publisher and
    validates ``numRows == 1`` on every tick::

        "Number of rows must be equal to 1. High LOD not supported for
         LaserScan, only 2D Lidar Supported for LaserScan."

    The fleet's RotatingLidarPhysX is 360 cols x 16 rows (Phase-2 high
    LOD, frozen), so connecting its buffer directly makes the node refuse
    to compute and publish nothing at all - exactly what Phase-3A run 1
    measured (``/carter1/scan = 0`` while odom/tf were healthy).

    IsaacReadLidarBeams still reads the fleet's own sensor (reading it a
    second time does not modify it); here we slice out the ring whose
    elevation is closest to horizontal and push it - plus the matching
    intensities ring, which the node requires to be the same length -
    into PublishScan.  Every number is a real LiDAR return.
    """
    if not bot.get("enabled"):
        return
    lin_out, lin_in = bot.get("scan_lin_out"), bot.get("scan_lin_in")
    int_out, int_in = bot.get("scan_int_out"), bot.get("scan_int_in")
    if lin_out is None or lin_in is None or int_out is None or int_in is None:
        return

    # Buffer geometry, resolved once ReadLidar has produced a frame.
    if bot["scan_cols"] is None:
        try:
            c = int(bot["lidar_cols_attr"].get())
            r = int(bot["lidar_rows_attr"].get())
        except Exception:
            return
        if c <= 0 or r <= 0:
            return
        bot["scan_cols"], bot["scan_rows"] = c, r
        log_print(
            f"PHASE3_ROS2_SCAN_GEOMETRY: ns={bot['ns']} {c} cols x {r} rows"
        )
    n_az, n_ze = bot["scan_cols"], bot["scan_rows"]

    # Pick the ring once, with the SAME zenith->elevation convention
    # build_masks() uses, so the ROS scan and the fleet's own forward
    # mask are looking at the same slice of the same sensor.
    if bot["scan_ring"] is None:
        bot["scan_ring_tries"] += 1
        try:
            fr = lidars[bot["i"]].get_current_frame()
            ze = to_numpy(fr.get("zenith")) if fr else None
            if ze is not None and ze.ndim == 2:
                ze = ze[0, :]
            if ze is None or ze.size != n_ze:
                ze_deg = 90.0 - 15.0 + np.arange(n_ze) * 2.0
            else:
                ze_deg = ze.copy()
                if np.nanmax(np.abs(ze_deg)) <= 6.5:
                    ze_deg = np.degrees(ze_deg)
            elev = 90.0 - ze_deg if np.nanmedian(ze_deg) > 45.0 else ze_deg
            ring = int(np.argmin(np.abs(elev)))
        except Exception as e:
            if bot["scan_ring_tries"] <= 3:
                log_print(f"PHASE3_ROS2_SCAN_RING_ERROR: ns={bot['ns']} {e}")
            return
        bot["scan_ring"] = ring
        bot["scan_elev"] = float(elev[ring])
        log_print(
            f"PHASE3_ROS2_SCAN_RING: ns={bot['ns']} ring={ring}/{n_ze} "
            f"elevation={elev[ring]:+.1f} deg "
            f"(ring elevations {np.round(elev, 1).tolist()})"
        )

    try:
        flat = np.asarray(lin_out.get(), dtype=np.float64)
    except Exception:
        return
    if flat.size != n_az * n_ze:
        return                                  # sensor frame not ready yet

    # IsaacReadLidarBeams flattens this sensor as (n_az, n_ze), i.e. the
    # same layout as lidars[i].get_current_frame()["linear_depth"].
    ring = bot["scan_ring"]
    row = flat.reshape(n_az, n_ze)[:, ring]

    try:
        inten = np.asarray(int_out.get(), dtype=np.float64)
    except Exception:
        inten = np.empty(0)
    if inten.size != flat.size:
        # The node hard-errors on unequal buffers and publishes nothing,
        # so rather than invent intensities we wait for real ones.
        if bot["scan_fed"] == 0 and bot["scan_ring_tries"] <= 3:
            log_print(
                f"PHASE3_ROS2_SCAN_INTENSITIES: ns={bot['ns']} unusable "
                f"size={inten.size} (need {flat.size}) - scan held"
            )
        return
    irow = inten.reshape(n_az, n_ze)[:, ring]

    lin_in.set([float(v) for v in row])
    int_in.set([float(v) for v in irow])
    bot["scan_fed"] += 1

    if bot["scan_fed"] == 1 or step == 30:
        _f = row[np.isfinite(row)]
        log_print(
            f"PHASE3_ROS2_SCAN_ROW: ns={bot['ns']} ring={ring}/{n_ze} "
            f"(elev {bot['scan_elev']:+.1f} deg) -> {row.size} ranges, "
            f"intensities={irow.size}, finite={_f.size}"
            + (
                f", min={_f.min():.2f} max={_f.max():.2f}"
                if _f.size
                else ""
            )
        )


def _ros2_step(step):
    """Called once per fleet step.  Observation only: pumps the monitor
    and records what arrived.  It never writes to the stage, to an
    articulation, or to the control OmniGraphs."""
    if not ros2["enabled"]:
        return
    ex = ros2.get("executor")
    if ex is not None:
        try:
            ex.spin_once(timeout_sec=0)
        except Exception as e:
            ros2["spin_errors"] += 1
            if ros2["spin_errors"] <= 3:
                log_print(f"PHASE3_ROS2_SPIN_ERROR (step {step}): {e}")

    for bot in ros2["bots"]:
        if not bot["enabled"]:
            continue
        # OGN-side view of /<ns>/cmd_vel (the subscriber holds the last
        # value it saw, so a single poll per step is enough).
        if bot.get("cmd_attr") is not None:
            try:
                v = tuple(float(x) for x in bot["cmd_attr"].get())
            except Exception:
                v = None
            if v is not None and v != bot["cmd_ogn_last"]:
                bot["cmd_ogn_last"] = v
                bot["cmd_ogn_changes"] += 1
                if bot["cmd_ogn_changes"] <= 10:
                    log_print(
                        f"PHASE3_ROS2_CMD_VEL_OGN: ns={bot['ns']} "
                        f"#{bot['cmd_ogn_changes']} "
                        f"lin=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f}) "
                        f"at step {step}"
                    )

        # Give ROS2PublishLaserScan this frame's horizontal ring.  The
        # value is written AFTER the tick, so the scan carries a
        # one-frame (16 ms) latency - never a fabricated one.
        _ros2_feed_scan(step, bot)

    # One-shot proof that every IsaacReadLidarBeams really holds a scan.
    if step == 30:
        for bot in ros2["bots"]:
            if not bot["enabled"] or bot.get("lidar_cols_attr") is None:
                continue
            try:
                _c = int(bot["lidar_cols_attr"].get())
                _r = int(bot["lidar_rows_attr"].get())
                log_print(
                    f"PHASE3_ROS2_LIDAR_BUFFER: ns={bot['ns']} cols={_c} "
                    f"rows={_r} (expect 360 x 16)"
                )
            except Exception as e:
                log_print(
                    f"PHASE3_ROS2_LIDAR_BUFFER_ERROR: ns={bot['ns']} {e}"
                )


_ros2_monitor_start()

# ============================================================
# STEP 9: LiDAR analysis helpers (Milestone 5/6 conventions)
# ============================================================
FORWARD_CONE_DEG = 30.0     # forward cone: azimuth +-30 deg
# The ROUTE cone ("is the way to the goal blocked?") is deliberately much
# narrower than the forward cone.  The rack gaps are only 2.0 m wide, so
# the corners FLANKING a gap always sit ~25-32 deg off the goal line at
# ~1.5 m - inside a +-30 deg fan they read as "route blocked" even though
# a +-15 deg fan passes straight through the opening (measured: every
# false AVOID trigger sat 24.7-32.3 deg off route while the centre ray
# landed at x=0.885, 0.187 and -4.982 - all inside their gap).  Acting on
# those pushed CARTER_3's glide into rack S1 (clearance 0.1397).  Only an
# object roughly ACROSS the goal line blocks the route; a corner beside
# the opening does not.
ROUTE_CONE_DEG = 15.0       # route cone: azimuth +-15 deg about the goal
ELEV_CONE_DEG = 3.0         # elevation +-3 deg (floor rays excluded)
SECTOR_DEG_INNER = 30.0     # LEFT  = azimuth +30..+90 deg
SECTOR_DEG_OUTER = 90.0     # RIGHT = azimuth -90..-30 deg
SECTOR_PERCENTILE = 20.0    # robust clearance statistic
SELF_HIT_MAX = 0.5          # exclude the robot's own body


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
    """Build forward-cone, LEFT/RIGHT/rear sector ray masks from the
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
    rear = (np.abs(az) >= REAR_DEG_INNER)[:, None] & elev_ok[None, :]
    # also hand back the wrapped azimuth grid (degrees, sensor frame) so
    # sense() can build a ROUTE-cone mask (rays pointing at the goal)
    # without re-deriving the azimuth convention.
    return fwd, left, right, rear, az


def sector_clearance(d, mask, valid):
    """Robust clearance statistic (p20 of valid ranges) inside a sector."""
    sel = mask & valid & (d > SELF_HIT_MAX)
    if not np.any(sel):
        return 0.0
    return float(np.percentile(d[sel], SECTOR_PERCENTILE))


def sector_min(d, mask, valid):
    """Conservative clearance statistic (minimum valid range) inside a
    sector: always safe, noise-free PhysX scan (verified)."""
    sel = mask & valid & (d > SELF_HIT_MAX)
    if not np.any(sel):
        return 0.0
    return float(np.min(d[sel]))


def ray_aabb_forward(origin_xy, dir_xy, center_xy, half_xy):
    """Slab test: distance along dir_xy to the AABB, or None on miss."""
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


def entity_in_forward_cone(pos_xy, yaw_d, entities, half_angle=30.0):
    """Closest entity (traffic OR another Carter) the forward LiDAR cone
    can actually see: aims a ray at each entity CENTRE and keeps
    entities whose direction lies within +-half_angle of the nose.
    Returns (pred, entity) or None."""
    cf = np.array(
        [math.cos(math.radians(yaw_d)), math.sin(math.radians(yaw_d))],
        dtype=np.float64,
    )
    lidar_xy = np.asarray(pos_xy, dtype=np.float64)[:2] - 0.06 * cf
    best = None
    for o in entities:
        v = np.asarray(o["center"], dtype=np.float64)[:2] - lidar_xy
        n = float(math.hypot(v[0], v[1]))
        if n < 1e-6:
            return (0.0, o)
        az = math.degrees(
            math.atan2(cf[0] * v[1] - cf[1] * v[0], cf[0] * v[0] + cf[1] * v[1])
        )
        if abs(az) > half_angle:
            continue
        p = ray_aabb_forward(lidar_xy, v / n, o["center"], o["half"])
        if p is None:
            continue
        if best is None or p < best[0]:
            best = (p, o)
    return best


def attribute_forward_entity(pos_xy, yaw_d, forward_d, entities):
    """Which entity does the forward cone actually see? The closest
    entity in the cone must explain the forward reading (AABB entry
    within 0.35 m of the measured cone minimum), else None."""
    cand = entity_in_forward_cone(pos_xy, yaw_d, entities)
    if cand is None:
        return None
    pred, o = cand
    if forward_d is None or pred <= forward_d + 0.35:
        return cand
    return None


def resolve_forward_entity(pos_xy, yaw_d, forward_d, entities):
    """Entity DICT for labelling and the PASS "past" geometry.

    attribute_forward_entity() returns a (predicted_distance, entity)
    pair; assigning that pair straight to self.cur_obs made every later
    self.cur_obs["idx"] raise "tuple indices must be integers or
    slices, not str" and killed the run.  Always unwrap to the entity
    here, falling back to the nearest entity (label only - the AVOID
    trigger itself is driven purely by the measured lidar reading).
    """
    cand = attribute_forward_entity(pos_xy, yaw_d, forward_d, entities)
    if cand is not None:
        return cand[1]
    return min(
        entities, key=lambda o: aabb_distance(pos_xy, o["center"], o["half"])
    )


def priority_label(ent):
    """M16 priority concept (labeling only - NO priority planner):
    HUMAN > FORKLIFT > OTHER CARTER > STATIC."""
    kind = ent.get("kind", "static") if ent is not None else "static"
    if kind == "human":
        return "HUMAN"
    if kind == "forklift":
        return "FORKLIFT"
    if kind == "robot":
        return "OTHER_CARTER"
    return "STATIC"


def decide_avoid_side(d, valid, left_mask, right_mask, last_dir,
                      steps_since_dir, path_since_dir):
    """M10/M11 avoidance decision from MEASURED sector clearance.
    Returns (direction, left_clear, right_clear, lock_note). Side is
    computed every event - never hard-coded - with an anti-oscillation
    direction lock (held unless the other side beats it by HYST_MARGIN)."""
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


def rect_gap(pos_a, yaw_a, half_a, pos_b, yaw_b, half_b):
    """Conservative rectangle-to-rectangle gap (m): project each robot's
    oriented box into the other's axis frame (SAT-style), take the
    MINIMUM of the two frame results. Negative = overlap. Used only for
    the M16 robot-vs-robot safety check (static/traffic keep the proven
    point-to-AABB check)."""
    best = float("inf")
    for (p, pyaw, ph, q, qyaw, qh) in (
        (pos_a, yaw_a, half_a, pos_b, yaw_b, half_b),
        (pos_b, yaw_b, half_b, pos_a, yaw_a, half_a),
    ):
        th = math.radians(pyaw)
        c, s = math.cos(th), math.sin(th)
        dx = float(q[0] - p[0])
        dy = float(q[1] - p[1])
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        rel = math.radians(qyaw - pyaw)
        cr, sr = abs(math.cos(rel)), abs(math.sin(rel))
        bx = qh[0] * cr + qh[1] * sr
        by = qh[0] * sr + qh[1] * cr
        gx = abs(lx) - (ph[0] + bx)
        gy = abs(ly) - (ph[1] + by)
        if gx < 0.0 and gy < 0.0:
            gap = max(gx, gy)
        else:
            gap = math.hypot(max(gx, 0.0), max(gy, 0.0))
        if gap < best:
            best = gap
    return float(best)


# ============================================================
# STEP 10: State machine configuration (frozen thresholds)
# ============================================================
CLEAR_T = 2.5          # m: forward distance for CLEAR / NAVIGATE
AVOID_T = 1.5          # m: forward distance that triggers avoidance
EMERG_T = 1.5          # m: M16 robot-vs-robot EMERGENCY zone (stop);
                       # 1.5 < forward <= 2.5 is the SLOW safety zone
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

APPROACH_V = 0.30      # m/s: reduced (SLOW) speed while in APPROACH
APPROACH_K = 1.5       # heading P-gain in APPROACH
APPROACH_W_MAX = 0.6   # rad/s clamp
TURN_V = 0.10          # m/s: near-stop while rotating away
TURN_W = 0.85          # rad/s: reactive turn rate
GLIDE_V = 0.45         # m/s: glide past the obstacle holding the turn heading
PASS_V = 0.45          # m/s: straighten back toward the goal
PASS_SLOW_V = 0.25     # m/s: slower if something re-enters the corridor
PASS_K = 2.0           # heading P-gain in PASS
PASS_W_MAX = 1.0       # rad/s clamp

LIDAR_HOLD_V = 0.15    # m/s: safe slow creep while a LiDAR frame is missing

# --- M17: quintic leg planning inputs (per retarget) ---
# The QuinticPathPlanner's goalVelocity/initialVelocity defaults are 0.5
# m/s.  initialVelocity is the START tangent of the spline (v = sv along
# the robot's heading at retarget); assuming a 0.5 m/s push along the
# heading makes CARTER_3's delivery spline bulge 0.9 m east and then swing
# 0.66 m WEST of the delivery point (offline replica of the node: 0.246 m
# to rack S1's NE corner; measured run: 0.3919 m -> COLLISION).  Planning
# from a realistic (slow) start speed keeps the leg inside the 2.0 m aisle
# (replica: 0.710 m clearance) while Stanley's speed profile still ramps
# to the same 1.5 m/s cruise (it is normalised by the path's own peak).
LEG_START_SPEED = 0.2  # m/s: assumed speed at the START of a planned leg

MAX_STEPS = 4000       # 66 s hard cap at 60 Hz
FLEET_OBSERVE_STEPS = 300  # hold after ALL tasks complete (visible result)

# Safety thresholds (fail the test if violated)
MIN_FORWARD_ALLOWED = 1.0     # m: closest LiDAR reading ever
MIN_CLEARANCE_CENTER = 0.5    # m: robot center -> any static AABB
COLLISION_CLEARANCE = 0.45    # m: below this = static/traffic collision
ROBOT_COLLISION_GAP = 0.25    # m: robot-rect gap below this = collision
ROBOT_CLEARANCE_GAP = 0.5     # m: required min robot-rect gap (M16)

# --- M10: measured re-evaluation of MOVING obstacles (per step) ---
RETRIGGER_STEPS = 5      # sustained forward <= AVOID_T before re-committing
# While ALREADY passing (PASS_OBSTACLE) the robot is skimming the very
# obstacle it just swerved from, so its forward cone keeps re-reading that
# obstacle at ~1.5 m for a few steps.  Re-committing after 5 such steps
# produced an endless glide-along-the-rack spiral (CARTER_3 walked 2.5 m
# west under the rack row and wedged with forward nan).  A pass only
# re-commits after the low reading is SUSTAINED; PASS itself still
# steers back to the goal and still hands over to RESUME when clear.
# The route-cone gate added for M16 now blocks that spiral (a side rack
# wall with a CLEAR route can no longer re-trigger), so the streak only
# needs to be long enough to ride out a legitimately blocking obstacle.
# 120 steps (2 s) let CARTER_2 coast into the returning forklift for
# ~0.8 s before re-committing (forward 1.3558 -> 0.6670 < 1.0 m); 30
# steps (0.5 s) re-commits while still ~1.15 m clear of it.
PASS_RETRIGGER_STEPS = 30
TRACK_PRED_T = 2.0       # m: only cross-check boxes predicted this close
TRACK_TOL = 0.5          # m: LiDAR may lag the prediction by this much

# --- M11: RECOVERY + anti-oscillation + progress detection ---
BLOCKED_T = 1.2          # m: sector clearance treated as blocked (recovery)
# --- M15/M16: narrow-aisle rule (the rack gaps are only 2.0 m wide) ---
# Inside a rack gap BOTH sectors read <= BLOCKED_T (the rack walls we are
# already driving through), and the wide forward cone re-reads the rack's
# side face at ~1.0-1.5 m.  Committing a 55 deg turn + 0.75 m glide in
# there cannot fit: CARTER_1/CARTER_3 were driven into the rack faces
# (clearance 0.448, forward nan) and never completed their tasks.  When
# both sides are enclosed BUT still leave the robot its clearance, the
# route is passable -> hold the route (straight) instead of swerving.
AISLE_MAX_T = 1.6        # m: both sector clearances <= this = enclosed
AISLE_MIN_T = 0.6        # m: ... and >= this = the robot still fits (>0.45)
REAR_DEG_INNER = 150.0   # deg: rear cone (|az| >= 150) gates reversing
REAR_MIN_ALLOWED = 0.6   # m: rear clearance required to reverse
REVERSE_V = -0.15        # m/s: slow reverse during RECOVERY
REVERSE_MAX_STEPS = 300  # steps: short reverse phase (5 s)
RECOVERY_MAX_STEPS = 400 # steps: RECOVERY state hard timeout
RECOVERY_FORCE_EVENTS = 6  # attempts before a side is forced
REASSESS_STEPS = 20      # steps stopped + re-scanning before re-commit
FWD_STALL_STEPS = 180    # steps: sustained <= AVOID_T counts as stalled
NO_PROGRESS_STEPS = 600  # steps: sliding goal-distance window (10 s)
NO_PROGRESS_M = 0.15     # m: required goal progress over that window
AVOID_FAIL_STREAK = 3    # same obstacle avoided this many times -> recovery
DIR_LOCK_STEPS = 600     # hold the chosen side at least 10 s ...
DIR_LOCK_DIST_M = 3.0    # ... and at least 3 m before it may change
HYST_MARGIN = 0.5        # m: the other side must beat the held side by

MOTION_STATES = ("NAVIGATE", "APPROACH", "AVOID", "PASS_OBSTACLE", "RESUME")

TRAJ_STEP = 6              # ~10 Hz at 60 Hz physics
TRAJ_MAX_POINTS = 500      # bounded history per robot


# ============================================================
# STEP 11: CarterAgent - one instance per robot (M15 refactor of
# the proven single-robot controller: initialize/sense/navigate/
# task layer/records, no navigation rewrite)
# ============================================================
class CarterAgent:
    """One Carter AMR: own articulation, LiDAR, ray masks, OmniGraph
    control handoff, reactive state machine, task layer, trajectory
    and safety bookkeeping. All thresholds/helpers are shared module
    constants (the proven controller)."""

    def __init__(self, idx, tag, art, lidar, lidar_path, graph_path,
                 start_xy, rec, half):
        self.idx = idx
        self.tag = tag
        self.art = art
        self.lidar = lidar
        self.lidar_path = lidar_path
        self.graph_path = graph_path
        self.LIN_SRC = graph_path + "/StanleyControlPID.outputs:linearVelocity"
        self.LIN_DST = graph_path + "/DifferentialController.inputs:linearVelocity"
        self.ANG_SRC = graph_path + "/StanleyControlPID.outputs:angularVelocity"
        self.ANG_DST = graph_path + "/DifferentialController.inputs:angularVelocity"
        self.manual_mode = False
        self.init_ok = True

        # sensing
        self.lidar_initialized = False
        self.lidar_data_available = False
        self.lidar_rows = 0
        self.fwd_mask = None
        self.left_mask = None
        self.right_mask = None
        self.rear_mask = None
        self.az_deg = None          # wrapped 360-col azimuth grid (deg)
        self.route_ahead = None     # min range along the GOAL direction
        self.frame_keys_logged = False
        self.first_frame_analyzed = False
        self.forward_dist = None
        self.valid = None
        # Latest PhysX LiDAR range image (real frame only). Always
        # exists from __init__ on: sense() refreshes it every step,
        # _scan_array() hands it to navigate() and returns None while
        # no frame has been produced yet (missing data is NEVER read
        # as "no obstacle").
        self._d = None
        self.lidar_gap_logged = False

        # pose / goal
        self.start_pos = np.array([start_xy[0], start_xy[1], 0.1], dtype=np.float64)
        self.pos = self.start_pos.copy()
        self.prev_pos = self.start_pos.copy()
        self.yaw = 0.0
        self.speed = 0.0
        self.goal = None
        self.route_dir = np.array([1.0, 0.0], dtype=np.float64)
        self.dist_goal = float("inf")
        self.heading_err_deg = 0.0
        self.err_rad = 0.0
        self._turn_hold = False

        # safety
        self.min_forward = float("inf")
        # where the worst forward reading happened (diagnostic: identifies
        # the object/side of the pass that sets the SAFETY min-forward gate)
        self.min_fwd_step = None
        self.min_fwd_pose = None
        self.min_fwd_az = None
        self.min_clearance = float("inf")
        self.static_collision = False
        self.rr_min_gap = float("inf")
        self.rr_collision = False
        self.rec = rec                 # own live record (for other robots)
        self.half = np.asarray(half, dtype=np.float64)
        self.forward_entities = []     # traffic + OTHER robots (live dicts)

        # M16 robot-vs-robot interaction bookkeeping
        self.rr_attr = None            # other-robot rec explaining forward
        self.rr_pred = None            # its predicted AABB entry
        self.rr_detected = False
        self.rr_reacted = False
        self.rr_cleared = False
        self.rr_stop_steps = 0
        self.rr_yield_logged = False
        self.rr_stop_logged = False

        # per-entity detection flags (forklift / human)
        self.ent_flags = {
            o["tag"]: {"detected": False, "reacted": False, "cleared": False,
                       "dist": None}
            for o in traffic_entities
        }
        self.dynamic_hit_frames = 0
        self.tracking_mismatch = 0

        # reactive state machine (frozen M6-M11 variables)
        self.state = "NAVIGATE"
        self.last_state = None
        self.avoid_phase = None
        self.cur_obs = None
        self.avoid_entry = None
        self.open_dir = None
        self.dir_sign = 1.0
        self.pending_lines = []
        self.app_streak = 0
        self.resume_counter = 0
        self.avoid_steps = 0
        self.phase_steps = 0
        self.pass_steps = 0
        self.pass_streak = 0
        self.goal_reached = False
        self.goal_step = -1
        self.avoid_events = 0
        self.detected_set = set()
        self.detection_dist = {}
        self.events = []
        self.forward_low_streak = 0
        self.fwd_stall_steps = 0
        self.progress_window = []
        self.same_obs_streak = 0
        self.avoid_direction = ""
        self.last_dir = None
        self.steps_since_dir = 0
        self.path_since_dir = 0.0
        self.direction_changes = 0
        self.recovery_events = 0
        self.recovery_steps = 0
        self.reassess_steps = 0
        self.recovery_reason = ""
        self.dynamic_hit_frames = 0

        # M17 task layer
        self.task = None
        self.task_state = "IDLE"
        self.dwell = 0
        self.tasks_completed = 0

        # records / visualization
        self.position_history = []
        self.speeds = []
        self.traj_prim = None
        self.traj_points = []
        self.traj_cap_logged = False
        self.state_op = None
        self.marker_cube = None

    # ---------------------------------------------------------
    # Control authority: ABSOLUTE attribute paths (Milestone 6 fix).
    # NORMAL: Stanley -> DifferentialController connected.
    # AVOIDANCE/MANUAL: disconnected, we write velocities directly.
    # ---------------------------------------------------------
    def enter_manual(self):
        if not self.manual_mode:
            # Phase 3C: when /carterN/cmd_vel owns this graph the planner
            # edge was detached ONCE at handover, so there is nothing left
            # to disconnect - only the safety latch is taken.  With an
            # empty ROS2_HANDOVER_GRAPHS (PHASE3C_HANDOVER=0) this is the
            # original Phase-2 code path, byte for byte.
            if self.graph_path not in ROS2_HANDOVER_GRAPHS:
                og.Controller.edit(
                    self.graph_path,
                    {keys.DISCONNECT: [(self.LIN_SRC, self.LIN_DST),
                                       (self.ANG_SRC, self.ANG_DST)]},
                )
            self.manual_mode = True

    def exit_manual(self):
        if self.manual_mode:
            if self.graph_path in ROS2_HANDOVER_GRAPHS:
                # Phase 3B: release the safety latch but do NOT reconnect
                # Stanley - the handover wire stays in place.  Quintic +
                # Stanley keep running and stay reconnectable; they simply
                # no longer drive Carter 1's wheels.
                self.manual_mode = False
                return
            og.Controller.edit(
                self.graph_path,
                {keys.CONNECT: [(self.LIN_SRC, self.LIN_DST),
                                (self.ANG_SRC, self.ANG_DST)]},
            )
            self.manual_mode = False

    def write_cmd(self, v, w):
        og.Controller.attribute(self.LIN_DST).set(float(v))
        og.Controller.attribute(self.ANG_DST).set(float(w))

    def set_goal(self, xy, step, note):
        """M17 retarget: point the proven QuinticPathPlanner at a new
        waypoint (re-plans via targetChanged) and reset the reactive
        machine to NAVIGATE with the new route direction."""
        # --- plan-shaping for the new leg (offline replica of
        # OgnQuinticPathPlanner/quintic_polynomials_planner verified this) ---
        # The graph's build-time targetOrientation is the chassis yaw at
        # start-up, which becomes the END tangent of every spline.  For
        # CARTER_3's delivery leg (start yaw 39 deg, goal 6.4 m due north)
        # a NE end tangent forces the quintic to bulge ~0.9 m east and then
        # swing 0.66 m WEST of the delivery point -- straight across rack
        # S1's NE corner (replica: 0.246 m to the rack; measured run:
        # 0.3919 m -> CARTER_3_COLLISION).  Ending along the leg's own
        # approach bearing, and starting from a realistic slow speed,
        # keeps the whole leg inside the 2.0 m aisle (replica: 0.710 m)
        # without touching any safety threshold or behaviour rule.
        _dx = float(xy[0]) - float(self.pos[0])
        _dy = float(xy[1]) - float(self.pos[1])
        _approach = math.degrees(math.atan2(_dy, _dx))
        _half = math.radians(_approach) / 2.0
        # OmniGraph quatd attributes are (x, y, z, w) - same order the
        # node unpacks in quatd4_to_euler(), and the order of the
        # build-time chassis_quat this replaces.
        og.Controller.attribute(
            self.graph_path + "/QuinticPathPlanner.inputs:targetOrientation"
        ).set((0.0, 0.0, math.sin(_half), math.cos(_half)))
        og.Controller.attribute(
            self.graph_path + "/QuinticPathPlanner.inputs:initialVelocity"
        ).set(LEG_START_SPEED)
        og.Controller.attribute(
            self.graph_path + "/QuinticPathPlanner.inputs:targetPosition"
        ).set((float(xy[0]), float(xy[1]), 0.0))
        log_print(
            f"{self.tag}_LEG_PLAN: goal ({float(xy[0]):.3f},{float(xy[1]):.3f}) "
            f"approach {_approach:.1f} deg start_speed {LEG_START_SPEED:.2f} m/s "
            f"[{note}] step {step}"
        )
        old = None if self.goal is None else (float(self.goal[0]), float(self.goal[1]))
        self.goal = np.array([xy[0], xy[1], 0.0], dtype=np.float64)
        r = self.goal[:2] - self.pos[:2]
        n = float(np.linalg.norm(r))
        if n > 1e-6:
            self.route_dir = r / n
        # reset the navigation machine for the new leg
        self.exit_manual()
        self.state = "NAVIGATE"
        self.last_state = None
        self.avoid_phase = None
        self.cur_obs = None
        self.goal_reached = False
        self.goal_step = -1
        self.app_streak = 0
        self.resume_counter = 0
        self.pass_steps = 0
        self.pass_streak = 0
        self.avoid_steps = 0
        self.phase_steps = 0
        self.progress_window.clear()
        self.forward_low_streak = 0
        self.fwd_stall_steps = 0
        self.same_obs_streak = 0
        self.dwell = 0
        _old = "None" if old is None else f"({old[0]:.3f},{old[1]:.3f})"
        log_print(
            f"{self.tag}_RETARGET [{note}]: {_old} -> "
            f"({xy[0]:.3f},{xy[1]:.3f}) at step {step} "
            f"(task_state={self.task_state})"
        )

    # ---------------------------------------------------------
    # M15/M16 narrow-aisle test: is the robot inside an enclosed but
    # PASSABLE aisle (the 2.0 m rack gaps) with no other Carter close
    # ahead?  Then the low forward reading is the side face of the rack
    # wall we are already driving through, so the route must be held
    # straight instead of swerved (a swerve cannot fit in 2.0 m).
    # ---------------------------------------------------------
    def _aisle_passable(self, lc, rc):
        if not (
            AISLE_MIN_T <= lc <= AISLE_MAX_T
            and AISLE_MIN_T <= rc <= AISLE_MAX_T
        ):
            return False
        cand = entity_in_forward_cone(
            self.pos[:2], self.yaw, self.forward_entities
        )
        if (
            cand is not None
            and cand[1].get("kind") == "robot"
            and cand[0] <= CLEAR_T
        ):
            # another Carter is close ahead: keep the normal M16 rules
            # (attribution -> yield/stop, else swerve) untouched.
            return False
        return True

    # ---------------------------------------------------------
    # SENSE: pose, speed, LiDAR scan, forward/sector geometry,
    # entity attribution (traffic + other Carters), clearance.
    # ---------------------------------------------------------
    def sense(self, step):
        pos, rot = self.art.get_world_pose()
        self.pos = np.array(pos, dtype=np.float64)
        self.yaw = yaw_of(rot)
        self.speed = float(np.linalg.norm(self.pos - self.prev_pos)) / physics_dt
        self.prev_pos = self.pos.copy()
        self.speeds.append(self.speed)
        self.position_history.append(self.pos.copy())
        self.rec["center"][:] = self.pos[:2]
        self.rr_attr = None
        self.rr_pred = None

        frame = self.lidar.get_current_frame()
        d = to_numpy(frame.get("linear_depth")) if frame else None
        self.forward_dist = None
        self.valid = None
        # publish the real scan for this step (None until a frame exists)
        self._d = d if (d is not None and getattr(d, "size", 0) > 0) else None
        if d is not None and d.size > 0:
            self.lidar_data_available = True
            if not self.frame_keys_logged:
                self.frame_keys_logged = True
                log_print(f"{self.tag}_FRAME_KEYS: {list(frame.keys())}")
            if self.fwd_mask is None:
                self.fwd_mask, self.left_mask, self.right_mask, self.rear_mask, \
                    self.az_deg = build_masks(frame, d.shape)
                log_print(f"{self.tag}_SCAN_SHAPE: {d.shape}")
                try:
                    self.lidar_rows = int(self.lidar.get_num_rows())
                    _cols = int(self.lidar.get_num_cols())
                except Exception:
                    self.lidar_rows, _cols = int(d.shape[1]), int(d.shape[0])
                log_print(
                    f"{self.tag}_GET_NUM_ROWS: {self.lidar_rows} "
                    f"GET_NUM_COLS: {_cols}"
                )
                log_print(f"{self.tag}_FORWARD_RAYS: {int(np.count_nonzero(self.fwd_mask))}")
                log_print(f"{self.tag}_LEFT_SECTOR_RAYS: {int(np.count_nonzero(self.left_mask))}")
                log_print(f"{self.tag}_RIGHT_SECTOR_RAYS: {int(np.count_nonzero(self.right_mask))}")
                log_print(f"{self.tag}_REAR_SECTOR_RAYS: {int(np.count_nonzero(self.rear_mask))}")
            self.valid = (d > 0.0) & (d < LIDAR_MAX_RANGE)
            sel = self.fwd_mask & self.valid & (d > SELF_HIT_MAX)
            if np.any(sel):
                self.forward_dist = float(np.min(d[sel]))
                if self.forward_dist < self.min_forward:
                    self.min_forward = self.forward_dist
                    self.min_fwd_step = step
                    self.min_fwd_pose = (
                        float(self.pos[0]),
                        float(self.pos[1]),
                        float(self.yaw),
                    )
                    _az_i = int(np.argmin(np.where(sel, d, np.inf))) // d.shape[1]
                    self.min_fwd_az = float(self.az_deg[_az_i])
                if not self.first_frame_analyzed:
                    self.first_frame_analyzed = True
                    log_print(
                        f"{self.tag}_VALID_RAYS: {int(np.count_nonzero(self.valid))}"
                    )
                    log_print(
                        f"{self.tag}_FORWARD_MIN_DISTANCE_FIRST_FRAME: "
                        f"{self.forward_dist:.4f}"
                    )

            if self.forward_dist is not None:
                # cross-check the live scan vs the scripted traffic motion
                _cf = np.array(
                    [math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))],
                    dtype=np.float64,
                )
                _lidar_xy = self.pos[:2] - 0.06 * _cf
                for _o in traffic_entities:
                    _pred = ray_aabb_forward(_lidar_xy, _cf, _o["center"], _o["half"])
                    if _pred is None:
                        continue
                    if _pred <= TRACK_PRED_T and self.forward_dist > _pred + TRACK_TOL:
                        self.tracking_mismatch += 1
                        log_print(
                            f"{self.tag}_TRAFFIC_TRACKING_MISMATCH: {_o['tag']} "
                            f"pred {_pred:.3f} m but forward "
                            f"{self.forward_dist:.3f} m at step {step}"
                        )
                        break

                if self.forward_dist <= CLEAR_T:
                    # cone-aware attribution: the closest entity whose
                    # AABB entry explains the measured forward reading
                    cand = attribute_forward_entity(
                        self.pos[:2], self.yaw, self.forward_dist,
                        self.forward_entities,
                    )
                    if cand is not None:
                        pred, ent = cand
                        if ent["kind"] == "robot":
                            # M16: another Carter is what LiDAR sees
                            self.rr_attr = ent
                            self.rr_pred = pred
                        else:
                            self.dynamic_hit_frames += 1
                            fl = self.ent_flags[ent["tag"]]
                            fl["detected"] = True
                            if fl["dist"] is None:
                                fl["dist"] = self.forward_dist
                            if self.state in ("APPROACH", "AVOID"):
                                fl["reacted"] = True

        # --- geometric clearance vs static colliders + LIVE traffic ---
        # (robot-vs-robot uses the rectangle gap check instead)
        step_min = float("inf")
        for _name, _c, _h in collision_boxes:
            dc = aabb_distance(self.pos[:2], _c, _h)
            if dc < step_min:
                step_min = dc
        for _o in traffic_entities:
            dc = aabb_distance(self.pos[:2], _o["center"], _o["half"])
            if dc < step_min:
                step_min = dc
        if step_min < self.min_clearance:
            self.min_clearance = step_min
        if not self.static_collision and step_min < COLLISION_CLEARANCE:
            self.static_collision = True
            log_print(
                f"{self.tag}_COLLISION_WARNING_STEP: {step} "
                f"MIN_CLEARANCE: {step_min:.4f}"
            )

        # --- goal / heading geometry ---
        self.dist_goal = float(
            math.hypot(self.goal[0] - self.pos[0], self.goal[1] - self.pos[1])
        )
        goal_bearing = math.degrees(
            math.atan2(self.goal[1] - self.pos[1], self.goal[0] - self.pos[0])
        )
        self.heading_err_deg = float(wrap_deg(goal_bearing - self.yaw))
        self.err_rad = math.radians(self.heading_err_deg)

        # --- ROUTE cone: min range along the direction to the GOAL ---
        # The forward cone follows the (possibly rotated) heading, so inside
        # a 2.0 m rack gap its side glances keep re-reading the rack wall we
        # are already driving past.  The route cone answers the question the
        # AVOID decision really needs: IS THE WAY TO THE GOAL BLOCKED?
        # (None = nothing in range = clear; 0.0 = something inside the
        # self-hit mask sits right on the route = blocked at point blank.)
        self.route_ahead = None
        if d is not None and self.valid is not None and self.az_deg is not None:
            _rz = wrap_deg(goal_bearing - self.yaw)
            _elev_ok = np.any(self.fwd_mask, axis=0)
            _rsel = (
                (np.abs(wrap_deg(self.az_deg - _rz)) <= ROUTE_CONE_DEG)[:, None]
                & _elev_ok[None, :]
                & self.valid
            )
            if np.any(_rsel):
                _hit = _rsel & (d > SELF_HIT_MAX)
                self.route_ahead = float(np.min(d[_hit])) if np.any(_hit) else 0.0

    # ---------------------------------------------------------
    # M17 task layer (runs only while the robot sits at its
    # current waypoint in GOAL_REACHED: dwell -> retarget -> done)
    # ---------------------------------------------------------
    def task_layer(self, step):
        if self.task is None:
            return
        if self.task_state == "GO_TO_PICKUP":
            self.task_state = "PICKUP"
            self.dwell = max(1, int(PICKUP_DWELL_S * physics_rate))
            log_print(
                f"{self.tag}_PICKUP: {self.task['id']} pickup reached "
                f"(step {step}, dwell {PICKUP_DWELL_S:.1f}s, "
                f"POSITION: ({self.pos[0]:.3f}, {self.pos[1]:.3f}))"
            )
        elif self.task_state == "PICKUP":
            self.dwell -= 1
            if self.dwell <= 0:
                self.set_goal(self.task["delivery"], step, f"{self.task['id']}:GO_TO_DELIVERY")
                self.task_state = "GO_TO_DELIVERY"
        elif self.task_state == "GO_TO_DELIVERY":
            self.task_state = "DELIVERY"
            self.dwell = max(1, int(DELIVERY_DWELL_S * physics_rate))
            log_print(
                f"{self.tag}_DELIVERY: {self.task['id']} delivery reached "
                f"(step {step}, dwell {DELIVERY_DWELL_S:.1f}s, "
                f"POSITION: ({self.pos[0]:.3f}, {self.pos[1]:.3f}))"
            )
        elif self.task_state == "DELIVERY":
            self.dwell -= 1
            if self.dwell <= 0:
                self.task["status"] = "COMPLETED"
                self.task["completed_step"] = step
                self.task_state = "TASK_COMPLETE"
                self.tasks_completed += 1
                log_print(
                    f"{self.task['id']}_COMPLETED_EVENT: YES ({self.tag}, "
                    f"step {step}, POSITION: ({self.pos[0]:.3f}, "
                    f"{self.pos[1]:.3f}))"
                )
                self.task_state = "IDLE"
                log_print(f"{self.tag}_TASK_STATE: IDLE (task complete)")
                self.task = None

    # ---------------------------------------------------------
    # NAVIGATE: the proven reactive state machine + control output
    # + the M16 robot-vs-robot safety zones (measured clearance).
    # ---------------------------------------------------------
    def navigate(self, step):
        d = self._scan_array()
        valid = self.valid
        forward_dist = self.forward_dist
        state = self.state  # local alias; written back at the end

        # No usable LiDAR frame for this step: never treat missing data
        # as "no obstacle" - hold a safe slow command and re-evaluate on
        # the next step once sense() has a real scan.
        if d is None:
            if not self.lidar_gap_logged:
                self.lidar_gap_logged = True
                log_print(
                    f"{self.tag}_LIDAR_FRAME_UNAVAILABLE: holding safe slow "
                    f"command (step {step})"
                )
            self.enter_manual()
            self.write_cmd(LIDAR_HOLD_V, 0.0)
            self.state = state
            return

        # ------------------------------------------------------------
        # GOAL (highest priority): stop safely, then run the task layer
        # ------------------------------------------------------------
        if state != "GOAL_REACHED" and self.dist_goal < GOAL_TOL:
            self.enter_manual()
            self.write_cmd(0.0, 0.0)
            state = "GOAL_REACHED"
            self.goal_reached = True
            self.goal_step = step
            log_print(
                f"{self.tag}_GOAL_REACHED_STEP: {step}, DIST_TO_GOAL: "
                f"{self.dist_goal:.4f}, POSITION: ({self.pos[0]:.3f}, "
                f"{self.pos[1]:.3f}), SPEED: {self.speed:.3f}"
            )

        if state == "GOAL_REACHED":
            for _o in traffic_entities:
                if self.ent_flags[_o["tag"]]["reacted"]:
                    self.ent_flags[_o["tag"]]["cleared"] = True
            self.write_cmd(0.0, 0.0)
            self.state = state
            self.task_layer(step)
            self._log_state_change(step, forward_dist)
            self._rr_bookkeeping(step)
            return

        # ------------------------------------------------------------
        # M10/M11 counters: sustained-block streaks, sliding progress
        # window, direction-lock timers (updated every non-goal step).
        # M16: an intentional YIELD stop for another Carter is NOT a
        # stall (the stall counters ignore it; the 10 s no-progress
        # window stays armed as a deadlock escape).
        # ------------------------------------------------------------
        do_avoid = False
        avoid_dir = None
        avoid_lc = 0.0
        avoid_rc = 0.0
        motion_state = state in MOTION_STATES
        rr_yield = self.rr_attr is not None
        if (
            motion_state
            and forward_dist is not None
            and forward_dist <= AVOID_T
            and not rr_yield
        ):
            self.forward_low_streak += 1
            self.fwd_stall_steps += 1
        else:
            self.forward_low_streak = 0
            self.fwd_stall_steps = 0
        if state in ("NAVIGATE", "APPROACH", "RESUME", "PASS_OBSTACLE", "AVOID"):
            self.progress_window.append((step, self.dist_goal))
        while self.progress_window and self.progress_window[0][0] < step - NO_PROGRESS_STEPS:
            self.progress_window.pop(0)
        self.steps_since_dir += 1
        self.path_since_dir += self.speed * physics_dt

        # --- M11 recovery triggers #2/#3: forward stalled or no progress
        if motion_state and self.progress_window:
            _win_span = self.progress_window[-1][0] - self.progress_window[0][0]
            _prog = self.progress_window[0][1] - self.dist_goal
            _stalled = self.fwd_stall_steps >= FWD_STALL_STEPS and _prog < NO_PROGRESS_M
            _stuck = _win_span >= NO_PROGRESS_STEPS and _prog < NO_PROGRESS_M
            if _stalled or _stuck:
                self.recovery_reason = "FORWARD_STALL" if _stalled else "NO_PROGRESS"
                self.pending_lines = [
                    f"{self.tag}_RECOVERY_REASON: {self.recovery_reason} (step {step})",
                    f"{self.tag}_RECOVERY_PROGRESS: "
                    f"{self.progress_window[0][1]:.3f} -> {self.dist_goal:.3f} m "
                    f"over {_win_span} steps",
                ]
                self.enter_manual()
                self.recovery_events += 1
                self.recovery_steps = 0
                self.progress_window.clear()
                self.fwd_stall_steps = 0
                self.forward_low_streak = 0
                state = "RECOVERY"

        # ------------------------------------------------------------
        # M11 RECOVERY: stop -> reverse slowly/short (only while the
        # rear sector is clear) -> REASSESS (stopped re-scan) ->
        # compare sectors -> pick the safer side -> resume
        # ------------------------------------------------------------
        if state == "RECOVERY":
            self.recovery_steps += 1
            _rear = (
                sector_min(d, self.rear_mask, valid)
                if d is not None and self.rear_mask is not None and valid is not None
                else 0.0
            )
            _may_reverse = (
                _rear >= REAR_MIN_ALLOWED
                and self.recovery_steps < REVERSE_MAX_STEPS
                and forward_dist is not None
                and forward_dist < CLEAR_T
            )
            if not _may_reverse:
                state = "REASSESS"
                self.reassess_steps = 0
                self.pending_lines = [
                    f"{self.tag}_RECOVERY_PHASE: REASSESS (step {step}, "
                    f"rear {_rear:.3f} m, reverse steps {self.recovery_steps})"
                ]
            elif self.recovery_steps >= RECOVERY_MAX_STEPS:
                state = "REASSESS"
                self.reassess_steps = 0
                self.pending_lines = [f"{self.tag}_RECOVERY_TIMEOUT -> REASSESS (step {step})"]

        if state == "REASSESS":
            self.reassess_steps += 1
            if (
                self.reassess_steps >= REASSESS_STEPS
                and d is not None
                and valid is not None
            ):
                direction, lc, rc, lock_note = decide_avoid_side(
                    d, valid, self.left_mask, self.right_mask,
                    self.last_dir, self.steps_since_dir, self.path_since_dir,
                )
                blocked = lc <= BLOCKED_T and rc <= BLOCKED_T
                # a clear ROUTE to the goal also means "no need to swerve":
                # the yaw cone may still be grazing the rack wall beside us
                route_clear = (
                    self.route_ahead is None or self.route_ahead > AVOID_T
                )
                fwd_clear = (
                    forward_dist is None
                    or forward_dist > AVOID_T
                    or route_clear
                )
                if self._aisle_passable(lc, rc):
                    # enclosed but passable rack gap: the low forward
                    # reading is the rack wall beside us.  Forcing an
                    # AVOID here turned CARTER_1 into rack B (clearance
                    # 0.448) - hold the route instead.
                    self.exit_manual()
                    state = "NAVIGATE"
                    self.recovery_steps = 0
                    self.pending_lines = [
                        f"{self.tag}_AISLE_RESUME: LEFT {lc:.4f} RIGHT "
                        f"{rc:.4f} enclosed passable aisle - hold route "
                        f"(step {step})"
                    ]
                elif (
                    fwd_clear
                    and not blocked
                    and self.recovery_events < RECOVERY_FORCE_EVENTS
                ):
                    # forward_dist may be None here (fwd_clear is True when
                    # the forward cone has no usable return); formatting it
                    # with :.3f raised "unsupported format string passed to
                    # NoneType.__format__" and killed the run.
                    _frd = (
                        f"{forward_dist:.3f}"
                        if forward_dist is not None
                        else "nan"
                    )
                    _rt = (
                        f"{self.route_ahead:.3f} m"
                        if self.route_ahead is not None
                        else "no-return"
                    )
                    self.pending_lines = [
                        f"{self.tag}_RECOVERY_RESOLVED: forward "
                        f"{_frd} m clear (route {_rt}, step {step})"
                    ]
                    self.exit_manual()
                    state = "NAVIGATE"
                    self.recovery_steps = 0
                elif blocked and self.recovery_events < RECOVERY_FORCE_EVENTS:
                    self.recovery_events += 1
                    self.recovery_steps = 0
                    state = "RECOVERY"
                    self.pending_lines = [
                        f"{self.tag}_REASSESS: BOTH_SIDES_BLOCKED -> RECOVERY "
                        f"(attempt {self.recovery_events}, step {step})",
                        f"{self.tag}_RECOVERY_SECTORS: LEFT {lc:.4f} RIGHT {rc:.4f}",
                    ]
                else:
                    if blocked:
                        direction = "LEFT" if lc >= rc else "RIGHT"
                        lock_note = "RECOVERY_FORCED_DIRECTION"
                    self.cur_obs = resolve_forward_entity(
                        self.pos[:2], self.yaw, forward_dist,
                        self.forward_entities,
                    )
                    avoid_dir = direction
                    avoid_lc = lc
                    avoid_rc = rc
                    do_avoid = True
                    self.pending_lines = [
                        f"{self.tag}_RECOVERY_DECISION: AVOID_{direction} (step {step})",
                        f"{self.tag}_LEFT_CLEARANCE: {lc:.4f}",
                        f"{self.tag}_RIGHT_CLEARANCE: {rc:.4f}",
                        f"{self.tag}_AVOIDANCE_DIRECTION: {direction}",
                        f"{self.tag}_DIRECTION_LOCK: {lock_note}",
                    ]

        # ------------------------------------------------------------
        # STATE TRANSITIONS
        # ------------------------------------------------------------
        # NAVIGATE -> APPROACH (something entering the 2.5 m band)
        if state == "NAVIGATE":
            self.same_obs_streak = 0
            if forward_dist is not None and forward_dist <= CLEAR_T:
                self.enter_manual()
                state = "APPROACH"
                self.app_streak = 0

        # RESUME -> APPROACH/NAVIGATE (after a fixed Stanley hold)
        if state == "RESUME":
            self.resume_counter += 1
            if self.resume_counter >= RESUME_STEPS:
                if forward_dist is not None and forward_dist <= CLEAR_T:
                    self.enter_manual()
                    state = "APPROACH"
                    self.app_streak = 0
                else:
                    state = "NAVIGATE"

        # APPROACH -> (AVOID via the unified trigger below) or NAVIGATE
        if state == "APPROACH":
            if forward_dist is None or forward_dist > CLEAR_T:
                if forward_dist is not None and forward_dist > CLEAR_T:
                    self.app_streak += 1
                else:
                    self.app_streak = 0
                if self.app_streak >= NAV_RECOVER_STEPS:
                    self.exit_manual()
                    state = "NAVIGATE"

        # ------------------------------------------------------------
        # M10: unified AVOID trigger with ONE measured decision path.
        # M16: when ANOTHER CARTER explains the forward reading the
        # trigger is suppressed - the robot YIELDS (slow/stop from the
        # measured clearance) instead of swerving into the crossing.
        # ------------------------------------------------------------
        trig_avoid = False
        # Is the way to the GOAL blocked?  (route-cone reading, sense())
        # Without this the yaw-cone alone re-triggers AVOID on the rack
        # side wall we are already driving past in a 2.0 m gap - that is
        # what turned CARTER_3 into rack B (clearance 0.1266) at step 1856
        # while its route was in fact 4.37 m clear.
        route_blocked = (
            self.route_ahead is not None and self.route_ahead <= AVOID_T
        )
        if (
            state == "APPROACH"
            and forward_dist is not None
            and forward_dist <= AVOID_T
            and route_blocked
        ):
            trig_avoid = True
        elif (
            state in ("NAVIGATE", "RESUME", "PASS_OBSTACLE")
            and self.forward_low_streak
            >= (
                PASS_RETRIGGER_STEPS
                if state == "PASS_OBSTACLE"
                else RETRIGGER_STEPS
            )
            and forward_dist is not None
            and forward_dist <= AVOID_T
            and route_blocked
        ):
            trig_avoid = True

        if trig_avoid and self.rr_attr is not None:
            trig_avoid = False
            if not self.rr_yield_logged:
                self.rr_yield_logged = True
                log_print(
                    f"{self.tag}_RR_YIELD: forward {forward_dist:.3f} m explained "
                    f"by {self.rr_attr['tag']} - slow/stop instead of swerve "
                    f"(step {step})"
                )

        if trig_avoid and d is not None and valid is not None:
            # ---- AVOIDANCE DECISION FROM LIVE LIDAR CLEARANCE ----
            hold_route = False
            avoid_dir, avoid_lc, avoid_rc, lock_note = decide_avoid_side(
                d, valid, self.left_mask, self.right_mask,
                self.last_dir, self.steps_since_dir, self.path_since_dir,
            )
            self.cur_obs = resolve_forward_entity(
                self.pos[:2], self.yaw, forward_dist, self.forward_entities
            )
            cur_idx = self.cur_obs["idx"]
            if self.events and self.events[-1]["obstacle"] == cur_idx + 1:
                self.same_obs_streak += 1
            else:
                self.same_obs_streak = 1

            if self._aisle_passable(avoid_lc, avoid_rc):
                # enclosed but passable rack gap (2.0 m): hold the route
                # straight - neither swerve nor recovery fits in here.
                hold_route = True
                trig_avoid = False
                self.forward_low_streak = 0
                self.fwd_stall_steps = 0
                # a held route is NOT a failed avoid attempt - do not let
                # these accumulate into REPEATED_AVOIDANCE_FAILURE
                self.same_obs_streak = 0
                self.pending_lines = [
                    f"{self.tag}_AISLE_HOLD: LEFT {avoid_lc:.4f} RIGHT "
                    f"{avoid_rc:.4f} enclosed passable aisle - hold route "
                    f"instead of swerving (step {step})"
                ]
            elif avoid_lc <= BLOCKED_T and avoid_rc <= BLOCKED_T:
                self.recovery_reason = "BOTH_SIDES_BLOCKED"
                self.pending_lines = [
                    f"{self.tag}_RECOVERY_REASON: {self.recovery_reason} (step {step})",
                    f"{self.tag}_RECOVERY_SECTORS: LEFT {avoid_lc:.4f} RIGHT {avoid_rc:.4f}",
                ]
            elif self.same_obs_streak >= AVOID_FAIL_STREAK:
                self.recovery_reason = "REPEATED_AVOIDANCE_FAILURE"
                self.pending_lines = [
                    f"{self.tag}_RECOVERY_REASON: {self.recovery_reason} "
                    f"(obstacle {cur_idx + 1}, {self.same_obs_streak} attempts, "
                    f"step {step})",
                    f"{self.tag}_RECOVERY_SECTORS: LEFT {avoid_lc:.4f} RIGHT {avoid_rc:.4f}",
                ]
            else:
                do_avoid = True
                self.pending_lines = [
                    f"{self.tag}_LEFT_CLEARANCE: {avoid_lc:.4f}",
                    f"{self.tag}_RIGHT_CLEARANCE: {avoid_rc:.4f}",
                    f"{self.tag}_AVOIDANCE_DIRECTION: {avoid_dir}",
                    f"{self.tag}_DIRECTION_LOCK: {lock_note}",
                ]

            if not do_avoid and not hold_route:
                self.enter_manual()
                self.recovery_events += 1
                self.recovery_steps = 0
                self.progress_window.clear()
                self.fwd_stall_steps = 0
                self.forward_low_streak = 0
                state = "RECOVERY"
        elif trig_avoid:
            trig_avoid = False

        # AVOID sub-phases: TURN (rotate away) -> GLIDE (lateral gain)
        if state == "AVOID":
            self.avoid_steps += 1
            self.phase_steps += 1
            if self.avoid_phase == "TURN":
                if abs(self.heading_err_deg) >= TURN_DEV_DEG:
                    self.avoid_phase = "GLIDE"
                    self.phase_steps = 0
                    log_print(
                        f"{self.tag}_AVOID_PHASE: GLIDE (heading dev "
                        f"{abs(self.heading_err_deg):.1f} deg, step {step})"
                    )
                elif self.phase_steps > TURN_TIMEOUT:
                    self.avoid_phase = "GLIDE"
                    self.phase_steps = 0
                    log_print(f"{self.tag}_AVOID_TURN_TIMEOUT -> GLIDE (step {step})")
            if self.avoid_phase == "GLIDE":
                lateral = float(np.dot(self.pos[:2] - self.avoid_entry, self.open_dir))
                if lateral >= GLIDE_LATERAL_M:
                    state = "PASS_OBSTACLE"
                    self.pass_steps = 0
                    self.pass_streak = 0
                    log_print(
                        f"{self.tag}_AVOID_GLIDE_DONE: lateral {lateral:.3f} m "
                        f"(step {step})"
                    )
                elif self.phase_steps > GLIDE_TIMEOUT:
                    state = "PASS_OBSTACLE"
                    self.pass_steps = 0
                    self.pass_streak = 0
                    log_print(
                        f"{self.tag}_AVOID_GLIDE_TIMEOUT -> PASS_OBSTACLE (step {step})"
                    )

        # PASS_OBSTACLE -> RESUME (obstacle behind + corridor clear)
        if state == "PASS_OBSTACLE":
            self.pass_steps += 1
            if self.cur_obs is not None:
                past = (
                    float(
                        np.dot(
                            self.pos[:2] - self.cur_obs["center"], self.route_dir
                        )
                    )
                    > PASS_PAST_M
                )
            else:
                past = True
            heading_ok = abs(self.heading_err_deg) <= PASS_HEADING_TOL
            clear_ok = forward_dist is not None and forward_dist > CLEAR_T
            if past and heading_ok and clear_ok:
                self.pass_streak += 1
            else:
                self.pass_streak = 0
            if self.pass_streak >= PASS_CLEAR_STEPS or self.pass_steps > PASS_TIMEOUT_STEPS:
                if self.pass_steps > PASS_TIMEOUT_STEPS and self.pass_streak < PASS_CLEAR_STEPS:
                    log_print(f"{self.tag}_PASS_TIMEOUT -> RESUME (step {step})")
                self.exit_manual()
                state = "RESUME"
                self.resume_counter = 0

        # ------------------------------------------------------------
        # Single AVOID entry point
        # ------------------------------------------------------------
        if do_avoid:
            if self.last_dir is not None and avoid_dir != self.last_dir:
                self.direction_changes += 1
                log_print(
                    f"{self.tag}_DIRECTION_CHANGE: {self.last_dir} -> {avoid_dir} "
                    f"(event {self.avoid_events + 1}, step {step})"
                )
            self.last_dir = avoid_dir
            self.avoid_direction = avoid_dir
            self.steps_since_dir = 0
            self.path_since_dir = 0.0
            self.avoid_events += 1
            cur_idx = self.cur_obs["idx"]
            self.detected_set.add(cur_idx)
            self.detection_dist.setdefault(cur_idx, forward_dist)
            self.cur_obs["detected"] = True
            self.cur_obs["reacted"] = True
            if self.cur_obs.get("kind") in ("forklift", "human"):
                self.ent_flags[self.cur_obs["tag"]]["detected"] = True
                self.ent_flags[self.cur_obs["tag"]]["reacted"] = True
            self.events.append(
                {
                    "step": step,
                    "obstacle": cur_idx + 1,
                    "tag": self.cur_obs["tag"],
                    "priority": priority_label(self.cur_obs),
                    "forward": forward_dist,
                    "left": avoid_lc,
                    "right": avoid_rc,
                    "direction": avoid_dir,
                }
            )
            self.enter_manual()
            self.avoid_entry = self.pos[:2].copy()
            left_w = np.array(
                [-math.sin(math.radians(self.yaw)), math.cos(math.radians(self.yaw))],
                dtype=np.float64,
            )
            self.dir_sign = 1.0 if avoid_dir == "LEFT" else -1.0
            self.open_dir = left_w * self.dir_sign
            self.avoid_phase = "TURN"
            self.avoid_steps = 0
            self.phase_steps = 0
            state = "AVOID"
            self.forward_low_streak = 0
            self.fwd_stall_steps = 0

        # ------------------------------------------------------------
        # CONTROL OUTPUT: exactly one authority per state
        # ------------------------------------------------------------
        if state == "APPROACH":
            # ROUTE-IN-CONE GATE (Milestone 16 min-forward fix).
            # Inside the emergency band the forward LiDAR is measuring the
            # very object we are closing on, but that cone only spans
            # +-FORWARD_CONE_DEG around the NOSE.  While the goal bearing
            # lies outside it, advancing at APPROACH_V drives the robot
            # toward something the cone cannot see along the direction of
            # travel.  Measured: CARTER_1 crept ~0.23 m east past the
            # forklift at 45-58 deg off-route and its min forward fell
            # 1.3792 -> 0.9974 m, under MIN_FORWARD_ALLOWED=1.0.  Holding
            # the advance and rotating in place first keeps every ray of
            # that cone on the route; the turn itself is unchanged
            # (APPROACH_W_MAX still applies), so err -> 0 and it releases
            # by the same rule - no threshold is touched.
            hold_to_turn = (
                forward_dist is not None
                and forward_dist <= EMERG_T
                and abs(self.heading_err_deg) > FORWARD_CONE_DEG
            )
            if hold_to_turn and not self._turn_hold:
                log_print(
                    f"{self.tag}_TURN_TO_ROUTE: holding advance "
                    f"(forward {forward_dist:.4f} <= {EMERG_T} m and route "
                    f"bearing {self.heading_err_deg:+.1f} deg outside the "
                    f"+-{FORWARD_CONE_DEG:.0f} deg cone, step {step})"
                )
            self._turn_hold = hold_to_turn
            self.write_cmd(
                0.0 if hold_to_turn else APPROACH_V,
                float(np.clip(APPROACH_K * self.err_rad, -APPROACH_W_MAX, APPROACH_W_MAX)),
            )
        elif state == "AVOID":
            if self.avoid_phase == "TURN":
                self.write_cmd(TURN_V, self.dir_sign * TURN_W)
            else:
                self.write_cmd(GLIDE_V, 0.0)
        elif state == "PASS_OBSTACLE":
            if forward_dist is not None and forward_dist <= CLEAR_T:
                self.write_cmd(
                    PASS_SLOW_V,
                    float(np.clip(PASS_K * self.err_rad, -PASS_W_MAX, PASS_W_MAX)),
                )
            else:
                self.write_cmd(
                    PASS_V,
                    float(np.clip(PASS_K * self.err_rad, -PASS_W_MAX, PASS_W_MAX)),
                )
        elif state == "RECOVERY":
            self.write_cmd(REVERSE_V, 0.0)
        elif state == "REASSESS":
            self.write_cmd(0.0, 0.0)
        # NAVIGATE / RESUME: Stanley pipeline owns the command

        # ------------------------------------------------------------
        # M16 robot-to-robot SAFETY ZONES (measured LiDAR clearance):
        #   forward <= EMERG_T with another Carter attributed -> STOP
        #   (emergency zone). Resume happens through the proven
        #   APPROACH -> NAVIGATE path once the cone reads > CLEAR_T.
        #   NEVER hard-coded: it depends on which robot measures which.
        # ------------------------------------------------------------
        if (
            self.rr_attr is not None
            and forward_dist is not None
            and forward_dist <= EMERG_T
            and state in MOTION_STATES
        ):
            self.enter_manual()
            self.write_cmd(0.0, 0.0)
            self.rr_stop_steps += 1
            if not self.rr_stop_logged:
                self.rr_stop_logged = True
                log_print(
                    f"{self.tag}_RR_EMERGENCY_STOP: forward {forward_dist:.3f} m "
                    f"<= {EMERG_T} m with {self.rr_attr['tag']} in the cone "
                    f"(step {step})"
                )

        # --- an entity is "cleared" once back to normal navigation ---
        for _o in traffic_entities:
            fl = self.ent_flags[_o["tag"]]
            if fl["reacted"] and state in ("RESUME", "NAVIGATE", "GOAL_REACHED"):
                fl["cleared"] = True

        self.state = state
        self._log_state_change(step, forward_dist)
        self._rr_bookkeeping(step)

    def _scan_array(self):
        """Real PhysX range image captured by sense() this step, or
        None when no frame is available yet (never fabricated)."""
        d = self._d
        if d is None or getattr(d, "size", 0) == 0:
            return None
        return d

    def _log_state_change(self, step, forward_dist):
        if self.state != self.last_state:
            for _ln in self.pending_lines:
                log_print(_ln)
            self.pending_lines = []
            fd_str = f"{forward_dist:.4f}" if forward_dist is not None else "nan"
            log_print(
                f"{self.tag}_STATE: {self.state} (step {step}, "
                f"FORWARD_DISTANCE: {fd_str})"
            )
            if self.state == "AVOID" and self.events:
                # A RECOVERY-forced avoid has no forward lidar reading
                # (forward_dist is None -> logged as "nan" above); formatting
                # it with :.4f raised "unsupported format string passed to
                # NoneType.__format__" and killed the run.
                ev = self.events[-1]

                def _f(v):
                    return f"{v:.4f}" if isinstance(v, (int, float)) else "nan"

                log_print(
                    f"{self.tag}_AVOID_EVENT {ev['step']}: STEP {ev['step']}, "
                    f"ENTITY {ev['tag']}, PRIORITY {ev['priority']}, "
                    f"DETECTION_DISTANCE: {_f(ev['forward'])}, "
                    f"LEFT_CLEARANCE: {_f(ev['left'])}, "
                    f"RIGHT_CLEARANCE: {_f(ev['right'])}, "
                    f"AVOIDANCE_DIRECTION: {ev['direction']}"
                )
            if self.state == "AVOID" and self.avoid_direction:
                _marker_key = f"AVOID_{self.avoid_direction}"
            else:
                _marker_key = self.state
            if self.marker_cube is not None:
                _col = STATE_COLORS.get(
                    _marker_key, STATE_COLORS.get(self.state, (1.0, 1.0, 1.0))
                )
                if self.task_state in ("PICKUP", "DELIVERY", "TASK_COMPLETE", "IDLE"):
                    _col = TASK_PHASE_COLORS.get(self.task_state, _col)
                self.marker_cube.GetDisplayColorAttr().Set([Gf.Vec3f(*_col)])
            self.last_state = self.state

    # ---------------------------------------------------------
    # M16 bookkeeping: DETECT / REACT / CLEAR, purely from the
    # measured forward-cone attribution (see sense()).
    # ---------------------------------------------------------
    def _rr_bookkeeping(self, step):
        if self.rr_attr is not None:
            if not self.rr_detected:
                self.rr_detected = True
                log_print(
                    f"MULTI_ROBOT_EVENT: DETECT {self.tag} -> "
                    f"{self.rr_attr['tag']} (step {step}, FORWARD: "
                    f"{self.forward_dist:.4f}, PRED: {self.rr_pred:.4f})"
                )
            if self.state in ("APPROACH", "AVOID"):
                if not self.rr_reacted:
                    log_print(
                        f"MULTI_ROBOT_EVENT: REACT {self.tag} {self.state} to "
                        f"{self.rr_attr['tag']} (step {step}, FORWARD: "
                        f"{self.forward_dist:.4f})"
                    )
                self.rr_reacted = True
        if (
            self.rr_reacted
            and not self.rr_cleared
            and self.state in ("NAVIGATE", "RESUME", "GOAL_REACHED")
        ):
            self.rr_cleared = True
            log_print(
                f"MULTI_ROBOT_EVENT: CLEAR {self.tag} resumed {self.state} "
                f"after yielding (step {step})"
            )

    # ---------------------------------------------------------
    # Visualization: floating state marker + trajectory trail +
    # periodic progress log (lightweight, per robot).
    # ---------------------------------------------------------
    def viz(self, step):
        if self.state_op is not None:
            self.state_op.Set(
                Gf.Vec3f(
                    float(self.pos[0]), float(self.pos[1]), float(self.pos[2]) + 1.6
                )
            )
        if step % TRAJ_STEP == 0 and self.traj_prim is not None:
            if len(self.traj_points) < TRAJ_MAX_POINTS:
                self.traj_points.append(
                    Gf.Vec3f(
                        float(self.pos[0]), float(self.pos[1]),
                        float(self.pos[2]) + 0.05,
                    )
                )
                self.traj_prim.GetPointsAttr().Set(self.traj_points)
                self.traj_prim.GetWidthsAttr().Set([0.07] * len(self.traj_points))
            elif not self.traj_cap_logged:
                self.traj_cap_logged = True
                log_print(f"{self.tag}_TRAJECTORY_POINT_CAP_REACHED: {TRAJ_MAX_POINTS}")
        if step % 50 == 0:
            fd_str = (
                f"{self.forward_dist:.4f}" if self.forward_dist is not None else "nan"
            )
            ra_str = (
                f"{self.route_ahead:.4f}"
                if self.route_ahead is not None
                else "nan"
            )
            log_print(
                f"{self.tag}_STEP {step}: POSITION: ({self.pos[0]:.3f}, "
                f"{self.pos[1]:.3f}), SPEED: {self.speed:.3f}, "
                f"YAW: {self.yaw:.1f}, FORWARD_DIST: {fd_str}, "
                f"ROUTE_AHEAD: {ra_str}, DIST_TO_GOAL: {self.dist_goal:.3f}, "
                f"STATE: {self.state}, TASK: {self.task_state}"
            )


# ============================================================
# STEP 12: Robot records + agent construction (M15/M16 wiring)
# ============================================================
# Live per-robot records: shared with the OTHER agents' forward-cone
# attribution so every robot can "see" the other Carters by LiDAR.
robot_entities = []
for i, (_rp, sx, sy) in enumerate(ROBOT_SPECS):
    robot_entities.append(
        {
            "path": _rp,
            "tag": f"CARTER_{i + 1}",
            "kind": "robot",
            "idx": 2 + i,          # traffic uses 0/1
            "center": np.array([sx, sy], dtype=np.float64),
            "half": robot_halves[i],
            "detected": False,
            "reacted": False,
            "cleared": False,
        }
    )

agents = []
for i, (_rp, sx, sy) in enumerate(ROBOT_SPECS):
    a = CarterAgent(
        idx=i,
        tag=f"CARTER_{i + 1}",
        art=articulations[i],
        lidar=lidars[i],
        lidar_path=f"{_rp}/chassis_link/lidar",
        graph_path=graphs[i],
        start_xy=(sx, sy),
        rec=robot_entities[i],
        half=robot_halves[i],
    )
    a.lidar_initialized = lidar_init_flags[i]
    a.forward_entities = list(traffic_entities) + [
        r for r in robot_entities if r["idx"] != robot_entities[i]["idx"]
    ]
    agents.append(a)

# ============================================================
# STEP 12b: PHASE 3C - COMMAND HANDOVER ARMED FOR ALL THREE CARTERS
#             (PHASE3C_HANDOVER=1)
# ============================================================
# /ActionGraph_1, /ActionGraph_2 and /ActionGraph_3 each keep their
# EXACT Phase-2 pipeline:
#     QuinticPathPlanner -> CheckGoal2D -> StanleyControlPID
#         -> DifferentialController -> ArticulationController
# NOTHING is detached here.  The three handovers are only ARMED: every
# Carter keeps driving itself with the proven Phase-2 stack and runs the
# whole mission untouched, and each Carter's single Stanley ->
# DifferentialController edge is detached only when the first Twist for
# THAT Carter arrives (_ros2_handover_engage).  From that moment that
# Carter's /carterN/cmd_vel occupies the slot Stanley held.
#
# Quintic, CheckGoal2D, Stanley, the DifferentialController wheel
# geometry and the ArticulationController joint list are all left
# exactly as they are, for all three robots.  Each Carter keeps its own
# independent engagement state, so:
#   - PHASE3C_HANDOVER=1 alone engages nothing;
#   - engaging Carter 2 does not detach Carter 1's or Carter 3's edge;
#   - Carter 1's verified Phase-3B behaviour is the same code path, only
#     now indexed by robot instead of hard-coded to agents[0].
# With PHASE3C_HANDOVER=0 this whole block is skipped.
if PHASE3C_HANDOVER:
    if ros2.get("ext_ok"):
        log_print(
            "PHASE3C_HANDOVER: ARMED graphs=/ActionGraph_1,/ActionGraph_2,"
            "/ActionGraph_3 robots=CARTER_1,CARTER_2,CARTER_3 - Phase-2 "
            "control paths still connected; each Stanley->DifferentialController "
            "edge detaches only on that Carter's FIRST /carterN/cmd_vel "
            "(QuinticPathPlanner + Stanley stay running, manual_mode safety "
            "latch unchanged, no robot engages another robot)"
        )
    else:
        log_print(
            "PHASE3C_HANDOVER: FAILED - ROS 2 bridge not up, all three "
            "Carters stay on the Phase-2 control path"
        )
else:
    log_print(
        "PHASE3C_HANDOVER: DISABLED (PHASE3C_HANDOVER=0) - Phase-2 "
        "control paths untouched for Carter 1, Carter 2 and Carter 3"
    )

# Per-robot trajectory prim + floating state marker
for i, a in enumerate(agents):
    try:
        traj = UsdGeom.Points.Define(stage, f"/World/Trajectory_{i + 1}")
        traj.CreatePointsAttr([])
        a.traj_prim = traj
        try:
            traj.CreateWidthsAttr([])
        except Exception as e:
            log_print(f"{a.tag}_TRAJECTORY_WIDTHS_SKIPPED: {e}")
        try:
            traj.CreateDisplayColorAttr([Gf.Vec3f(*TRAJ_COLORS[i])])
        except Exception:
            try:
                _pv = UsdGeom.Primvar(traj.GetPrim(), "displayColor")
                _pv.Set([Gf.Vec3f(*TRAJ_COLORS[i])])
                _pv.SetInterpolation("constant")
            except Exception as e:
                log_print(f"{a.tag}_TRAJECTORY_COLOR_SKIPPED: {e}")
        log_print(
            f"{a.tag}_TRAJECTORY_HISTORY: /World/Trajectory_{i + 1} "
            f"(UsdGeom.Points, 1 point / {TRAJ_STEP} steps, cap {TRAJ_MAX_POINTS}, "
            f"color {TRAJ_COLORS[i]})"
        )
    except Exception as e:
        log_print(f"{a.tag}_TRAJECTORY_INITIALIZATION_ERROR: {e}")

    marker_xf = UsdGeom.Xformable(
        stage.DefinePrim(f"/World/StateMarker_{i + 1}", "Xform")
    )
    a.state_op = marker_xf.AddTranslateOp()
    a.state_op.Set(
        Gf.Vec3f(float(a.start_pos[0]), float(a.start_pos[1]), 1.7)
    )
    marker_xf.AddScaleOp().Set(Gf.Vec3f(0.3, 0.3, 0.3))
    a.marker_cube = UsdGeom.Cube.Define(stage, f"/World/StateMarker_{i + 1}/Geom")
    a.marker_cube.CreateSizeAttr(1.0)
    a.marker_cube.CreateDisplayColorAttr([Gf.Vec3f(*STATE_COLORS["NAVIGATE"])])

# Fixed perspective view over the aisle (lightweight, one-shot).
if not HEADLESS:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(np.array([3.0, -15.0, 9.0]), np.array([-4.5, -6.8, 0.5]))
        log_print("CAMERA_VIEW: fixed perspective over the navigation aisle")
    except Exception as e:
        log_print(f"CAMERA_VIEW: not set ({e})")

# ------------------------------------------------------------
# Optional screenshot capture (OFF unless SCREENSHOT_DIR is set).
# Purely observational: it never calls simulation_app.update(), so the
# physics timeline / step count of the verified run is untouched.
# ------------------------------------------------------------
_SCREENSHOT_DIR = os.environ.get("SCREENSHOT_DIR", "")
_SCREENSHOT_STEPS = frozenset(
    int(x) for x in os.environ.get("SCREENSHOT_STEPS", "").split(",") if x.strip()
)


def maybe_screenshot(step):
    if not _SCREENSHOT_DIR or step not in _SCREENSHOT_STEPS:
        return
    try:
        from omni.kit.viewport.utility import (
            capture_viewport_to_file,
            get_active_viewport,
        )
        os.makedirs(_SCREENSHOT_DIR, exist_ok=True)
        _p = os.path.join(_SCREENSHOT_DIR, f"frame_{step:05d}.png")
        _vp = get_active_viewport()
        if _vp is None:
            from omni.kit.viewport.utility import get_active_viewport_window
            _vp = get_active_viewport_window()
        # schedules the write on the viewport's capture queue - it never
        # calls simulation_app.update(), so physics stays step-aligned
        capture_viewport_to_file(_vp, _p)
        log_print(f"SCREENSHOT: {_p} (capture scheduled)")
    except Exception as _e:
        log_print(f"SCREENSHOT: failed at step {step} ({_e})")


# Let colliders register with physics
for _ in range(10):
    simulation_app.update()

# ------------------------------------------------------------
# Initial goals + task state (allocation already done): each robot
# starts in TASK_ASSIGNED with its pickup waypoint live in the
# planner (created above), then immediately enters GO_TO_PICKUP.
# ------------------------------------------------------------
for a in agents:
    task = TASKS[a.idx] if TASKS[a.idx]["assigned_robot"] == a.idx else next(
        t for t in TASKS if t["assigned_robot"] == a.idx
    )
    gx, gy, _ = first_goals[a.idx]
    a.task = task
    a.task_state = "TASK_ASSIGNED"
    a.goal = np.array([gx, gy, 0.0], dtype=np.float64)
    r = a.goal[:2] - a.pos[:2]
    n = float(np.linalg.norm(r))
    if n > 1e-6:
        a.route_dir = r / n
    log_print(
        f"{a.tag}_TASK_ASSIGNED: {task['id']} "
        f"(pickup {task['pickup'][0]:.3f},{task['pickup'][1]:.3f} -> "
        f"delivery {task['delivery'][0]:.3f},{task['delivery'][1]:.3f}); "
        f"state TASK_ASSIGNED -> GO_TO_PICKUP"
    )
    a.task_state = "GO_TO_PICKUP"
    log_print(
        f"CARTER_{a.idx + 1}_GOAL: ({gx:.3f}, {gy:.3f}, 0.000) "
        f"[{task['id']} pickup]"
    )

log_print(
    f"Running {len(agents)} Carter agents (max {MAX_STEPS} steps, "
    f"CLEAR>{CLEAR_T} m slow zone, EMERGENCY<={EMERG_T} m stop zone, "
    f"observe {FLEET_OBSERVE_STEPS} steps after all tasks)"
)
log_print(
    "M16_YIELD_RULE: forward entity from LiDAR attribution only; "
    "robot-vs-robot safety = slow <= 2.5 m, STOP <= 1.5 m, resume > 2.5 m "
    "(no hard-coded yielding, no ORCA/RVO/MPC/RL)"
)

# ============================================================
# STEP 13: Closed-loop fleet run
# ============================================================
min_pair_gap = float("inf")
min_pair_gap_at = None
fleet_hold = 0
all_complete = False

for step in range(MAX_STEPS):
    # Drive traffic BEFORE physics/LiDAR update so this frame's scans
    # reflect the scripted pose for this instant (frozen pattern).
    update_traffic(step / physics_rate)
    simulation_app.update()
    maybe_screenshot(step)

    # Phase 3A: pump the ROS 2 monitor (observation only, no sim writes)
    _ros2_step(step)

    # Phase 3B: the handover only engages once a /carter1/cmd_vel exists
    _ros2_handover_engage(step)

    # --- sense: pose + scan + attribution (each robot, own LiDAR) ---
    for a in agents:
        a.sense(step)

        # --- M16 robot-vs-robot rectangle gap (conservative) ---
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            ai, aj = agents[i], agents[j]
            g = rect_gap(ai.pos, ai.yaw, ai.half, aj.pos, aj.yaw, aj.half)
            if g < ai.rr_min_gap:
                ai.rr_min_gap = g
            if g < aj.rr_min_gap:
                aj.rr_min_gap = g
            if g < min_pair_gap:
                min_pair_gap = g
                min_pair_gap_at = (step, ai.tag, aj.tag, g)
            if g < ROBOT_COLLISION_GAP:
                if not ai.rr_collision:
                    log_print(
                        f"{ai.tag}_ROBOT_COLLISION_WARNING: gap {g:.4f} m to "
                        f"{aj.tag} at step {step}"
                    )
                if not aj.rr_collision:
                    log_print(
                        f"{aj.tag}_ROBOT_COLLISION_WARNING: gap {g:.4f} m to "
                        f"{ai.tag} at step {step}"
                    )
                ai.rr_collision = True
                aj.rr_collision = True
            if step == 0:
                # actual initial pairwise footprint gap (measured, not assumed)
                log_print(
                    f"INITIAL_PAIR_GAP: {ai.tag} x {aj.tag} = {g:.4f} m "
                    f"(half {ai.half.tolist()} @ ({ai.pos[0]:.3f},{ai.pos[1]:.3f},"
                    f"yaw {ai.yaw:.2f}) vs {aj.half.tolist()} @ "
                    f"({aj.pos[0]:.3f},{aj.pos[1]:.3f},yaw {aj.yaw:.2f}))"
                )

    # --- reactive navigation + control + task layer ---
    for a in agents:
        a.navigate(step)

    # Phase 3B: apply /carter1/cmd_vel AFTER navigate(), so any command
    # the Phase-2 safety machine asserted this step already stands and is
    # never overwritten by the operator.
    _ros2_handover_step(step)

    # --- visualization / progress logs ---
    for a in agents:
        a.viz(step)

    # --- fleet-level completion hold ---
    if not all_complete and all(t["status"] == "COMPLETED" for t in TASKS):
        all_complete = True
        log_print(
            f"ALL_TASKS_COMPLETED_STEP: {step} "
            f"({step / physics_rate:.1f} s sim time)"
        )
    if all_complete:
        fleet_hold += 1
        if fleet_hold >= FLEET_OBSERVE_STEPS:
            log_print(
                f"Observation complete at step {step} "
                f"({FLEET_OBSERVE_STEPS} steps after all tasks)"
            )
            break

# ============================================================
# STEP 14: Records and measurements
# ============================================================
log_print("---- FLEET RECORDS ----")
m15_init_flags = []
m15_lidar_flags = []
m15_moved_flags = []
m15_traj_flags = []
for a in agents:
    _p, _r = a.art.get_world_pose()
    _p = np.array(_p, dtype=np.float64)
    _final_speed = float(np.mean(a.speeds[-10:])) if a.speeds else float("nan")
    _disp = float(np.linalg.norm(_p[:2] - a.start_pos[:2]))
    _goal_dist = float(
        math.hypot(
            (a.goal[0] if a.goal is not None else _p[0]) - _p[0],
            (a.goal[1] if a.goal is not None else _p[1]) - _p[1],
        )
    )
    _hist = np.array(a.position_history)
    if len(_hist) >= 61:
        _drift = float(np.linalg.norm(_hist[-1, :2] - _hist[-61, :2]))
    elif len(_hist) >= 2:
        _drift = float(np.linalg.norm(_hist[-1, :2] - _hist[0, :2]))
    else:
        _drift = 0.0
    a.final_pos = _p
    a.final_speed = _final_speed
    a.moved = _disp > 0.3
    a.final_goal_dist = _goal_dist
    a.recent_drift = _drift

    log_print(
        f"{a.tag}_INITIAL_POSITION: ({a.start_pos[0]:.3f}, "
        f"{a.start_pos[1]:.3f}, {a.start_pos[2]:.3f})"
    )
    log_print(
        f"{a.tag}_FINAL_POSITION: ({_p[0]:.3f}, {_p[1]:.3f}, {_p[2]:.3f})"
    )
    log_print(f"{a.tag}_MOVED_DISTANCE: {_disp:.4f}")
    log_print(f"{a.tag}_FINAL_SPEED: {_final_speed:.4f}")
    log_print(f"{a.tag}_GOAL_DISTANCE: {_goal_dist:.4f}")
    log_print(f"{a.tag}_DRIFT_AFTER_STOP: {_drift:.4f}")
    log_print(
        f"{a.tag}_MIN_FORWARD_DISTANCE: "
        f"{a.min_forward if np.isfinite(a.min_forward) else -1.0:.4f}"
    )
    _mf_az = f"{a.min_fwd_az:.1f}" if a.min_fwd_az is not None else "n/a"
    _mf_xy = (
        f"({a.min_fwd_pose[0]:.3f}, {a.min_fwd_pose[1]:.3f})"
        if a.min_fwd_pose
        else "n/a"
    )
    _mf_yw = f"{a.min_fwd_pose[2]:.1f}" if a.min_fwd_pose else "n/a"
    log_print(
        f"{a.tag}_MIN_FORWARD_AT: step {a.min_fwd_step} POSITION: {_mf_xy} "
        f"YAW: {_mf_yw} SENSOR_AZ: {_mf_az}"
    )
    log_print(f"{a.tag}_MIN_CLEARANCE_CENTER: {a.min_clearance:.4f}")
    log_print(f"{a.tag}_MIN_ROBOT_GAP: {a.rr_min_gap:.4f}")
    log_print(f"{a.tag}_AVOIDANCE_EVENTS_TOTAL: {a.avoid_events}")
    for n_, ev in enumerate(a.events, start=1):
        # forward/left/right can be None on a RECOVERY-forced avoid
        _ff = (
            f"{ev['forward']:.4f}"
            if isinstance(ev.get("forward"), (int, float))
            else "nan"
        )
        _fl = (
            f"{ev['left']:.4f}" if isinstance(ev.get("left"), (int, float)) else "nan"
        )
        _fr = (
            f"{ev['right']:.4f}"
            if isinstance(ev.get("right"), (int, float))
            else "nan"
        )
        log_print(
            f"{a.tag}_AVOIDANCE_EVENT_{n_}: STEP {ev['step']}, ENTITY {ev['tag']}, "
            f"PRIORITY {ev['priority']}, FORWARD_DISTANCE: {_ff}, "
            f"LEFT_CLEARANCE: {_fl}, RIGHT_CLEARANCE: {_fr}, "
            f"DIRECTION: {ev['direction']}"
        )
    log_print(f"{a.tag}_RECOVERY_EVENTS_TOTAL: {a.recovery_events}")
    log_print(f"{a.tag}_DIRECTION_CHANGES_TOTAL: {a.direction_changes}")
    log_print(f"{a.tag}_TRAFFIC_LIDAR_FRAMES: {a.dynamic_hit_frames}")
    log_print(f"{a.tag}_TRAFFIC_TRACKING_MISMATCH: {a.tracking_mismatch}")
    log_print(f"{a.tag}_TRAJECTORY_POINTS: {len(a.traj_points)}")
    for _t, _fl in a.ent_flags.items():
        log_print(
            f"{a.tag}_{_t}_FLAGS: detected={_fl['detected']} "
            f"reacted={_fl['reacted']} cleared={_fl['cleared']} "
            f"dist={_fl['dist'] if _fl['dist'] is None else round(_fl['dist'], 4)}"
        )
    log_print(
        f"{a.tag}_RR_FLAGS: detected={a.rr_detected} reacted={a.rr_reacted} "
        f"cleared={a.rr_cleared} stop_steps={a.rr_stop_steps} "
        f"min_gap={a.rr_min_gap:.4f}"
    )

    m15_init_flags.append(a.init_ok)
    m15_lidar_flags.append(
        a.lidar_initialized and a.lidar_data_available and a.lidar_rows == 16
    )
    m15_moved_flags.append(a.moved)
    m15_traj_flags.append(len(a.traj_points) > 0)

for _o in traffic_entities:
    log_print(
        f"{_o['tag']}_FINAL_POSITION: ({_o['center'][0]:.3f}, "
        f"{_o['center'][1]:.3f}, 0.000)"
    )
    log_print(f"{_o['tag']}_PATH_LENGTH: {_o['motion_len']:.4f} m")

if min_pair_gap_at is not None:
    log_print(
        f"MIN_PAIR_GAP: {min_pair_gap:.4f} at step {min_pair_gap_at[0]} "
        f"({min_pair_gap_at[1]} x {min_pair_gap_at[2]})"
    )
else:
    log_print(f"MIN_PAIR_GAP: {min_pair_gap:.4f}")

log_print("---- TASK RECORDS ----")
for task in TASKS:
    log_print(
        f"{task['id']}_RECORD: assigned=CARTER_{task['assigned_robot'] + 1} "
        f"pickup=({task['pickup'][0]:.3f},{task['pickup'][1]:.3f}) "
        f"delivery=({task['delivery'][0]:.3f},{task['delivery'][1]:.3f}) "
        f"status={task['status']} "
        f"completed_step={task.get('completed_step', -1)}"
    )

# ============================================================
# STEP 15: M15 + M16 + M17 verification markers
# ============================================================
visual_ok = True
lidar_ok = all(m15_lidar_flags)
moved_ok = all(m15_moved_flags)
traj_ok = all(m15_traj_flags)
init_ok = all(m15_init_flags)

m15_multi_amr = visual_ok and init_ok and lidar_ok and moved_ok and traj_ok

rr_detected_any = any(a.rr_detected for a in agents)
rr_reacted_any = any(a.rr_reacted for a in agents)
rr_cleared_any = any(a.rr_cleared for a in agents)
rr_stopped_any = any(a.rr_stop_steps > 0 for a in agents)
any_rr_collision = any(a.rr_collision for a in agents)
any_static_collision = any(a.static_collision for a in agents)
clearance_ok = min_pair_gap >= ROBOT_CLEARANCE_GAP

m16_avoidance = (
    rr_detected_any
    and rr_reacted_any
    and rr_cleared_any
    and clearance_ok
    and not any_rr_collision
    and not any_static_collision
)

tasks_assigned = all(t["assigned_robot"] is not None for t in TASKS)
tasks_completed = all(t["status"] == "COMPLETED" for t in TASKS)
m17_allocation = tasks_assigned  # allocation ran; completion checked below

safety_ok = (
    lidar_ok
    and all(
        np.isfinite(a.min_forward) and a.min_forward >= MIN_FORWARD_ALLOWED
        for a in agents
    )
    and all(a.min_clearance >= MIN_CLEARANCE_CENTER for a in agents)
)
stopped_ok = all(a.final_speed < 0.1 for a in agents)

log_print("---- VERIFICATION MARKERS ----")
log_print(f"VISUAL_DEMO_INITIALIZED: {'YES' if visual_ok else 'NO'}")
log_print(f"LIDAR_INITIALIZED: {'YES' if lidar_ok else 'NO'}")
log_print(f"LIDAR_SCAN_ROWS: {', '.join(str(a.lidar_rows) for a in agents)} (expected 16)")

for a in agents:
    log_print(f"{a.tag}_INITIALIZED: {'YES' if a.init_ok else 'NO'}")
    log_print(
        f"{a.tag}_LIDAR: "
        f"{'YES' if a.lidar_initialized and a.lidar_data_available and a.lidar_rows == 16 else 'NO'}"
    )
    log_print(f"{a.tag}_MOVED: {'YES' if a.moved else 'NO'}")
    log_print(f"{a.tag}_COLLISION: {'YES' if (a.static_collision or a.rr_collision) else 'NO'}")

log_print(
    f"MILESTONE_15_MULTI_AMR: {'YES' if m15_multi_amr else 'NO'}"
)

log_print(f"MULTI_ROBOT_DETECTED: {'YES' if rr_detected_any else 'NO'}")
log_print(
    f"MULTI_ROBOT_REACTION: {'YES' if rr_reacted_any else 'NO'}"
)
log_print(
    f"MULTI_ROBOT_CLEARANCE: {'YES' if (rr_cleared_any and clearance_ok) else 'NO'}"
)
log_print(f"MULTI_ROBOT_STOP_OBSERVED: {'YES' if rr_stopped_any else 'NO'}")
log_print(f"MULTI_ROBOT_COLLISION: {'YES' if any_rr_collision else 'NO'}")
log_print(f"STATIC_COLLISION: {'YES' if any_static_collision else 'NO'}")
log_print(f"MIN_PAIR_GAP: {min_pair_gap:.4f} (required >= {ROBOT_CLEARANCE_GAP})")
log_print(
    f"MILESTONE_16_MULTI_ROBOT_AVOIDANCE: {'YES' if m16_avoidance else 'NO'}"
)

log_print(
    f"MILESTONE_17_TASK_ALLOCATION: {'YES' if (m17_allocation and tasks_completed) else 'NO'}"
)
for task in TASKS:
    log_print(
        f"{task['id']}_ASSIGNED: "
        f"{'YES' if task['assigned_robot'] is not None else 'NO'}"
    )
    log_print(f"{task['id']}_COMPLETED: {'YES' if task['status'] == 'COMPLETED' else 'NO'}")
log_print(f"ALL_TASKS_COMPLETED: {'YES' if tasks_completed else 'NO'}")

for a in agents:
    log_print(f"{a.tag}_GOAL_REACHED: {'YES' if a.goal_reached else 'NO'}")

log_print(
    f"SAFETY: {'PASS' if safety_ok else 'FAIL'} "
    f"(min forward "
    f"{min(a.min_forward for a in agents):.4f} >= {MIN_FORWARD_ALLOWED}, "
    f"min clearance {min(a.min_clearance for a in agents):.4f} >= "
    f"{MIN_CLEARANCE_CENTER}, min pair gap {min_pair_gap:.4f} >= "
    f"{ROBOT_CLEARANCE_GAP})"
)

# ============================================================
# STEP 16: Test result (M15 + M16 + M17 combined verdict)
# ============================================================
test_passed = (
    m15_multi_amr
    and m16_avoidance
    and m17_allocation
    and tasks_completed
    and not any_rr_collision
    and not any_static_collision
    and safety_ok
    and stopped_ok
)

log_print(f"FLEET_DEMO: {'PASS' if test_passed else 'FAIL'}")

# ============================================================
# STEP 16b: PHASE 3A/3C - ROS 2 VERIFICATION SUMMARY
# ------------------------------------------------------------
# Printed AFTER every Phase-2 marker, so nothing above can be affected
# by it.  "PASS" here means an independent ROS 2 subscriber actually
# RECEIVED odom + scan + tf for ALL THREE Carters during this run - not
# merely that the extension started or that the topic names exist.
# ============================================================
ros2_published_ok = False
ros2_verdict = "FAIL"
if ros2["requested"]:
    # all 9 sensor topics (3 Carters x odom/scan/tf) must carry data
    _pub_ok = all(
        ros2["counts"].get(t, 0) > 0 for t in ROS2_SENSOR_TOPICS
    )
    _graph_ok = bool(ros2["enabled"]) and not ros2["errors"]
    ros2_published_ok = _graph_ok and _pub_ok

    log_print(
        f"PHASE3_ROS2_STATE: ext={ROS2_EXT_ID} ext_ok={ros2['ext_ok']} "
        f"graphs={','.join(ros2.get('graphs') or [])} "
        f"graph_ok={ros2['enabled']}"
    )
    log_print(
        "PHASE3_ROS2_MSG_COUNTS: "
        + " ".join(
            f"{t}={ros2['counts'].get(t, 0)}" for t in ROS2_MONITORED
        )
        + " (received by /fleet_ros2_monitor)"
    )
    for _t in ROS2_MONITORED:
        if _t in ros2["first"]:
            log_print(f"PHASE3_ROS2_FIRST_MSG {_t}: {ros2['first'][_t]}")
        else:
            log_print(f"PHASE3_ROS2_FIRST_MSG {_t}: NEVER RECEIVED")
    log_print(f"PHASE3_ROS2_MONITOR_SPIN_ERRORS: {ros2['spin_errors']}")
    log_print(
        "PHASE3_ROS2_CMD_VEL_OGN_CHANGES: "
        + " ".join(
            f"{b['ns']}={b['cmd_ogn_changes']}" for b in ros2["bots"]
        )
        + " (changes seen on the OGN subscriber outputs)"
    )
    _cmd_total = sum(b["cmd_rx"] for b in ros2["bots"])
    log_print(
        f"PHASE3_ROS2_CMD_VEL: RECEIVED total={_cmd_total} "
        + " ".join(f"{b['ns']}={b['cmd_rx']}" for b in ros2["bots"])
    )
    for _b in ros2["bots"]:
        if _b["cmd_rx"] > 0 and _b["cmd_last"] is not None:
            log_print(
                f"PHASE3_ROS2_CMD_VEL {_b['ns']}: RECEIVED {_b['cmd_rx']} "
                f"message(s) last lin.x={_b['cmd_last'][0]:.3f} "
                f"angular.z={_b['cmd_last'][2]:.3f}"
            )
        else:
            log_print(
                f"PHASE3_ROS2_CMD_VEL {_b['ns']}: NOT EXERCISED (no message "
                f"published to /{_b['ns']}/cmd_vel during this run)"
            )
    if ros2["errors"]:
        log_print("PHASE3_ROS2_ERRORS: " + " | ".join(ros2["errors"]))
    ros2_verdict = "PASS" if ros2_published_ok else "FAIL"
    log_print(
        f"PHASE3_ROS2_VERDICT: {ros2_verdict} "
        "(independent subscriber received odom + scan + tf on "
        + " + ".join(ROS2_SENSOR_TOPICS)
        + ")"
    )
    log_print(
        "PHASE3_ROS2_PHASE2_UNCHANGED: /ActionGraph_1..3 not modified, "
        "no collision geometry added, thresholds/tasks/LiDAR untouched"
    )

    # --- optional hold: keep the timeline PLAYING so an external
    #     `ros2 topic list` / `ros2 topic echo` window is guaranteed.
    #     Runs after every marker and record, so no Phase-2 number is
    #     affected by it. ---
    if PHASE3_ROS2_HOLD_S > 0:
        _hold_ticks = int(PHASE3_ROS2_HOLD_S * physics_rate)
        log_print(
            f"PHASE3_ROS2_HOLD: {PHASE3_ROS2_HOLD_S:.0f}s "
            f"({_hold_ticks} ticks) timeline still PLAYING - external "
            "topic verification window"
        )
        # Snapshot every Carter's world pose at hold entry so the log can
        # prove ACTUAL physical movement during the hold, independently
        # of any ROS 2 message.
        _hold_start = {}
        for _i, _a in enumerate(agents):
            _p, _q = _a.art.get_world_pose()
            _hold_start[_i] = (
                np.array(_p, dtype=np.float64), float(_a.speed)
            )
            log_print(
                f"PHASE3C_HOLD_START: robot={_a.tag} ns=/{ROS2_ROBOTS[_i]} "
                f"pos=({float(_p[0]):.4f},{float(_p[1]):.4f}) "
                f"speed={_hold_start[_i][1]:.4f} engaged="
                f"{_a.graph_path in ROS2_HANDOVER_GRAPHS}"
            )
        # Phase 3C: while a command is handed over that Carter is MOVING
        # again, so its own LiDAR sense + reactive machine must keep
        # running for the whole hold.  That is what makes the teleop test
        # safe: as soon as something enters the CLEAR_T band the Phase-2
        # machine takes manual_mode and _ros2_handover_apply stands down.
        # sense() is read-only, so it is run for every Carter to keep the
        # additive envelope's neighbour poses fresh; navigate() - the
        # control-affecting call - runs only for ENGAGED Carters.
        # With PHASE3C_HANDOVER=0 (or no Twist seen) this branch is
        # skipped entirely and the hold is byte-identical to Phase 3A/3B.
        _hs = step + 1
        for _h in range(_hold_ticks):
            _ros2_step(-1)          # keep the monitor pumping
            _ros2_handover_engage(_hs)
            if PHASE3C_HANDOVER and ROS2_HANDOVER_GRAPHS:
                for _a in agents:
                    _a.sense(_hs)
                for _a in agents:
                    if _a.graph_path in ROS2_HANDOVER_GRAPHS:
                        _a.navigate(_hs)
                _ros2_handover_step(_hs)
            simulation_app.update()
            _hs += 1
        log_print(
            "PHASE3_ROS2_MSG_COUNTS_AFTER_HOLD: "
            + " ".join(
                f"{t}={ros2['counts'].get(t, 0)}" for t in ROS2_MONITORED
            )
        )
        # In-sim world-frame movement evidence for each Carter.
        for _i, _a in enumerate(agents):
            _p, _q = _a.art.get_world_pose()
            _p = np.array(_p, dtype=np.float64)
            _dp = _p[:2] - _hold_start[_i][0][:2]
            _dd = float(np.hypot(_dp[0], _dp[1]))
            log_print(
                f"PHASE3C_HOLD_MOVE: robot={_a.tag} ns=/{ROS2_ROBOTS[_i]} "
                f"engaged={_a.graph_path in ROS2_HANDOVER_GRAPHS} "
                f"start=({_hold_start[_i][0][0]:.4f},"
                f"{_hold_start[_i][0][1]:.4f}) "
                f"end=({float(_p[0]):.4f},{float(_p[1]):.4f}) "
                f"world_delta={_dd:.4f} m "
                f"speed_start={_hold_start[_i][1]:.4f} "
                f"speed_end={float(_a.speed):.4f}"
            )
        log_print("PHASE3_ROS2_HOLD: done")
elif not ros2["requested"]:
    log_print("PHASE3_ROS2_VERDICT: DISABLED (PHASE3_ROS2=0)")

# ============================================================
# STEP 16c: PHASE 3C - PER-CARTER COMMAND HANDOVER SUMMARY
# ============================================================
if PHASE3C_HANDOVER:
    _eng_count = 0
    for _i, _a in enumerate(agents):
        _b = ros2["bots"][_i]
        _engaged = _a.graph_path in ROS2_HANDOVER_GRAPHS
        _eng_count += 1 if _engaged else 0
        _park = (_a.state == "GOAL_REACHED" and _a._d is not None)
        if not _engaged:
            _auth = "PHASE2-STANLEY"
        elif _a.manual_mode and not _park:
            _auth = "PHASE2-SAFETY-manual_mode"
        else:
            _auth = f"ROS2-/{_b['ns']}/cmd_vel"
        log_print(
            f"PHASE3C_HANDOVER_SUMMARY: robot={_a.tag} ns=/{_b['ns']} "
            f"engaged={_engaged} graph={_a.graph_path} "
            f"cmd_vel_received={_b['cmd_rx']} "
            f"cmd_vel_applied_msgs={_b['handover_applied_msgs']} "
            f"writes={_b['handover_writes']} distinct_cmds="
            f"{_b['handover_changes']} safety_holds="
            f"{_b['handover_safety_holds']} envelope_stops="
            f"{_b['handover_env_stops']} "
            f"authority_at_end={_auth} state_at_end={_a.state}"
        )
    log_print(
        f"PHASE3C_HANDOVER_FLEET: engaged_count={_eng_count}/3 "
        f"graphs_owned={sorted(ROS2_HANDOVER_GRAPHS)} "
        "each Carter engaged only by its OWN first Twist"
    )
    log_print(
        "PHASE3C_HANDOVER_STACK: QuinticPathPlanner -> CheckGoal2D -> "
        "StanleyControlPID -> DifferentialController -> "
        "ArticulationController, unchanged per Carter; only that Carter's "
        "Stanley->DifferentialController edge is swapped with its own "
        "/carterN/cmd_vel, and only after that Carter's first Twist. "
        "safety thresholds / task coordinates / LiDAR / warehouse untouched"
    )
else:
    log_print(
        "PHASE3C_HANDOVER_SUMMARY: engaged_count=0/3 "
        "(PHASE3C_HANDOVER=0) - Phase-2 control paths active for "
        "Carter 1, Carter 2 and Carter 3, ROS 2 bridge observation-only"
    )

# ---- Phase 3A ROS 2 teardown (after the hold, before app cleanup) ----
if ros2.get("executor") is not None:
    try:
        ros2["executor"].shutdown(timeout_sec=1.0)
    except Exception:
        pass
if ros2.get("node") is not None:
    try:
        ros2["node"].destroy_node()
    except Exception as e:
        log_print(f"PHASE3_ROS2_TEARDOWN: node destroy: {e}")
if ros2.get("ctx") is not None:
    try:
        ros2["ctx"].shutdown()
    except Exception as e:
        log_print(f"PHASE3_ROS2_TEARDOWN: context shutdown: {e}")
    ros2["ctx"] = None
    ros2["node"] = None
    ros2["executor"] = None

# ============================================================
# STEP 17: Cleanup
# ============================================================
my_world.stop()
simulation_app.update()
simulation_app.close()

log_print("TEST_COMPLETED")

sys.exit(0 if test_passed else 1)
