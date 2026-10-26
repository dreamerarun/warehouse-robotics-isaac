#!/usr/bin/env python3
"""
Carter Autonomous Navigation Test in a Static Warehouse - Isaac Sim 5.1

Reuses the proven navigation pipeline from test_carter_navigation.py:
- QuinticPathPlanner
- StanleyControlPID
- DifferentialController
- CheckGoal2D

Adds only a lightweight static warehouse:
- 20m x 20m floor
- 4 outer walls
- 3 pairs of box racks with clear aisles
- static collision geometry (no ROS2/Nav2/LiDAR/cameras)

Layout (top view, meters):
    x: -10 .. 10,  y: -10 .. 10
    Racks (4 x 1.5 x 2) at (+-5, -4), (+-5, 0), (+-5, 4)
    Central aisle (x: -3..3): Carter START (0, -7)
    North cross-aisle (y: 4.75..10): GOAL (0, 7)

Start: (0, -7, 0.1)
Goal: (0, 7, 0)
"""

import sys
import os
import numpy as np

# Write start marker
with open("/tmp/carter_warehouse_results.txt", "w") as f:
    f.write("TEST_STARTED\n")

def log_print(msg):
    """Print to console and write to file"""
    print(msg)
    with open("/tmp/carter_warehouse_results.txt", "a") as f:
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
from pxr import Gf, PhysicsSchemaTools, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, open_stage_async
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.rotations import quat_to_euler_angles

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
# ------------------------------------------------------------
WALL_COLOR = (0.6, 0.6, 0.65)
RACK_COLOR = (0.25, 0.4, 0.7)

# Outer walls (height 3m, thickness 0.4m) at the +/-10m boundaries
add_warehouse_box("/World/Wall_South", (0, -10, 1.5), (20.6, 0.4, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_North", (0, 10, 1.5), (20.6, 0.4, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_West", (-10, 0, 1.5), (0.4, 20.6, 3.0), WALL_COLOR)
add_warehouse_box("/World/Wall_East", (10, 0, 1.5), (0.4, 20.6, 3.0), WALL_COLOR)

# 3 pairs of rack/box obstacles (4m wide x 1.5m deep x 2m tall)
# Left column x=-5 (spans x: -7..-3), right column x=+5 (spans x: 3..7)
# -> central aisle x: -3..3 (6m clear), side aisles x: 7..10 / -10..-7 (3m clear)
# Rows at y = -4, 0, 4 -> cross aisles of 2.5m between rows
RACK_SIZE = (4.0, 1.5, 2.0)
RACK_Y = (-4.0, 0.0, 4.0)
for i, y in enumerate(RACK_Y):
    add_warehouse_box(f"/World/Rack_L{i + 1}", (-5.0, y, 1.0), RACK_SIZE, RACK_COLOR)
    add_warehouse_box(f"/World/Rack_R{i + 1}", (5.0, y, 1.0), RACK_SIZE, RACK_COLOR)

log_print("Warehouse loaded (walls + 6 racks)")

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
# STEP 5: Create SingleArticulation Carter
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
# STEP 6: Setup OmniGraph with navigation pipeline
# ============================================================
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
            # Target position and orientation (goal in north cross-aisle)
            ("QuinticPathPlanner.inputs:targetPosition", (0, 7, 0)),
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

# ============================================================
# STEP 7: Reset Carter position and initialize
# ============================================================
# Reset Carter to start of central aisle
ar.set_world_pose(position=[0, -7, 0.1], orientation=[0, 0, 0, 1])
ar.set_world_velocity([0, 0, 0, 0, 0, 0])

# Reset controller
og.Controller.attribute(graph_path + "/DifferentialController.inputs:linearVelocity").set(0)
og.Controller.attribute(graph_path + "/DifferentialController.inputs:angularVelocity").set(0)

# Wait for robot to settle
for _ in range(10):
    simulation_app.update()

# Get start position
start_pos, start_rot = ar.get_world_pose()
log_print(f"START_POSITION: {start_pos}")

# Target position
target_pos = (0, 7, 0)
log_print(f"TARGET_POSITION: {target_pos}")

# ============================================================
# STEP 8: Run navigation simulation
# ============================================================
max_steps = 2000  # Maximum simulation steps
reached_goal = False
distance_to_goal = float('inf')

for step in range(max_steps):
    simulation_app.update()

    # Get current position
    current_pos, current_rot = ar.get_world_pose()

    # Calculate distance to goal (2D)
    distance_to_goal = np.linalg.norm(np.array(current_pos[:2]) - np.array(target_pos[:2]))

    # Check reached goal from CheckGoal2D node
    reached_goal_attr = og.Controller.attribute(graph_path + "/CheckGoal2D.outputs:reachedGoal")
    reached_goal_val = reached_goal_attr.get()

    # Print progress every 50 steps
    if step % 50 == 0:
        log_print(f"STEP {step}: CURRENT_POSITION: {current_pos}, DISTANCE_TO_GOAL: {distance_to_goal:.4f}, REACHED_GOAL: {reached_goal_val}")

    # Check if reached goal (both x,y position and orientation)
    if reached_goal_val is not None:
        if isinstance(reached_goal_val, (list, tuple, np.ndarray)):
            reached_goal = reached_goal_val[0] and reached_goal_val[1]  # position and orientation
        else:
            reached_goal = bool(reached_goal_val)

    # Also check by distance as backup
    if distance_to_goal < 0.5:  # Within 0.5m of goal
        reached_goal = True

    if reached_goal:
        log_print(f"Goal reached at step {step}!")
        break

# Final position
final_pos, final_rot = ar.get_world_pose()
final_distance = np.linalg.norm(np.array(final_pos[:2]) - np.array(target_pos[:2]))

log_print(f"FINAL_POSITION: {final_pos}")
log_print(f"DISTANCE_TO_GOAL: {final_distance:.4f}")
log_print(f"REACHED_GOAL: {reached_goal}")

# Check if Carter actually moved
moved = np.linalg.norm(np.array(final_pos[:2]) - np.array(start_pos[:2])) > 0.01
log_print(f"CARTER_MOVED: {moved}")

# ============================================================
# STEP 9: Determine test result
# ============================================================
test_passed = reached_goal and moved and final_distance < 1.0

log_print(f"NAVIGATION_STATUS: {'SUCCESS' if test_passed else 'FAILED'}")
log_print(f"TEST_RESULT: {'PASS' if test_passed else 'FAIL'}")

# ============================================================
# STEP 10: Cleanup
# ============================================================
my_world.stop()
simulation_app.update()
simulation_app.close()

log_print("TEST_COMPLETED")
