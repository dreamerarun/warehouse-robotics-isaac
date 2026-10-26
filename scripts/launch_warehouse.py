#!/usr/bin/env python3
"""
Minimum Warehouse Robotics Prototype for Isaac Sim 5.1

Launches a simple warehouse with one Carter robot, navigation, and teleoperation.
Optimized for RTX 4050 6 GB GPU.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Isaac Sim Warehouse Robotics - Minimum Prototype"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["autonomous", "teleop"],
        default="autonomous",
        help="Simulation mode: autonomous or teleop (default: autonomous)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim in headless mode (no rendering, saves VRAM)"
    )
    
    args = parser.parse_args()
    
    # Simulation configuration optimized for RTX 4050 6 GB
    if args.headless:
        sim_config = {
            "renderer": "iray",
            "headless": True,
            "window_width": 0,
            "window_height": 0,
            "limit_sampling_buffer_size": True,
        }
    else:
        sim_config = {
            "renderer": "RaytracedLighting",
            "headless": False,
            "window_width": 1920,
            "window_height": 1080,
            "limit_sampling_buffer_size": True,
            "depth_value_enabled": False,  # Save VRAM
        }
    
    # Launch Isaac Sim SimulationApp
    from isaacsim import SimulationApp
    simulation_app = SimulationApp(sim_config)
    
    # Now import Isaac Sim APIs
    import isaacsim
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.physics.fabric.utils import get_fabric
    from isaacsim import omni
    
    print("=" * 60)
    print("WAREHOUSE ROBOTICS - MINIMUM PROTOTYPE")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Headless: {args.headless}")
    print("=" * 60)
    
    # ==========================================
    # 1. CREATE WAREHOUSE SCENE
    # ==========================================
    print("\n[1/5] Creating warehouse scene...")
    
    stage = get_current_stage()
    
    # Set up physics scene
    physics_scene = stage.GetPhysicsScene()
    if physics_scene:
        physics_api = physics_scene.GetPhysicsSceneAPI()
        if physics_api:
            physics_api.CreateGravityZAttr().Set(-9.81)
    
    # Add floor
    floor_path = "/World/Floor"
    add_reference_to_stage(
        usd_path="/Isaac/Props/Grid/plane.usd",
        prim_path=floor_path
    )
    
    # Add walls (simple box colliders)
    wall_config = [
        ("/World/Wall_North", (10, 0, 5), (0, 0, 0), (20, 1, 10)),  # North wall
        ("/Wall_South", (10, 0, -5), (0, 0, 0), (20, 1, 10)),      # South wall
        ("/Wall_East", (5, 0, 10), (0, 0, 0), (10, 1, 20)),         # East wall
        ("/Wall_West", (-5, 0, 10), (0, 0, 0), (10, 1, 20)),        # West wall
    ]
    
    for prim_path, pos, rpy, size in wall_config:
        add_reference_to_stage(
            usd_path="/Isaac/Props/Box.usd",
            prim_path=prim_path
        )
        # Scale and position the wall
        from isaacsim.core.utils.prims import set_prim_xform
        from pxr import UsdGeom, Gf
        
        prim = stage.GetPrimAtPath(prim_path)
        if prim:
            xformable = UsdGeom.Xformable(prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(*pos))
    
    # Add storage racks
    print("  Adding storage racks...")
    for i in range(4):
        rack_path = f"/World/Rack_{i:02d}"
        add_reference_to_stage(
            usd_path="/Isaac/Props/Industrial_Rack/industrial_rack.usd",
            prim_path=rack_path
        )
        if stage.GetPrimAtPath(rack_path):
            xformable = UsdGeom.Xformable(stage.GetPrimAtPath(rack_path))
            xformable.AddTranslateOp().Set(Gf.Vec3d(i * 5 - 7.5, 0, 1.5))
    
    # Add boxes at pickup/dropoff locations
    print("  Adding boxes...")
    pickup_location = "/World/Pickup_A"
    dropoff_location = "/World/Dropoff_A"
    
    add_reference_to_stage(
        usd_path="/Isaac/Props/Box.usd",
        prim_path=pickup_location
    )
    if stage.GetPrimAtPath(pickup_location):
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(pickup_location))
        xformable.AddTranslateOp().Set(Gf.Vec3d(-5.0, 0, 0.5))
    
    add_reference_to_stage(
        usd_path="/Isaac/Props/Box.usd",
        prim_path=dropoff_location
    )
    if stage.GetPrimAtPath(dropoff_location):
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(dropoff_location))
        xformable.AddTranslateOp().Set(Gf.Vec3d(5.0, 0, 0.5))
    
    # ==========================================
    # 2. SPAWN CARTER ROBOT
    # ==========================================
    print("\n[2/5] Spawning Carter robot...")
    
    carter_usd = "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
    
    # Check if the USD exists
    assets_root = get_assets_root_path()
    if assets_root:
        carter_full_path = assets_root + carter_usd
        if os.path.exists(carter_full_path):
            print(f"  Loading Carter from: {carter_full_path}")
            carter_path = "/World/Carter"
            add_reference_to_stage(
                usd_path=carter_full_path,
                prim_path=carter_path
            )
            print("  Carter loaded successfully!")
        else:
            print(f"  WARNING: Carter USD not found at {carter_full_path}")
            # Try alternative location
            carter_path = "/World/Carter"
            add_reference_to_stage(
                usd_path="/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd",
                prim_path=carter_path
            )
            print("  Attempted alternative Carter loading...")
    else:
        print("  WARNING: Could not find assets root path")
        carter_path = "/World/Carter"
        add_reference_to_stage(
            usd_path="/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd",
            prim_path=carter_path
        )
    
    # Position Carter in the warehouse
    if stage.GetPrimAtPath(carter_path):
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(carter_path))
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    
    # ==========================================
    # 3. SET UP NAVIGATION
    # ==========================================
    print("\n[3/5] Setting up navigation...")
    
    # Simple navigation setup - create a NavMesh-like grid
    # In a full implementation, this would use Isaac Sim's navigation graph
    # For now, we'll set up the environment for navigation
    
    # Ensure floor has collider
    floor_prim = stage.GetPrimAtPath(floor_path)
    if floor_prim:
        from isaacsim.physics.collision import CollisionAPI
        if not floor_prim.HasAPI("CollisionAPI"):
            from isaacsim.physics import PhysicsSchema
            collision_api = PhysicsSchema.CollisionAPI.Apply(floor_prim)
            collision_api.CreateCollisionEnabledAttr(True)
    
    # ==========================================
    # 4. SET UP SIMULATION CONTEXT
    # ==========================================
    print("\n[4/5] Initializing simulation...")
    
    simulation_app.update()
    
    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import set_named_xform
    
    # Create world
    world = World(
        stage_units_in_meters=1.0,
        gravity=(0.0, 0.0, -9.81),
        physics_dt=1.0/60.0,
        rendering_dt=1.0/60.0
    )
    
    # ==========================================
    # 5. MAIN SIMULATION LOOP
    # ==========================================
    print("\n[5/5] Starting simulation loop...")
    print("Controls:")
    print("  SPACE - Emergency stop")
    print("  R - Reset robot")
    print("  P - Pause/Resume")
    print("  M - Toggle mode")
    print()
    
    # Play the simulation
    world.play()
    
    # Initialize variables
    mode = args.mode
    teleop_active = False
    emergency_stop = False
    pause = False
    
    # Carter state
    carter_prim = world.get_prim_at_path("/World/Carter/chassis_link")
    carter_pos = [0.0, 0.0, 0.0]
    
    # Target goals
    pickup_pos = [-5.0, 0.0, 0.5]
    dropoff_pos = [5.0, 0.0, 0.5]
    
    frame_count = 0
    max_frames = 600  # 10 seconds at 60 FPS
    
    try:
        while simulation_app.is_running() and frame_count < max_frames:
            if not pause:
                # Step simulation
                world.step()
                
                # Get Carter position
                if carter_prim:
                    carter_xform = carter_prim.GetAttribute("xformOp:translate")
                    if carter_xform:
                        carter_pos = list(carter_xform.Get())
                
                # Simple autonomous logic: navigate toward dropoff
                if mode == "autonomous" and not emergency_stop:
                    # Simple navigation: move toward dropoff if close to pickup
                    carter_x = carter_pos[0]
                    carter_y = carter_pos[1]
                    
                    # Check if near pickup
                    dist_to_pickup = ((carter_x - pickup_pos[0])**2 + (carter_y - pickup_pos[1])**2)**0.5
                    dist_to_dropoff = ((carter_x - dropoff_pos[0])**2 + (carter_y - dropoff_pos[1])**2)**0.5
                    
                    # If near pickup (dist < 2), go to dropoff
                    if dist_to_pickup < 2.0 and not hasattr(world, 'object_picked'):
                        # Navigate to dropoff
                        target_x, target_y = dropoff_pos[0], dropoff_pos[1]
                    elif dist_to_dropoff < 2.0:
                        # Task completed, reset for next
                        print(f"  Task completed! Robot at dropoff.")
                        # Reset positions
                        world.get_prim_at_path("/World/Carter/chassis_link").GetAttribute("xformOp:translate").Set(Gf.Vec3d(*pickup_pos))
                        carter_pos = list(pickup_pos)
                    else:
                        # Simple navigation toward dropoff
                        if mode == "autonomous":
                            # Move toward dropoff
                            if carter_x < dropoff_pos[0]:
                                carter_x += 0.1
                            elif carter_x > dropoff_pos[0]:
                                carter_x -= 0.1
                            
                            # Move toward y
                            if carter_y < dropoff_pos[1]:
                                carter_y += 0.05
                            elif carter_y > dropoff_pos[1]:
                                carter_y -= 0.05
                            
                            # Set new position
                            if carter_prim:
                                carter_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(carter_x, carter_y, 0.0))
                
                # Print status
                if frame_count % 30 == 0:  # Every half second at 60 FPS
                    dist_to_dropoff = ((carter_pos[0] - dropoff_pos[0])**2 + (carter_pos[1] - dropoff_pos[1])**2)**0.5
                    print(f"  Frame {frame_count}: Mode={mode}, Position=({carter_pos[0]:.2f}, {carter_pos[1]:.2f}), "
                          f"Dist to dropoff={dist_to_dropoff:.2f}, Emergency Stop={emergency_stop}")
                
                frame_count += 1
            
            # Check for input/events
            # Note: In a real implementation, we'd use Isaac Sim's input API
            # For now, just continue the loop
            
            simulation_app.update()
    
    except KeyboardInterrupt:
        print("\n  Simulation interrupted by user")
    finally:
        # Cleanup
        print("\n  Cleaning up...")
        world.stop()
        simulation_app.close()
        print("  Simulation complete.")


if __name__ == "__main__":
    main()