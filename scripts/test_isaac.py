#!/usr/bin/env python3
"""
MINIMAL TEST: Verify Isaac Sim 5.1 can start and run.

This test MUST use the Isaac Sim 5.1 standalone launcher correctly.
"""

import sys
import os

# CRITICAL: Run through Isaac Sim 5.1 standalone launcher
# The python.sh script sets up the correct PYTHONPATH and environment

# Method 1: Run via the standalone python.sh
# This must be executed as: /path/to/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh -c "..."

# For direct testing, we'll use the approach that worked earlier:
print("Testing Isaac Sim 5.1 startup...")

# Import isaacsim after ensuring environment is set up
try:
    from isaacsim import SimulationApp
    print("✓ SimulationApp imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SimulationApp: {e}")
    print("\nThis usually means Isaac Sim is not properly initialized.")
    print("Try running through the standalone launcher:")
    print("  /home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh -c \"import isaacsim; print('OK')\"")
    sys.exit(1)

# Configure simulation app - optimized for RTX 4050 6 GB
sim_config = {
    "renderer": "iray",  # Use iray instead of RaytracedLighting to save VRAM
    "headless": False,
    "window_width": 1280,  # Reduced from 1920 to save VRAM
    "window_height": 720,  # Reduced from 1080 to save VRAM
    "limit_sampling_buffer_size": True,
    "depth_value_enabled": False,
}

print(f"Simulation config: {sim_config}")

# Launch SimulationApp
try:
    simulation_app = SimulationApp(sim_config)
    print("✓ SimulationApp launched successfully")
except Exception as e:
    print(f"✗ Failed to launch SimulationApp: {e}")
    sys.exit(1)

# Now import other Isaac Sim APIs
try:
    from isaacsim.core.api import World
    print("✓ World API imported successfully")
except Exception as e:
    print(f"✗ Failed to import World: {e}")
    simulation_app.close()
    sys.exit(1)

try:
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    print("✓ Stage utilities imported successfully")
except Exception as e:
    print(f"✗ Failed to import stage utilities: {e}")
    simulation_app.close()
    sys.exit(1)

try:
    from isaacsim.storage.native import get_assets_root_path
    print("✓ Assets root path imported successfully")
except Exception as e:
    print(f"✗ Failed to import assets root path: {e}")
    simulation_app.close()
    sys.exit(1)

# Create a new stage
try:
    stage = get_current_stage()
    print("✓ Stage created/getting current stage successfully")
    
    # Add a reference to a simple USD (floor plane)
    # Check if assets root path is available
    assets_root = get_assets_root_path()
    if assets_root:
        print(f"Assets root: {assets_root}")
        
        # Add a floor plane
        floor_path = "/World/Floor"
        add_reference_to_stage(
            usd_path=assets_root + "/Isaac/Props/Grid/plane.usd",
            prim_path=floor_path
        )
        print(f"✓ Floor plane added at {floor_path}")
    else:
        print("⚠ Warning: Could not find assets root path")
        
except Exception as e:
    print(f"✗ Failed to create stage: {e}")
    import traceback
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

# Play the simulation
try:
    print("\n--- Starting simulation ---")
    simulation_app.update()
    
    # Step the simulation a few times
    print("Stepping simulation 5 times...")
    for i in range(5):
        simulation_app.update()
        print(f"  Step {i+1}: OK")
    
    print("\n✓ Minimal test PASSED - Isaac Sim 5.1 is working!")
    
except Exception as e:
    print(f"✗ Failed during simulation: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up
    print("\n--- Closing simulation ---")
    simulation_app.close()
    print("✓ Simulation closed cleanly")

sys.exit(0)