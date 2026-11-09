#!/usr/bin/env python3
"""Warehouse Robotics - Isaac Sim + ROS 2 multi-robot simulation.

Scene
-----
20 m x 14 m warehouse with two rack rows, four coloured work stations, a
Franka Panda pick-and-place cell and a Unitree H1 humanoid patrol route.

Robots
------
* 3 x Nova Carter AMRs, each with its own 2-D LiDAR and ROS 2 namespace.
* 1 x Franka Panda, joint-state controlled from ROS 2.
* 1 x Unitree H1 humanoid, velocity controlled through Isaac's flat-terrain
  walking policy.

Everything is driven from ROS 2.  Start the autonomy stack separately:

    scripts/run_demo.sh                 # visual simulation + autonomy
    scripts/run_demo.sh --headless      # headless
    scripts/run_sim.sh                  # simulation only
    scripts/run_autonomy.sh             # ROS 2 stack only

The simulator never imports host ROS 2 Python; see warehouse_ros_env.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import warehouse_ros_env  # noqa: E402  (must precede isaacsim)


def parse_args():
    p = argparse.ArgumentParser(description="Isaac Sim warehouse multi-robot demo")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "warehouse.yaml"))
    p.add_argument("--headless", action="store_true", help="no GUI (recommended for tests)")
    p.add_argument("--no-ros", action="store_true", help="build the scene without ROS 2 topics")
    p.add_argument("--no-franka", action="store_true")
    p.add_argument("--no-humanoid", action="store_true")
    p.add_argument("--no-industrial", action="store_true",
                   help="skip the UR10 conveyor palletising cell")
    p.add_argument("--nvidia-assets", action="store_true", dest="nvidia_assets",
                   help="enable NVIDIA SimReady set dressing (default in visual mode)")
    p.add_argument("--no-nvidia-assets", action="store_false", dest="nvidia_assets",
                   help="skip NVIDIA set dressing (faster headless runs)")
    p.add_argument("--domain-id", type=int, default=None)
    p.add_argument("--duration", type=float, default=0.0,
                   help="stop after N simulated seconds (0 = run forever)")
    p.add_argument("--no-realtime", action="store_false", dest="realtime",
                   help="run as fast as possible instead of pacing to wall time")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--camera", choices=("overview", "ur10", "h1"), default="overview",
                   help="visual camera preset")
    p.add_argument("--no-hud", action="store_true", help="hide the status HUD")
    p.add_argument("--people", action="store_true", dest="people",
                   help="enable NVIDIA People dynamic crew + NavMesh (visual mode)")
    p.add_argument("--no-people", action="store_false", dest="people",
                   help="disable the NVIDIA People dynamic crew + NavMesh")
    p.set_defaults(realtime=None, nvidia_assets=None, people=None)
    return p.parse_args()


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    world_cfg = cfg["world"]
    use_nvidia = (not args.headless) if args.nvidia_assets is None else bool(args.nvidia_assets)
    realtime = world_cfg.get("realtime", True) if args.realtime is None else args.realtime
    domain_id = args.domain_id
    if domain_id is None:
        domain_id = int(world_cfg.get("ros_domain_id", 0))
    use_people = (not args.headless) if args.people is None else bool(args.people)

    warehouse_ros_env.prepare(domain_id=domain_id)

    width = args.width or (640 if args.headless else 1600)
    height = args.height or (360 if args.headless else 900)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": bool(args.headless),
            "renderer": "RaytracedLighting",
            "width": width,
            "height": height,
            "window_width": width,
            "window_height": height,
            "limit_sampling_buffer_size": True,
            "anti_aliasing": 2 if not args.headless else 0,
        }
    )

    # -- imports that require a running Kit app ---------------------------
    import omni.usd
    from pxr import Usd

    from isaacsim.core.api import World
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.storage.native import get_assets_root_path

    from warehouse_robots import AMR, Forklift, FrankaCell, HumanoidWalker
    from warehouse_ros2 import RosBridge
    from warehouse_scene import Materials, build_scene
    from warehouse_nvidia import (
        NvidiaAssetLayer,
        build_nvidia_site,
    )
    from warehouse_process import ConveyorTotes, IndustrialCell

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Could not resolve the Isaac Sim asset root")

    if use_people:
        # The People / animation / navigation extensions must be enabled before
        # the stage is opened, and the real stage has to go through the USD
        # context so omni.anim.graph.get_character() can find the characters.
        from warehouse_people import enable_people_extensions, open_people_stage

        enable_people_extensions(app)
        stage = open_people_stage(app, PROJECT_ROOT)
    else:
        stage = get_current_stage()
    stage.SetTimeCodesPerSecond(200.0)

    physics_dt = float(world_cfg["physics_dt"])
    rendering_dt = float(world_cfg["rendering_dt"])
    render_steps = max(1, int(round(rendering_dt / physics_dt)))

    print("=" * 72)
    print(" WAREHOUSE ROBOTICS - Isaac Sim 5.1 / ROS 2 Humble")
    print(f" config : {args.config}")
    print(f" mode   : {'HEADLESS' if args.headless else 'VISUAL'}   "
          f"ROS={'off' if args.no_ros else 'on'}   domain={domain_id}")
    print(f" robots : {len(cfg['amrs'])} AMR, "
          f"Franka={'on' if cfg['franka'].get('enabled') and not args.no_franka else 'off'}, "
          f"Humanoid={'on' if cfg['humanoid'].get('enabled') and not args.no_humanoid else 'off'}, "
          f"People={'on' if use_people else 'off'}")
    print("=" * 72, flush=True)

    # -- extensions ------------------------------------------------------
    enable_extension("isaacsim.robot.policy.examples")
    enable_extension("isaacsim.sensors.physx")
    for _ in range(3):
        app.update()

    # -- world (physics context) ----------------------------------------
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=physics_dt,
        rendering_dt=rendering_dt,
    )

    materials = Materials(stage)
    # Headless runs do not render, so use the same light-weight collision-only
    # representation that the NVIDIA visual mode uses.  This keeps the tested
    # physics model identical with and without the SimReady meshes.
    procedural_visuals = (not use_nvidia) and (not args.headless)
    build_scene(stage, cfg, materials, procedural_visuals=procedural_visuals)
    print(
        f"[scene] warehouse built "
        f"({'NVIDIA SimReady set dressing' if use_nvidia else 'procedural visuals'})",
        flush=True,
    )

    layer = None
    if use_nvidia:
        layer = NvidiaAssetLayer(stage, assets_root, app)
        build_nvidia_site(layer, cfg, stage)

    if use_people and not use_nvidia:
        print("[people] disabled: --people requires NVIDIA SimReady set dressing "
              "(use --nvidia-assets) so the NavMesh can resolve obstacles",
              flush=True)
        use_people = False

    amrs = []
    for spec in cfg["amrs"]:
        amr = AMR(spec["name"], spec, cfg, assets_root, visual=not args.headless)
        amr.create()
        app.update()
        amr.create_lidar(app)
        if not args.headless:
            amr.create_payload(layer, materials, visual=True)
        amrs.append(amr)
        print(f"[scene] {amr.name} spawned at "
              f"({spec['spawn']['x']:.1f}, {spec['spawn']['y']:.1f})", flush=True)

    franka = None
    if cfg["franka"].get("enabled") and not args.no_franka:
        franka = FrankaCell(cfg["franka"], cfg, assets_root, materials,
                            visual=not args.headless)
        franka.create()
        app.update()
        print("[scene] Franka Panda cell built", flush=True)

    humanoid = None
    if cfg["humanoid"].get("enabled") and not args.no_humanoid:
        humanoid = HumanoidWalker(cfg["humanoid"], visual=not args.headless)
        humanoid.create()
        for _ in range(3):
            app.update()
        if layer is not None:
            humanoid.create_payload(layer)
        print("[scene] Unitree H1 humanoid loaded", flush=True)

    forklift = None
    fork_cfg = cfg["scene"].get("forklift", {})
    if fork_cfg.get("enabled", True):
        forklift = Forklift(fork_cfg, visual=use_nvidia)
        # Author the visual root before the asset layer measures and fits the
        # NVIDIA forklift mesh.  The kinematic proxy is added after
        # World.reset() by Forklift.activate().
        forklift.create(layer)
        print(
            f"[scene] dynamic forklift configured on x={forklift.lane_x:.1f} "
            f"({forklift.y_min:.1f}..{forklift.y_max:.1f} m)",
            flush=True,
        )

    process = None
    belts = None
    ind_cfg = cfg["scene"].get("industrial", {})
    if ind_cfg.get("enabled", True) and not args.no_industrial:
        process = IndustrialCell(
            ind_cfg, materials, assets_root, layer, visual=not args.headless
        )
        process.create()
        print(
            f"[scene] UR10 palletising cell queued at "
            f"({ind_cfg['base']['x']:.1f}, {ind_cfg['base']['y']:.1f})",
            flush=True,
        )
    if not args.headless:
        belts = ConveyorTotes(cfg, materials, visual=True)
        belts.create(layer)

    if layer is not None:
        layer.apply()
        if humanoid is not None:
            humanoid.after_assets()
        if process is not None:
            process.after_assets()
        print(
            f"[scene] NVIDIA assets applied: {len(layer.applied)} "
            f"({dict(layer.stats)})",
            flush=True,
        )

    # -- NVIDIA People dynamic crew + NavMesh ---------------------------
    people = None
    if use_people:
        from warehouse_people import setup_people

        people_roots = []
        for amr in amrs:
            people_roots.append(amr.robot_path)
        if franka is not None:
            people_roots.append(franka.robot_path)
        if humanoid is not None:
            people_roots.append(humanoid.prim_path)
        if forklift is not None:
            people_roots.append(forklift.root_path)
            people_roots.append(forklift.body_path)
        if process is not None:
            people_roots.append(process.robot_path)
            people_roots.append(process.cell_path)
        people_roots.extend(("/World/Characters/Biped_Setup", "/World/Process"))

        people = setup_people(
            app,
            stage,
            cfg,
            assets_root,
            dynamic_roots=people_roots,
            materials=materials,
            project_root=PROJECT_ROOT,
            log=print,
        )
        if people is None:
            print("[people] NOT READY: dynamic crew skipped", flush=True)

    # -- physics initialisation -----------------------------------------
    world.reset()
    for _ in range(5):
        app.update()

    for amr in amrs:
        amr.initialize()
    if franka is not None:
        franka.initialize()
    if humanoid is not None:
        humanoid.initialize()
    if process is not None:
        process.initialize()
    if forklift is not None:
        forklift.activate()
    print("[scene] articulations initialised", flush=True)

    # -- ROS 2 bridge ----------------------------------------------------
    bridge = RosBridge(app, domain_id=domain_id)
    if not args.no_ros:
        if bridge.start():
            for amr in amrs:
                bridge.add_amr(
                    amr.name, amr.robot_path, amr.chassis_path, amr.lidar_path,
                    amr.laser_frame, amr.spec.get("lidar_mount", [-0.06, 0.0, 0.38]),
                )
            if franka is not None:
                bridge.add_franka(franka.robot_path)
            if humanoid is not None:
                bridge.add_humanoid(humanoid.name, humanoid.prim_path + "/pelvis")
            if forklift is not None:
                bridge.add_forklift(forklift.name, forklift.body_path)
            if process is not None:
                bridge.add_joint_states(
                    "ur10", process.articulation_path, "ur10/joint_states"
                )
            # Logical payload state from the fleet manager / dashboard and
            # the industrial-cell command channel.  The sim only uses these
            # when a publisher is actually present.
            for amr in amrs:
                bridge.add_twist_input(
                    "load_" + amr.name, f"{amr.name}/load_state"
                )
            bridge.add_twist_input("ur10_cmd", "ur10/command")
            print(f"[ros] bridge ready: {len(bridge.graphs)} action graphs", flush=True)
            if bridge.errors:
                print("[ros] warnings: " + " | ".join(bridge.errors), flush=True)
        else:
            print("[ros] bridge NOT available; simulation continues without ROS",
                  flush=True)
            if bridge.errors:
                print("[ros] " + " | ".join(bridge.errors), flush=True)

    world.play()
    for _ in range(5):
        app.update()

    if bridge.enabled and process is not None:
        bridge.add_string_publisher("ur10_status", "ur10/status")
        bridge.add_string_publisher(
            "warehouse_process", "warehouse/process_status"
        )
        if bridge.errors:
            print("[ros] telemetry warnings: " + " | ".join(bridge.errors),
                  flush=True)

    # -- camera / HUD ----------------------------------------------------
    if not args.headless:
        from isaacsim.core.utils.viewports import set_camera_view

        # High south-east isometric: the whole 20x14 m hall is legible and
        # the camera looks over the east rack bank into the forklift aisle.
        if args.camera == "ur10":
            set_camera_view(eye=[7.6, -1.2, 3.4], target=[4.1, 2.9, 1.1])
        elif args.camera == "h1":
            set_camera_view(eye=[1.8, -1.8, 3.2], target=[-0.8, 4.2, 1.1])
        else:
            set_camera_view(eye=[15.0, -16.0, 14.5], target=[0.0, 0.2, 0.8])
    hud = (
        _build_hud(args, amrs, franka, humanoid, forklift, process, people)
        if not args.headless and not args.no_hud else None
    )

    # -- main loop -------------------------------------------------------
    dt = physics_dt
    step_count = 0
    t0_wall = time.time()
    t0_sim = world.current_time
    last_status = 0.0
    last_hud = 0.0
    last_process_status = 0.0
    load_last = {amr.name: None for amr in amrs}
    load_seen = {amr.name: False for amr in amrs}
    process_ros_ready = False
    stations_cfg = cfg["scene"]["stations"]
    parking_cfg = cfg["scene"].get("parking", [])

    print("[sim] running. Ctrl+C to stop.", flush=True)
    try:
        while app.is_running():
            # 1. Poll ROS commands (graphs run during app.update()).
            for amr in amrs:
                cmd = bridge.twist(amr.name)
                amr.apply_twist(float(cmd[0]), float(cmd[5]))
            if franka is not None and not args.no_ros:
                franka.apply_command(bridge.franka_command())
            if humanoid is not None and not args.no_ros:
                cmd = bridge.twist(humanoid.name)
                humanoid.set_command(float(cmd[0]), float(cmd[1]), float(cmd[5]))
            if process is not None:
                seen_cmd, cmd = bridge.input_twist("ur10_cmd")
                if seen_cmd:
                    process.apply_command(float(cmd[0]), float(cmd[1]))

            # Logical payload transitions: show the tote only between the
            # pickup and dropoff dwells, and count a real staging delivery
            # when a loaded AMR unloads on the staging pad.
            for amr in amrs:
                seen, value = bridge.load_state(amr.name)
                load_seen[amr.name] = seen
                if not seen:
                    continue
                loaded = value > 0.5
                if not process_ros_ready and process is not None:
                    process_ros_ready = True
                    process.set_ros_ready(True)
                if (load_last[amr.name] is True and not loaded
                        and process is not None):
                    pos, _ = amr.pose()
                    staging = next(
                        (s for s in stations_cfg if s["name"] == "staging"),
                        None,
                    )
                    if staging is not None and np.hypot(
                        pos[0] - float(staging["x"]),
                        pos[1] - float(staging["y"]),
                    ) < 1.65:
                        process.notify_delivery()
                load_last[amr.name] = loaded

            # 2. Apply articulation targets, then step physics.
            if forklift is not None:
                amr_xy = []
                for amr in amrs:
                    p = amr.pose()[0]
                    amr_xy.append((float(p[0]), float(p[1])))
                forklift.update(
                    dt,
                    world.current_time - t0_sim,
                    amr_xy,
                )
            if humanoid is not None:
                humanoid.update(dt)
            if franka is not None:
                franka.update(dt)
            sim_t = world.current_time - t0_sim
            if process is not None:
                process.pre_step(dt, sim_t)
            world.step(render=False)
            step_count += 1
            if process is not None:
                process.post_step(dt, sim_t)
            if belts is not None and step_count % 2 == 0:
                belts.update(dt, sim_t)
            if not args.headless:
                for amr in amrs:
                    amr.update_visual(dt, stations_cfg, parking_cfg)
                    if load_seen.get(amr.name):
                        amr.set_payload_visible(
                            bool(load_last.get(amr.name))
                        )

            # 3. Render / execute OmniGraph publishers at the render rate.
            if step_count % render_steps == 0:
                app.update()

            now = time.time()
            if (bridge.enabled and process is not None
                    and now - last_process_status > 0.5):
                last_process_status = now
                ur10_status = process.status_dict()
                belt_status = (
                    belts.status_dict() if belts is not None else {
                        "running": True,
                        "speed": 0.16,
                        "belts": len(cfg["scene"].get("conveyors", [])),
                        "loads": 0,
                        "render_only": bool(args.headless),
                    }
                )
                bridge.publish_string(
                    "ur10_status", json.dumps(ur10_status)
                )
                bridge.publish_string(
                    "warehouse_process",
                    json.dumps({
                        "time": round(world.current_time, 3),
                        "ur10": ur10_status,
                        "conveyors": belt_status,
                        "ros_ready": bool(process_ros_ready),
                    }),
                )

            # 4. Optional real-time pacing.
            if realtime:
                target_wall = t0_wall + (world.current_time - t0_sim)
                delay = target_wall - time.time()
                if delay > 0:
                    time.sleep(min(delay, 0.05))

            if now - last_status > 2.0:
                last_status = now
                _print_status(
                    amrs, franka, humanoid, forklift, process, world, bridge, args, people
                )
            if hud is not None and now - last_hud > 0.25:
                last_hud = now
                _update_hud(
                    hud, amrs, franka, humanoid, forklift, process,
                    world, bridge, args, people,
                )

            if args.duration > 0.0 and world.current_time - t0_sim >= args.duration:
                print(f"[sim] duration {args.duration:.1f}s reached", flush=True)
                break
    except KeyboardInterrupt:
        print("\n[sim] interrupted", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        print("[sim] shutting down", flush=True)
        try:
            world.stop()
        except Exception:
            pass
        app.close()


def _print_status(amrs, franka, humanoid, forklift, process, world, bridge, args, people=None):
    parts = [f"t={world.current_time:7.2f}s"]
    if people is not None:
        parts.append(
            f"people={people['worker_count']} agents "
            f"navmesh={'ok' if people['navmesh'] is not None else 'FAIL'}"
        )
        people_stage = people.get("stage")
        if people_stage is not None:
            import omni.anim.graph.core as _ag
            import carb

            for name in people.get("workers") or []:
                try:
                    prim = people_stage.GetPrimAtPath(f"/World/People/{name}/Asset")
                    if prim is None or not prim.IsValid():
                        parts.append(f"{name}=? no prim")
                        continue
                    character = _ag.get_character(str(prim.GetPath()))
                    if character is None:
                        parts.append(f"{name}=? no character")
                        continue
                    pos = carb.Float3()
                    rot = carb.Float4()
                    character.get_world_transform(pos, rot)
                    parts.append(
                        f"{name}=({pos.x:+.2f},{pos.y:+.2f},{pos.z:+.2f})"
                    )
                except Exception:
                    parts.append(f"{name}=?")
    for amr in amrs:
        pos, quat = amr.pose()
        from warehouse_robots import quat_yaw

        parts.append(
            f"{amr.name}=({pos[0]:5.2f},{pos[1]:5.2f},{quat_yaw(quat):+5.2f})"
        )
    if franka is not None:
        parts.append(f"franka={'holding ' + str(franka.attached) if franka.attached else 'free'}")
    if humanoid is not None:
        pos, _ = humanoid.pose()
        parts.append(
            f"H1=({pos[0]:5.2f},{pos[1]:5.2f}) "
            f"{'carrying' if humanoid.carrying else 'empty'} loads={humanoid.loads}"
        )
    if forklift is not None:
        fpos, _ = forklift.pose()
        parts.append(f"forklift=({fpos[0]:5.2f},{fpos[1]:5.2f})")
    if process is not None:
        parts.append(f"ur10={process.status()}")
    parts.append(f"ros_graphs={len(bridge.graphs)}")
    print("[sim] " + "  ".join(parts), flush=True)


def _build_hud(args, amrs, franka, humanoid, forklift, process=None, people=None):
    try:
        import omni.ui as ui
    except Exception:
        return None
    try:
        window = ui.Window(
            "Warehouse Fleet", width=470, height=370,
            position_x=18, position_y=64,
        )
    except TypeError:
        window = ui.Window("Warehouse Fleet", width=470, height=370)
        try:
            window.position_x = 18
            window.position_y = 64
        except Exception:
            pass
    labels = {}
    with window.frame:
        with ui.VStack(spacing=4):
            ui.Label("INDUSTRIAL WAREHOUSE | NVIDIA ASSETS",
                     style={"font_size": 18, "color": 0xFF7FD4FF})
            ui.Label(
                f"{len(amrs)} Nova Carter AMRs | Franka | UR10 | H1 | Forklift",
                style={"font_size": 12, "color": 0xFFB0B0B0},
            )
            ui.Spacer(height=6)
            for amr in amrs:
                labels[amr.name] = ui.Label("", style={"font_size": 13})
            if franka is not None:
                labels["franka"] = ui.Label("", style={"font_size": 13})
            if process is not None:
                labels["ur10"] = ui.Label("", style={"font_size": 13})
            if humanoid is not None:
                labels["humanoid"] = ui.Label("", style={"font_size": 13})
            if forklift is not None:
                labels["forklift"] = ui.Label("", style={"font_size": 13})
            if people is not None:
                labels["people"] = ui.Label("", style={"font_size": 12, "color": 0xFF9AD29A})
            ui.Spacer(height=6)
            labels["ros"] = ui.Label("", style={"font_size": 12, "color": 0xFF9AD29A})
            labels["hint"] = ui.Label(
                "conveyor -> UR10 palletising | H1 tote transfer | forklift: dynamic obstacle",
                style={"font_size": 11, "color": 0xFF909090},
            )
    # Keep the window object alive.
    _build_hud.window = window
    return labels


def _update_hud(labels, amrs, franka, humanoid, forklift, process,
                world, bridge, args, people=None):
    from warehouse_robots import quat_yaw

    for amr in amrs:
        pos, quat = amr.pose()
        labels[amr.name].text = (
            f"{amr.name.upper():5s}  pos ({pos[0]:+6.2f}, {pos[1]:+6.2f})  "
            f"yaw {quat_yaw(quat):+5.2f} rad"
        )
    if franka is not None:
        labels["franka"].text = (
            "FRANKA  holding " + (franka.attached or "-")
            + f"  gripper {franka.gripper_target:.3f} m"
        )
    if process is not None:
        labels["ur10"].text = f"UR10    {process.status()}"
    if humanoid is not None:
        pos, _ = humanoid.pose()
        labels["humanoid"].text = (
            f"H1      pos ({pos[0]:+6.2f}, {pos[1]:+6.2f})  "
            f"{'CARRYING' if humanoid.carrying else 'empty':8s} loads {humanoid.loads}"
        )
    if forklift is not None:
        fpos, _ = forklift.pose()
        labels["forklift"].text = (
            f"FORKLIFT pos ({fpos[0]:+6.2f}, {fpos[1]:+6.2f})  "
            f"dynamic obstacle"
        )
    if people is not None and "people" in labels:
        labels["people"].text = (
            f"PEOPLE {people['worker_count']} agents | navmesh "
            f"{'ok' if people['navmesh'] is not None else 'FAIL'}"
        )
    labels["ros"].text = (
        f"t = {world.current_time:8.2f} s    ROS graphs = {len(bridge.graphs)}"
    )


if __name__ == "__main__":
    main()
