#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Import the LeKiwi URDF into Isaac Sim 5.1 and save a USD stage.

Run it through the wrapper so a ROS 2 runtime is always reachable::

    ./scripts/lekiwi_isaac.sh scripts/import_lekIwi_urdf.py
    ./scripts/lekiwi_isaac.sh scripts/import_lekIwi_urdf.py --gui

What this script does
---------------------
1. parses ``/home/arun/Desktop/lekiwi.urdf`` (the original is never modified)
   with the official ``isaacsim.asset.importer.urdf`` extension;
2. moves the articulation to ``/World/LeKiwi`` and lifts it so the three omni
   wheels rest exactly on the z = 0 ground plane;
3. configures joint drives:
     * wheels -> velocity drive (stiffness 0 + damping + force cap)
     * arm    -> position drive (stiffness + damping + force cap)
   which is what makes ``isaacsim.core.nodes.IsaacArticulationController``
   respond to ``/lekIwi/wheel_command`` (velocity) and ``/lekIwi/arm_command``
   (position);
4. writes explicit ``physics:lowerLimit`` / ``physics:upperLimit`` on all nine
   moving joints, because the source URDF declares every joint as
   ``continuous`` with no ``<limit>`` element;
5. adds a physics scene + ground plane and saves the stage.

The result is opened by ``create_lekIwi_action_graph.py``.
"""

import argparse
import math
import os
import sys

# --- ROS 2 / Python path hygiene MUST happen before Kit starts ---------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lekiwi_ros_env  # noqa: E402
import lekiwi_model as model  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--urdf", default=model.SOURCE_URDF, help="input URDF (never modified)")
    p.add_argument("--out", default=model.IMPORTED_USD, help="USD stage to write")
    p.add_argument("--gui", action="store_true", help="open a window instead of running headless")
    p.add_argument("--no-ground", action="store_true", help="skip the ground plane")
    p.add_argument(
        "--no-lift",
        action="store_true",
        help="leave the robot root at z=0 instead of raising it so the wheels touch z=0",
    )
    p.add_argument("--wheel-damping", type=float, default=model.WHEEL_DRIVE["damping"])
    p.add_argument("--wheel-max-force", type=float, default=model.WHEEL_DRIVE["max_force"])
    p.add_argument("--arm-stiffness", type=float, default=model.ARM_DRIVE["stiffness"])
    p.add_argument("--arm-damping", type=float, default=model.ARM_DRIVE["damping"])
    p.add_argument("--arm-max-force", type=float, default=model.ARM_DRIVE["max_force"])
    return p.parse_args(argv)


ARGS = parse_args()
ENV_REPORT = lekiwi_ros_env.prepare(ros_domain_id=os.environ.get("ROS_DOMAIN_ID", 42))

from isaacsim import SimulationApp  # noqa: E402  (must follow env preparation)

APP = SimulationApp({"headless": not ARGS.gui, "renderer": "RaytracedLighting"})


def log(msg):
    print("[import_lekIwi] {}".format(msg), flush=True)


def die(msg):
    log("FAIL: " + msg)
    APP.close()
    sys.exit(1)


def main():
    log("ROS environment: {}".format(
        {k: v for k, v in ENV_REPORT.items() if k != "dropped_pythonpath"}))
    if ENV_REPORT["dropped_pythonpath"]:
        log("  stripped host ROS 2 from PYTHONPATH: {}".format(
            ENV_REPORT["dropped_pythonpath"]))

    if not os.path.isfile(ARGS.urdf):
        die("URDF not found: {}".format(ARGS.urdf))
    if not os.path.isdir(model.MESHES_DIR):
        die("mesh directory not found: {}".format(model.MESHES_DIR))

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")
    APP.update()

    import omni.kit.commands
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics

    # ------------------------------------------------------------------ stage
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if list(stage.GetPseudoRoot().GetChildren()):
        task = ctx.new_stage()
        for _ in range(200):
            APP.update()
            if task is None or not hasattr(task, "is_done") or task.is_done():
                break
        stage = ctx.get_stage()
        APP.update()

    # --------------------------------------------------------------- import
    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        die("URDFCreateImportConfig failed")

    # merge_fixed_joints=False keeps all 45 link frames so the TF tree Isaac
    # publishes matches the URDF frame-for-frame.
    cfg.merge_fixed_joints = False
    cfg.import_inertia_tensor = True
    cfg.fix_base = False                 # the base must drive around
    cfg.make_default_prim = False        # we author /World ourselves
    cfg.create_physics_scene = True
    cfg.self_collision = False
    cfg.distance_scale = 1.0
    cfg.up_vector = "Z"
    cfg.parse_mimic = True
    cfg.override_joint_dynamics = False  # keep URDF joint dynamics
    cfg.convex_decomp = False            # convex hulls: much faster to build
    cfg.collision_from_visuals = False   # the URDF already has collision meshes

    log("importing {} ...".format(ARGS.urdf))
    result = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=ARGS.urdf, import_config=cfg
    )
    APP.update()

    imported_path = None
    if isinstance(result, (tuple, list)):
        imported_path = next((r for r in result if isinstance(r, str)), None)
    elif isinstance(result, str):
        imported_path = result
    log("importer returned: {!r}".format(result))

    # --------------------------------------------------- locate articulation
    robot_path = None
    for candidate in filter(None, [imported_path, "/LeKiwi", "/World/LeKiwi"]):
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            robot_path = candidate
            break

    if robot_path is None:
        from pxr import Usd

        roots = [
            p.GetPath()
            for p in Usd.PrimRange(stage.GetPseudoRoot())
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if not roots:
            die("no articulation root found on the imported stage")
        robot_path = str(sorted(roots, key=lambda s: s.count("/"))[0])
    log("articulation root imported at: {}".format(robot_path))

    # --------------------------------------------------- /World + final path
    if not stage.GetPrimAtPath("/World"):
        UsdGeom.Xform.Define(stage, "/World")

    if robot_path != model.ROBOT_PRIM:
        omni.kit.commands.execute(
            "MovePrimCommand",
            path_from=robot_path,
            path_to=model.ROBOT_PRIM,
            keep_world_transform=True,
        )
        APP.update()
        robot_path = model.ROBOT_PRIM
    log("articulation root at: {}".format(robot_path))

    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        die("robot prim {} disappeared after the move".format(robot_path))

    joints_root = "{}/joints".format(robot_path)

    # ------------------------------------------------------------- placement
    lift = 0.0 if ARGS.no_lift else model.wheel_lift(ARGS.urdf)
    from isaacsim.core.utils.xforms import reset_and_set_xform_ops

    reset_and_set_xform_ops(
        robot_prim, Gf.Vec3d(0.0, 0.0, lift), Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    )
    log("root placed at z = {:.6f} m (wheel radius {} m - wheel centre z {} m)"
        .format(lift, model.WHEEL_RADIUS, model.WHEEL_CENTER_Z))

    # ---------------------------------------------------------------- drives
    from lekiwi_common import configure_drives, configure_limits, report_joints

    joint_paths = {}
    for name in model.ALL_JOINTS:
        p = stage.GetPrimAtPath("{}/{}".format(joints_root, name))
        if not p or not p.IsValid():
            die("joint prim not found: {}/{}".format(joints_root, name))
        joint_paths[name] = p

    for name in model.WHEEL_JOINTS:
        configure_drives(
            joint_paths[name], "velocity",
            target=0.0,
            stiffness=0.0,
            damping=ARGS.wheel_damping,
            max_force=ARGS.wheel_max_force,
        )
    for name in model.ARM_JOINTS:
        configure_drives(
            joint_paths[name], "position",
            target=0.0,
            stiffness=ARGS.arm_stiffness,
            damping=ARGS.arm_damping,
            max_force=ARGS.arm_max_force,
        )
    configure_limits(joint_paths, model.ARM_LIMIT_DEG, model.WHEEL_LIMIT_DEG)
    report_joints(stage, joint_paths)

    # ------------------------------------------------------------ physics
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path(model.PHYSICS_SCENE_PRIM))
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

    if not ARGS.no_ground and not stage.GetPrimAtPath(model.GROUND_PRIM):
        PhysicsSchemaTools.addGroundPlane(
            stage, model.GROUND_PRIM, "Z", 50.0,
            Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f([0.4, 0.4, 0.4]),
        )
        log("ground plane at z = 0 ({})".format(model.GROUND_PRIM))

    # -------------------------------------------------------------- save
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    out_dir = os.path.dirname(ARGS.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ok = stage.GetRootLayer().Export(ARGS.out)
    if not ok or not os.path.isfile(ARGS.out):
        die("could not export {}".format(ARGS.out))
    size = os.path.getsize(ARGS.out)
    log("saved {} ({:.1f} KiB)".format(ARGS.out, size / 1024.0))
    log("next: ./scripts/lekiwi_isaac.sh scripts/create_lekIwi_action_graph.py")
    log("PASS")


def run():
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        import traceback

        traceback.print_exc()
        log("FAIL: {}".format(exc))
        APP.close()
        sys.exit(1)
    APP.close()


if __name__ == "__main__":
    run()
