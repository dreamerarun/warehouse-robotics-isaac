#!/usr/bin/env python3
"""NVIDIA People dynamic characters + NavMesh integration.

The warehouse demo runs its crew through Isaac Sim's own human motion
package: ``omni.anim.people`` (behavior / command layer), ``omni.anim.graph``
(animation graph) and ``omni.anim.navigation`` (NavMesh baking + path
planning).

The flow in ``warehouse_sim.py`` is:

1. enable the People extensions,
2. open ``assets/warehouse_start.usda`` through the USD context,
3. author the shared ``Biped_Setup`` animation graph,
4. attach a Reallusion 101-joint worker under ``/World/People/WorkerN``,
5. bake a NavMesh over the visible SimReady geometry and start the
   ``GoTo`` / ``Idle`` / ``LookAround`` command loop.

The crew routes themselves live in ``config/warehouse.yaml`` under the
``people`` section; this module only turns them into an
``omni.anim.people`` command file at runtime.
"""

from __future__ import annotations

import os
import tempfile
import time

PEOPLE_EXTENSIONS = (
    "omni.anim.people",
    "omni.anim.navigation.bundle",
    "omni.anim.navigation.core",
    "omni.anim.timeline",
    "omni.anim.graph.bundle",
    "omni.anim.graph.core",
    "omni.anim.graph.ui",
    "omni.anim.retarget.bundle",
    "omni.anim.retarget.core",
    "omni.anim.retarget.ui",
    "omni.kit.scripting",
)


def _log(msg: str):
    print(f"[people] {msg}", flush=True)


def _nav_schema():
    import NavSchema

    return NavSchema


def enable_people_extensions(app) -> None:
    from isaacsim.core.utils.extensions import enable_extension

    for ext in PEOPLE_EXTENSIONS:
        enable_extension(ext)
        for _ in range(2):
            app.update()
    _log("People / animation / navigation extensions enabled")


def _wait_for_stage(app, predicate, timeout: float = 60.0) -> bool:
    from isaacsim.core.utils.stage import is_stage_loading

    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        try:
            done = predicate()
        except Exception:
            done = True
        if done:
            return True
        time.sleep(0.005)
    return False


def open_people_stage(app, project_root) -> object:
    """Open a real template USD stage before the simulation world is built.

    ``isaacsim.usd.get_context().open_stage()`` is required: a stage created
    only through ``get_current_stage()`` makes ``omni.anim.graph.get_character``
    return ``None`` even when a valid character is present.
    """
    import omni.usd
    from isaacsim.core.utils.stage import is_stage_loading

    root = os.fspath(project_root)
    template = os.path.join(root, "assets", "warehouse_start.usda")
    if not os.path.isfile(template):
        raise FileNotFoundError(f"People stage template not found: {template}")

    context = omni.usd.get_context()
    context.open_stage(template)

    def stage_ready():
        try:
            return not is_stage_loading()
        except Exception:
            return True

    if not _wait_for_stage(app, stage_ready, timeout=60.0):
        raise RuntimeError("People template stage did not open cleanly")
    stage = context.get_stage()
    default = stage.GetDefaultPrim()
    default_str = str(default.GetPath().pathString) if default is not None else ""
    if default is None or not default.IsValid() or default_str != "/World":
        stage.SetDefaultPrim("/World")
    stage.SetTimeCodesPerSecond(200.0)
    _log(f"template stage opened: {template}")
    return stage


def _find_prim(stage, prefix: str, type_name: str):
    prim = stage.GetPrimAtPath(prefix)
    if prim is not None and prim.IsValid():
        if prim.GetTypeName() == type_name:
            return prim
        for child in prim.GetChildren():
            found = _find_prim(stage, str(child.GetPath()), type_name)
            if found is not None:
                return found
    return None


def _find_animation_graph(stage) -> object:
    return _find_prim(stage, "/World/Characters/Biped_Setup", "AnimationGraph")


def _collect_descendants(stage, prefix: str):
    prim = stage.GetPrimAtPath(prefix)
    if prim is None or not prim.IsValid():
        return
    yield prim
    for child in prim.GetChildren():
        yield from _collect_descendants(stage, str(child.GetPath()))


def load_animation_graph(stage, cfg, assets_root, app) -> object:
    people_cfg = cfg.get("people") or {}
    rel = str(
        people_cfg.get("animation_graph")
        or "Isaac/People/Characters/Biped_Setup.usd"
    )
    setup_path = str(people_cfg.get("graph_root") or "/World/Characters/Biped_Setup")
    prim = stage.GetPrimAtPath(setup_path)
    if prim is None or not prim.IsValid():
        from isaacsim.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(
            usd_path=str(assets_root) + "/" + rel, prim_path=setup_path
        )
        prim = stage.GetPrimAtPath(setup_path)
        prim.GetAttribute("visibility").Set("invisible")

    graph_prim = _find_animation_graph(stage)
    if graph_prim is None:
        _log("animation graph prim not found under " + setup_path)
    else:
        _log(f"animation graph ready: {graph_prim.GetPath()}")
    return graph_prim


def _apply_graph_and_script(stage, prim, graph_prim, script_path) -> None:
    import omni.kit.commands
    from pxr import Sdf

    paths = [Sdf.Path(prim.GetPath())]
    omni.kit.commands.execute(
        "ApplyAnimationGraphAPICommand",
        paths=paths,
        animation_graph_path=Sdf.Path(graph_prim.GetPath()),
    )
    omni.kit.commands.execute("ApplyScriptingAPICommand", paths=paths)
    prim.GetAttribute("omni:scripting:scripts").Set([script_path])


def _create_worker(stage, assets_root, spec, graph_prim, script_path) -> str:
    from pxr import Gf, UsdGeom

    name = str(spec.get("name") or "Worker")
    start = spec.get("start") or {}
    wrapper_path = f"/World/People/{name}"
    wrapper = UsdGeom.Xform.Define(stage, wrapper_path)
    xform = UsdGeom.Xformable(wrapper)
    xform.AddTranslateOp().Set(
        Gf.Vec3d(float(start.get("x", 0.0) or 0.0), float(start.get("y", 0.0) or 0.0), float(start.get("z", 0.0) or 0.0))
    )
    yaw = float(start.get("yaw") or 0.0)
    if abs(yaw) > 1e-9:
        xform.AddRotateZOp().Set(float(yaw))

    asset_path = f"{wrapper_path}/Asset"
    asset = stage.DefinePrim(asset_path, "SkelRoot")
    asset.GetReferences().AddReference(str(assets_root) + "/" + str(spec["asset"]))
    _apply_graph_and_script(stage, asset, graph_prim, script_path)
    asset.ApplyAPI(_nav_schema().NavMeshExcludeAPI)

    return asset_path


def _prim_has_skel_descendant(stage, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        return False
    from pxr import UsdSkel

    if prim.GetTypeName() in ("SkelRoot", "Skeleton"):
        return True
    for child in prim.GetChildren():
        if _prim_has_skel_descendant(stage, str(child.GetPath())):
            return True
    return False


def _clear_worker_animation_source(stage, prim_path: str) -> None:
    from pxr import UsdSkel

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        return
    for prim_in_range in _collect_descendants(stage, prim_path):
        try:
            if prim_in_range.IsA(UsdSkel.Skeleton):
                rel = UsdSkel.BindingAPI(prim_in_range).GetAnimationSourceRel()
                if rel is not None and rel.GetTargetCount() > 0:
                    rel.ClearTargets(True)
        except Exception:
            pass
    prim.ApplyAPI(_nav_schema().NavMeshExcludeAPI)


def _exclude_dynamic_geometry(stage, roots) -> None:
    nav_schema = _nav_schema()

    def exclude(prefix: str) -> None:
        prim = stage.GetPrimAtPath(prefix)
        if prim is None or not prim.IsValid():
            return
        try:
            prim.ApplyAPI(nav_schema.NavMeshExcludeAPI)
        except Exception:
            pass
        for child in prim.GetChildren():
            exclude(str(child.GetPath()))

    for root in roots:
        if not root:
            continue
        exclude(str(root))


def _add_navmesh_floor(stage, materials, bounds) -> None:
    from pxr import Gf, UsdGeom, UsdShade

    b = bounds or {}
    x_min = float(b.get("x_min", -10.0))
    x_max = float(b.get("x_max", 10.0))
    y_min = float(b.get("y_min", -7.0))
    y_max = float(b.get("y_max", 7.0))
    z = 0.001

    floor_path = "/World/Floor"
    mesh = UsdGeom.Mesh.Define(stage, floor_path)
    mesh.CreatePointsAttr([
        (x_min, y_min, z), (x_max, y_min, z),
        (x_max, y_max, z), (x_min, y_max, z),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([(x_min, y_min, z), (x_max, y_max, z)])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    if materials is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials.get("floor"))
    _log(f"navmesh floor authored at {floor_path}")


def _author_navmesh_settings(stage, agent_radius: float, people_cfg) -> None:
    nav_cfg = people_cfg.get("navmesh") or {}
    nav_settings = nav_cfg.get("settings") or {}
    root_layer = stage.GetRootLayer()
    data = dict(root_layer.customLayerData or {})
    data["navmeshSettings"] = {
        "agentMaxFloorSlope": float(nav_settings.get("agent_max_floor_slope", 45.0)),
        "agentMaxRadius": float(int(agent_radius * 100.0)),
        "agentMaxStepHeight": float(nav_settings.get("agent_max_step_height", 40.0)),
        "agentMinHeight": float(nav_settings.get("agent_min_height", 100.0)),
        "areas": {
            "0": {"areaName": "Walkable", "color": (0.2, 0.8, 1.0), "defaultCost": 1.0},
            "1": {"areaName": "NotWalkable", "color": (1.0, 0.0, 0.0), "defaultCost": -1.0},
        },
    }
    root_layer.customLayerData = data


def _add_navmesh_volume(stage, bounds) -> None:
    from pxr import Gf, UsdGeom
    import NavSchema

    b = bounds or {}
    x_min = float(b.get("x_min", -10.0)) + 0.25
    x_max = float(b.get("x_max", 10.0)) - 0.25
    y_min = float(b.get("y_min", -7.0)) + 0.25
    y_max = float(b.get("y_max", 7.0)) - 0.25
    z0 = -0.05
    z1 = 3.0
    width = x_max - x_min
    depth = y_max - y_min
    height = z1 - z0
    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5
    center_z = (z0 + z1) * 0.5

    nav_vol = NavSchema.NavMeshVolume.Define(stage, "/World/NavMeshVolume")
    nav_vol.GetNavVolumeTypeAttr().Set("Include")
    vprim = nav_vol.GetPrim()
    vprim.ApplyAPI(NavSchema.NavMeshAreaAPI)
    UsdGeom.Boundable(vprim).GetExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    vxf = UsdGeom.Xformable(vprim)
    vxf.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, center_z))
    vxf.AddScaleOp().Set(Gf.Vec3f(width, depth, height))
    _log(f"navmesh volume authored at /World/NavMeshVolume (include)")


def bake_navmesh(app) -> object:
    import omni.anim.navigation.core as nav

    iface = nav.acquire_interface()
    iface.start_navmesh_baking()
    _log("navmesh bake started")
    deadline = time.time() + 30.0
    while time.time() < deadline:
        app.update()
        try:
            baking = bool(iface.is_navmesh_baking())
        except Exception:
            baking = False
        if not baking:
            break
        time.sleep(0.005)
    navmesh = iface.get_navmesh()
    if navmesh is None:
        _log("navmesh bake FAILED: no navmesh produced")
        return None
    agent_radius = 0.0
    try:
        agent_radius = float(navmesh.get_agent_max_radius())
    except Exception:
        agent_radius = 0.0
    _log(f"navmesh baked: agent_radius={agent_radius:.3f} m")
    return navmesh


def write_command_file(people_cfg, project_root) -> str:
    path = people_cfg.get("command_file") or os.path.join(
        tempfile.gettempdir(), "warehouse_people_commands.txt"
    )
    if not os.path.isabs(path):
        path = os.path.join(str(project_root or "/"), str(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lines = []
    for worker in people_cfg.get("workers") or []:
        name = str(worker.get("name") or worker.get("role") or "Worker")
        route = worker.get("route") or []
        if not route:
            lines.append(f"{name} Idle 10")
            continue
        for wp in route:
            x = float(wp.get("x") or 0.0)
            y = float(wp.get("y") or 0.0)
            yaw = wp.get("yaw")
            rot = f"{float(yaw):.1f}" if yaw is not None else "_"
            lines.append(f"{name} GoTo {x:.4f} {y:.4f} 0 {rot}")
            if wp.get("idle") is not None:
                lines.append(f"{name} Idle {float(wp['idle']):.2f}")
            if wp.get("look_around") is not None:
                lines.append(f"{name} LookAround {float(wp['look_around']):.2f}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def configure_people_settings(
    people_cfg, script_path, command_path, assets_root, agent_radius
) -> None:
    import carb
    from omni.anim.people.settings import PeopleSettings

    settings = carb.settings.get_settings()
    settings.set(
        PeopleSettings.CHARACTER_PRIM_PATH,
        str(people_cfg.get("character_root") or "/World/People"),
    )
    settings.set(PeopleSettings.COMMAND_FILE_PATH, command_path)
    settings.set(PeopleSettings.NUMBER_OF_LOOP, str(people_cfg.get("number_of_loop") or "inf"))
    settings.set(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED, bool(people_cfg.get("dynamic_avoidance", True)))
    settings.set(PeopleSettings.NAVMESH_ENABLED, bool((people_cfg.get("navmesh") or {}).get("enabled", True)))
    settings.set(PeopleSettings.BEHAVIOR_SCRIPT_PATH, script_path)
    settings.set(PeopleSettings.CHARACTER_ASSETS_PATH, str(assets_root) + "/Isaac/People/Characters")
    final_dist = (people_cfg.get("navmesh") or {}).get("final_target_distance")
    if final_dist is not None:
        settings.set(PeopleSettings.CHARACTER_FINAL_TARGET_DISTANCE, float(final_dist))

    nav_cfg = people_cfg.get("navmesh") or {}
    nav_settings = nav_cfg.get("settings") or {}
    settings.set(
        "/exts/omni.anim.navigation.core/navMesh/config/agentMaxRadius",
        float(int(agent_radius * 100.0)),
    )
    settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius", 5.0)
    settings.set(
        "/exts/omni.anim.navigation.core/navMesh/config/agentMaxStepHeight",
        float(nav_settings.get("agent_max_step_height", 40.0)),
    )
    settings.set(
        "/exts/omni.anim.navigation.core/navMesh/config/agentMinHeight",
        float(nav_settings.get("agent_min_height", 100.0)),
    )
    settings.set(
        "/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope",
        float(nav_settings.get("agent_max_floor_slope", 45.0)),
    )


def setup_people(
    app,
    stage,
    cfg,
    assets_root,
    dynamic_roots=(),
    materials=None,
    project_root=None,
    log=print,
) -> object:
    people_cfg = cfg.get("people") or {}
    if not people_cfg.get("enabled", True):
        return None
    workers_cfg = people_cfg.get("workers") or []
    if not workers_cfg:
        return None

    from isaacsim.core.utils.extensions import get_extension_path_from_name

    script_path = os.path.join(
        get_extension_path_from_name("omni.anim.people"),
        "omni/anim/people/scripts/character_behavior.py",
    )
    command_path = write_command_file(people_cfg, project_root or "/")
    agent_radius = float(people_cfg.get("agent_radius") or 0.5)
    configure_people_settings(people_cfg, script_path, command_path, assets_root, agent_radius)

    graph_prim = load_animation_graph(stage, cfg, assets_root, app)
    if graph_prim is None:
        return None

    workers = []
    for spec in workers_cfg:
        asset_path = _create_worker(stage, assets_root, spec, graph_prim, script_path)
        workers.append(dict(spec, asset_path=asset_path))

    def workers_loaded():
        for worker in workers:
            if not _prim_has_skel_descendant(stage, worker["asset_path"]):
                return False
        return True

    if not _wait_for_stage(app, workers_loaded, timeout=90.0):
        _log("worker characters did not finish loading in time")
        return None

    for worker in workers:
        _clear_worker_animation_source(stage, worker["asset_path"])

    _exclude_dynamic_geometry(
        stage, tuple(dynamic_roots) + ("/World/Characters/Biped_Setup", "/World/Process")
    )

    _add_navmesh_floor(stage, materials, (cfg.get("world") or {}).get("bounds"))
    _add_navmesh_volume(stage, (cfg.get("world") or {}).get("bounds"))
    _author_navmesh_settings(stage, agent_radius, people_cfg)
    for _ in range(5):
        app.update()
    navmesh = bake_navmesh(app)

    _log(f"People crew ready: {len(workers)} agents -> commands: {command_path}")
    return {
        "workers": [w.get("name") for w in workers],
        "worker_count": len(workers),
        "agent_radius": agent_radius,
        "navmesh": navmesh,
        "commands": command_path,
        "graph": graph_prim,
        "stage": stage,
    }
