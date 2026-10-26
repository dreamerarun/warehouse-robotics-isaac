#!/usr/bin/env python3
"""Static warehouse scene authored directly with USD.

Everything here is collision geometry that the fleet planner also knows about
(via config/warehouse.yaml).  No external asset downloads are required for the
building itself - only the robots come from the Isaac Sim asset root.
"""

from __future__ import annotations

import math

from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
class Materials:
    def __init__(self, stage):
        self.stage = stage
        self._cache = {}
        self._make("floor", (0.16, 0.17, 0.19), roughness=0.85)
        self._make("wall", (0.34, 0.36, 0.40), roughness=0.75)
        self._make("rack", (0.10, 0.32, 0.55), roughness=0.55, metallic=0.4)
        self._make("rack_shelf", (0.72, 0.74, 0.78), roughness=0.45, metallic=0.6)
        self._make("carton", (0.62, 0.42, 0.20), roughness=0.9)
        self._make("pallet", (0.45, 0.30, 0.14), roughness=0.9)
        self._make("table", (0.30, 0.32, 0.36), roughness=0.5, metallic=0.4)
        self._make("safety_yellow", (0.95, 0.72, 0.05), roughness=0.6,
                   emissive=(0.30, 0.20, 0.0))
        self._make("beacon", (1.0, 0.95, 0.6), roughness=0.4,
                   emissive=(2.5, 2.2, 1.0))

    def _make(self, name, color, roughness=0.5, metallic=0.0, emissive=None):
        path = f"/World/Looks/{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        if emissive is not None:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*emissive)
            )
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self._cache[name] = mat
        return mat

    def get(self, name):
        return self._cache[name]

    def tint(self, name, color, emissive=None):
        """Create (or reuse) an arbitrary emissive station material."""
        key = f"station_{name}"
        if key not in self._cache:
            self._make(key, color, roughness=0.55, emissive=emissive or color)
        return self._cache[key]


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------
def add_box(stage, path, size, position, material=None, collision=True,
            rotation=(0.0, 0.0, 0.0), visible=True):
    """Author a scaled USD cube (the workhorse for the whole warehouse)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
    if any(abs(v) > 1e-9 for v in rotation):
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*[float(v) for v in rotation]))
    xform.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    if not visible:
        UsdGeom.Imageable(prim).MakeInvisible()
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def add_cylinder(stage, path, radius, height, position, material=None,
                 collision=False):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    prim = cyl.GetPrim()
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*position))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def add_sphere(stage, path, radius, position, material=None):
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(float(radius))
    prim = sph.GetPrim()
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*position))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return prim


def _rotated_footprint(x, y, yaw, sx, sy):
    """Return the four world XY corners of a yaw-rotated box."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2),
                   (sx / 2, sy / 2), (-sx / 2, sy / 2)):
        out.append((x + dx * c - dy * s, y + dx * s + dy * c))
    return out


# ---------------------------------------------------------------------------
# Scene pieces
# ---------------------------------------------------------------------------
def build_scene(stage, cfg, materials, procedural_visuals=True):
    """Build floor, walls, racks, pallets, stations and lights.

    With ``procedural_visuals=False`` the same collision geometry is authored
    invisibly and the matching NVIDIA assets are added by
    ``warehouse_nvidia.build_nvidia_site``.  Physics, LiDAR and the fleet grid
    therefore see identical geometry in both modes.
    """
    world = cfg["world"]
    b = world["bounds"]
    width = b["x_max"] - b["x_min"]
    depth = b["y_max"] - b["y_min"]
    cx = (b["x_max"] + b["x_min"]) / 2.0
    cy = (b["y_max"] + b["y_min"]) / 2.0

    UsdGeom.Xform.Define(stage, "/World/Scene")

    # Floor ---------------------------------------------------------------
    # The visible floor is a thin box, but physics uses a plane: wheel
    # contacts on a scaled box are noticeably less stable in PhysX (the
    # Nova Carter creeps and tips while settling), while a ground plane is
    # rock solid.  The plane sits 1 mm below the visible surface.
    add_box(
        stage, "/World/Scene/Floor",
        (width, depth, world["floor_thickness"]),
        (cx, cy, -world["floor_thickness"] / 2.0),
        materials.get("floor"), collision=False,
    )
    PhysicsSchemaTools.addGroundPlane(
        stage,
        "/World/Scene/GroundPlane",
        "Z",
        50.0,
        Gf.Vec3f(cx, cy, -0.001),
        Gf.Vec3f(0.0, 0.0, 0.0),
    )

    # Perimeter walls -----------------------------------------------------
    t = world["wall_thickness"]
    h = world["wall_height"]
    walls = [
        ("north", (cx, b["y_max"] + t / 2, h / 2), (width + 2 * t, t, h)),
        ("south", (cx, b["y_min"] - t / 2, h / 2), (width + 2 * t, t, h)),
        ("east", (b["x_max"] + t / 2, cy, h / 2), (t, depth, h)),
        ("west", (b["x_min"] - t / 2, cy, h / 2), (t, depth, h)),
    ]
    for name, pos, size in walls:
        add_box(stage, f"/World/Scene/Wall_{name}", size, pos,
                materials.get("wall"), collision=True,
                visible=procedural_visuals)

    # Rack rows -----------------------------------------------------------
    for rack in cfg["scene"]["racks"]:
        if procedural_visuals:
            _add_rack(stage, rack, materials)
        else:
            _add_rack_collider(stage, rack, materials)

    # Static keep-out boxes (Franka table, charging bank, conveyors) ------
    for box in cfg["scene"].get("static_obstacles", []):
        sx, sy, sz = box["size"]
        yaw = float(box.get("yaw", 0.0))
        rot = (0.0, 0.0, math.degrees(yaw))
        name = box["name"]
        if name.startswith("conveyor_") or name.startswith("humanoid_conveyor"):
            # Invisible conveyor collider; the NVIDIA conveyor mesh is added
            # by the visual layer at exactly the same footprint.
            add_box(
                stage, f"/World/Scene/{name}", (sx, sy, sz),
                (box["x"], box["y"], sz / 2.0), materials.get("table"),
                collision=True, rotation=rot, visible=False,
            )
            continue
        if name == "franka_table":
            # A solid block would swallow the Panda's shoulder (the base sits
            # right at the table edge).  Author a real table: thin top plus
            # four legs, leaving the arm free to swing underneath the top.
            c, s = math.cos(yaw), math.sin(yaw)
            top = 0.04
            leg = 0.06

            def place(tag, dx, dy, size, z, collision=True, visible=True):
                px = box["x"] + dx * c - dy * s
                py = box["y"] + dx * s + dy * c
                add_box(
                    stage, f"/World/Scene/{box['name']}_{tag}", size,
                    (px, py, z), materials.get("table"),
                    collision=collision, rotation=rot, visible=visible,
                )

            # Visual tabletop and legs, but no collision near the robot base:
            # the Panda shoulder sits below the top and would otherwise jam
            # against it when the arm sweeps from cube A to cube B.
            place("top", 0.0, 0.0, (sx, sy, top), sz - top / 2.0,
                  collision=False)
            inset = leg * 0.6
            for ix, iy, tag in (
                (-1, -1, "leg_sw"), (1, -1, "leg_se"),
                (1, 1, "leg_ne"), (-1, 1, "leg_nw"),
            ):
                place(
                    tag,
                    ix * (sx / 2.0 - inset),
                    iy * (sy / 2.0 - inset),
                    (leg, leg, sz - top),
                    (sz - top) / 2.0,
                    collision=False,
                )
            # Invisible support slab under the cubes/tray only.
            place("support", -0.05, 0.0, (0.42, sy, top), sz - top / 2.0,
                  collision=True, visible=False)
            continue
        if name == "ur10_pedestal":
            # Dark machine base gives the white UR10 arm a readable outline.
            mat = materials.tint(name, (0.12, 0.15, 0.20))
        else:
            mat = materials.get("table")
        add_box(
            stage, f"/World/Scene/{box['name']}", (sx, sy, sz),
            (box["x"], box["y"], sz / 2.0), mat,
            collision=True, rotation=rot,
        )

    # Pallets with cartons ------------------------------------------------
    for pallet in cfg["scene"].get("pallets", []):
        if procedural_visuals:
            _add_pallet(stage, pallet, materials)
        else:
            _add_pallet_collider(stage, pallet, materials)

    # Station pads + beacons ----------------------------------------------
    for station in cfg["scene"]["stations"]:
        _add_station(stage, station, materials)

    # Lights --------------------------------------------------------------
    _add_lights(stage)

    return _rotated_footprint


def _add_rack_collider(stage, rack, materials):
    """Invisible solid rack footprint for physics / LiDAR."""
    x, y, yaw = rack["x"], rack["y"], rack.get("yaw", 0.0)
    sx, sy, h = rack["size"]
    add_box(
        stage, f"/World/Scene/{rack['name']}", (sx, sy, h),
        (x, y, h / 2.0), materials.get("rack"), collision=True,
        rotation=(0.0, 0.0, math.degrees(yaw)), visible=False,
    )


def _add_rack(stage, rack, materials):
    x, y, yaw = rack["x"], rack["y"], rack.get("yaw", 0.0)
    sx, sy, h = rack["size"]
    root = f"/World/Scene/{rack['name']}"
    UsdGeom.Xform.Define(stage, root)
    UsdGeom.Xformable(stage.GetPrimAtPath(root)).AddTranslateOp().Set(
        Gf.Vec3d(x, y, 0.0)
    )
    UsdGeom.Xformable(stage.GetPrimAtPath(root)).AddRotateXYZOp().Set(
        Gf.Vec3f(0.0, 0.0, math.degrees(yaw))
    )

    post = 0.08
    for i, (dx, dy) in enumerate(
        ((-sx / 2 + post / 2, -sy / 2 + post / 2),
         (sx / 2 - post / 2, -sy / 2 + post / 2),
         (sx / 2 - post / 2, sy / 2 - post / 2),
         (-sx / 2 + post / 2, sy / 2 - post / 2))
    ):
        add_box(stage, f"{root}/Post_{i}", (post, post, h), (dx, dy, h / 2),
                materials.get("rack"), collision=True)

    levels = (0.20, 0.78, 1.36, 1.94)
    for i, z in enumerate(levels):
        add_box(stage, f"{root}/Shelf_{i}", (sx, sy, 0.04), (0.0, 0.0, z),
                materials.get("rack_shelf"), collision=True)
        # Two cartons per shelf, offset so they read as stocked inventory.
        for j, (dx, dy) in enumerate(((-sx * 0.23, -sy * 0.12),
                                      (sx * 0.20, sy * 0.10))):
            if (i + j) % 3 == 0:
                continue
            cw = min(0.55, sx * 0.30)
            add_box(
                stage, f"{root}/Carton_{i}_{j}",
                (cw, sy * 0.45, 0.30),
                (dx, dy, z + 0.17),
                materials.get("carton"), collision=True,
            )


def _add_pallet_collider(stage, pallet, materials):
    """Invisible pallet + cargo block; the NVIDIA pallet is visual only."""
    x, y, yaw = pallet["x"], pallet["y"], pallet.get("yaw", 0.0)
    sx, sy, sz = pallet["size"]
    # Empty output pallets stay low so a robot arm can stack onto them.
    h = sz if pallet.get("empty") else max(sz, 0.85)
    add_box(
        stage, f"/World/Scene/{pallet['name']}", (sx, sy, h),
        (x, y, h / 2.0), materials.get("pallet"), collision=True,
        rotation=(0.0, 0.0, math.degrees(yaw)), visible=False,
    )


def _add_pallet(stage, pallet, materials):
    x, y, yaw = pallet["x"], pallet["y"], pallet.get("yaw", 0.0)
    sx, sy, sz = pallet["size"]
    root = f"/World/Scene/{pallet['name']}"
    add_box(stage, f"{root}/Base", (sx, sy, sz), (x, y, sz / 2),
            materials.get("pallet"), collision=True,
            rotation=(0.0, 0.0, math.degrees(yaw)))
    if pallet.get("empty"):
        return
    # A couple of cartons stacked on top, rotated with the pallet.
    c, s = math.cos(yaw), math.sin(yaw)
    for i, (dx, dy, dz) in enumerate(((0.0, 0.0, sz + 0.18),
                                      (0.18, -0.12, sz + 0.52))):
        wx = x + dx * c - dy * s
        wy = y + dx * s + dy * c
        add_box(stage, f"{root}/Carton_{i}", (0.55, 0.45, 0.34),
                (wx, wy, dz), materials.get("carton"), collision=True,
                rotation=(0.0, 0.0, math.degrees(yaw)))


def _add_station(stage, station, materials):
    x, y = station["x"], station["y"]
    color = tuple(station.get("color", (0.2, 0.6, 1.0)))
    mat = materials.tint(station["name"], color)
    # A thin emissive pad that AMRs drive onto.
    add_box(stage, f"/World/Scene/{station['name']}_pad", (1.45, 1.45, 0.012),
            (x, y, 0.008), mat, collision=False)
    # Corner posts + beacon, offset out of the driving line.
    for i, (dx, dy) in enumerate(((-0.72, -0.72), (0.72, -0.72),
                                  (0.72, 0.72), (-0.72, 0.72))):
        add_cylinder(stage, f"/World/Scene/{station['name']}_post{i}",
                     0.025, 0.55, (x + dx, y + dy, 0.275),
                     materials.get("safety_yellow"))
    add_sphere(stage, f"/World/Scene/{station['name']}_beacon",
               0.07, (x + 0.72, y + 0.72, 0.62), materials.get("beacon"))


def _add_lights(stage):
    dome = UsdGeom.Xform.Define(stage, "/World/Lights/Dome")
    dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome/DomeLight")
    dome_light.CreateIntensityAttr(1100.0)
    dome_light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(2600.0)
    key.CreateAngleAttr(0.55)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-48.0, 0.0, 38.0)
    )

    fill = UsdLux.DistantLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(900.0)
    fill.CreateColorAttr(Gf.Vec3f(0.62, 0.72, 1.0))
    UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-35.0, 0.0, 215.0)
    )
