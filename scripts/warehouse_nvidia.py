#!/usr/bin/env python3
"""NVIDIA SimReady assets for the warehouse.

The navigation model stays authored in ``config/warehouse.yaml`` (simple
rectangles) so the fleet planner and PhysX always agree.  This module only
changes what the facility *looks* like and adds render-only set dressing:
real NVIDIA warehouse racks and stock, concrete floor, wall panels, conveyors,
pallets, safety equipment and workers.

Every referenced asset has its own colliders disabled.  The frozen cube
colliders authored by ``warehouse_scene.py`` remain the only physical scene
geometry, apart from the dynamic forklift's own kinematic box.  That keeps
LiDAR, A* and the safety gates unchanged while the camera sees a real
industrial facility.

The asset layer is also used for visuals parented to moving prims (the
forklift and the H1 payload).  In that case the wrapper is authored under the
parent and measured in the parent's local frame.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel

from isaacsim.core.utils.stage import add_reference_to_stage


DTW = "NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"

RACK_USD = DTW + "Shelving/Racks/Rack_A/Rack_A08_PR_NVD_01.usd"
PALLET_USD = DTW + "Shipping/Pallets/Wood/Block_A/BlockPallet_A08_PR_NVD_01.usd"
WALL_USD = "Isaac/Environments/Simple_Warehouse/Props/SM_WallA_6M.usd"
FLOOR_USD = "Isaac/Environments/Simple_Warehouse/Props/SM_floor02.usd"
FORKLIFT_USD = DTW + "Equipment/Forklifts/Forklift_A/Forklift_A01_PR_V_NVD_01.usd"

# A08 is the straight roller-bed module.  A01/A03 are 90-degree transfer /
# scanner modules whose belt plane runs diagonally, so loads placed on their
# fitted footprint appeared to float beside the conveyor.
CONVEYOR_USD = (
    DTW + "Equipment/Conveyors/ConveyorBelt_A/ConveyorBelt_A08_PR_NVD_01.usd"
)
CONVEYOR_USDS = (CONVEYOR_USD,)

CARTON_USDS = (
    DTW + "Shipping/Cardboard_Boxes/Cube_A/CubeBox_A10_92cm_PR_NVD_01.usd",
    DTW + "Shipping/Cardboard_Boxes/Cube_A/CubeBox_A09_77cm_PR_NVD_01.usd",
    DTW + "Storage/Boxes/Box_A/Box_A01_60x40x32cm_PR_V_NVD_01.usd",
    DTW + "Storage/Boxes/Box_A/Box_A05_40x30x32cm_PR_V_NVD_01.usd",
)

STOCK_USDS = (
    DTW + "Storage/Totes/Tote_A/Tote_A04_80x60x22cm_PR_V_NVD_01.usd",
    DTW + "Storage/Cases/Case_A/Case_A03_65x48x30cm_PR_V_NVD_01.usd",
    DTW + "Storage/Boxes/Box_A/Box_A01_60x40x32cm_PR_V_NVD_01.usd",
    DTW + "Storage/Boxes/Box_A/Box_A05_40x30x32cm_PR_V_NVD_01.usd",
    DTW + "Storage/Containers/Container_A/Container_A03_63x45x33cm_PR_V_NVD_01.usd",
)

PAYLOAD_USD = DTW + "Storage/Boxes/Box_A/Box_A05_40x30x32cm_PR_V_NVD_01.usd"

WORKER_USDS = (
    "Isaac/People/Characters/male_adult_construction_01_new/"
    "male_adult_construction_01_new.usd",
    "Isaac/People/Characters/male_adult_construction_03/"
    "male_adult_construction_03.usd",
    "Isaac/People/Characters/male_adult_construction_05_new/"
    "male_adult_construction_05_new.usd",
    "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "Isaac/People/Characters/male_adult_construction_01_new/"
    "male_adult_construction_01_new.usd",
)

PROP_ASSETS = {
    "crate": DTW + "Shipping/Wood_Crates/Standard_A/"
             "StandardWoodCrate_A19_80x80x80cm_PR_NV_01.usd",
    "handtruck": DTW + "Equipment/Hand_Trucks/Std_Alum_A/"
                 "StandardAluminumHandTruck_A01_PR_NVD_01.usd",
    "trashcan": DTW + "Facilities/Trash_Cans/TrashCanWithWheels_A/"
                "TrashCanWithWheels_A01_Yellow_01.usd",
    "cone": DTW + "Safety/Cones/Traffic/TrafficCone_A04_71cm_PR_V_NVD_01.usd",
    "guardrail": DTW + "Safety/Barriers/GuardRail_A/GuardRail_A01_01.usd",
    "barrel": "Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
    "pushcart": "Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd",
}

# Static set dressing, deliberately outside the AMR driving lanes.  Only the
# visual layer uses these positions; none of them are task stations.
PROPS = (
    ("crate", 9.2, -3.9, 0.0),
    ("crate", 9.2, -2.8, 0.0),
    ("handtruck", 9.3, -0.2, 180.0),
    ("guardrail", 8.7, -4.9, 0.0),
    ("barrel", -9.3, -0.6, 0.0),
    ("cone", -4.9, 2.7, 0.0),
    ("cone", -2.7, 2.7, 0.0),
    ("trashcan", 9.3, 5.8, 0.0),
    ("pushcart", 3.2, 6.2, 0.0),
)

# (x, y, yaw, character index, frozen-pose index)
WORKERS = (
    (-8.9, 2.6, 90.0, 0, 0),    # inbound receiving clerk
    (-4.9, 3.7, 90.0, 1, 3),    # forklift spotter, west of the lane
    (4.1, 6.2, 180.0, 2, 1),    # packing-line supervisor
    (8.9, -2.1, 180.0, 3, 4),   # robotics-cell technician
    (-9.0, -2.3, 0.0, 4, 2),    # west-aisle picker
)

_POSE_CLIPS = (
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


class NvidiaAssetLayer:
    """Queue, load, measure and fit NVIDIA assets before ``world.play()``."""

    def __init__(self, stage, assets_root, app, log=print):
        self.stage = stage
        self.assets_root = assets_root
        self.app = app
        self.log = log
        self.jobs = []
        self.applied = []
        self.stats = Counter()
        self.errors = []
        self._cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render"],
            useExtentsHint=True,
        )

    # -- queueing ---------------------------------------------------------
    def add(self, kind, rel, prim_path, pos, size=(0.0, 0.0, 0.0),
            mode="natural", anchor="floor", yaw=0.0, hide=(), cap=3.0,
            parent=None):
        self.jobs.append(
            {
                "kind": kind,
                "rel": rel,
                "path": prim_path,
                "pos": tuple(float(v) for v in pos),
                "size": tuple(float(v) for v in size),
                "mode": mode,
                "anchor": anchor,
                "yaw": float(yaw),
                "hide": tuple(hide),
                "cap": float(cap),
                "parent": parent,
            }
        )
        return prim_path

    # -- asset health -----------------------------------------------------
    def _exists(self, url):
        try:
            import omni.client

            code, _ = omni.client.stat(url)
            return code == omni.client.Result.OK
        except Exception:
            return False

    def _disable_physics(self, root_path):
        root = self.stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            return 0
        n = 0
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                n += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = UsdPhysics.RigidBodyAPI(prim)
                try:
                    rb.CreateRigidBodyEnabledAttr(False)
                except Exception:
                    try:
                        rb.CreateKinematicEnabledAttr(True)
                    except Exception:
                        pass
        return n

    def _measure(self, path, parent_path):
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        rng = self._cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        mn = np.array(rng.GetMin(), dtype=float)
        mx = np.array(rng.GetMax(), dtype=float)
        if parent_path:
            pp = self.stage.GetPrimAtPath(parent_path)
            if pp and pp.IsValid():
                t = UsdGeom.Xformable(pp).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ).ExtractTranslation()
                mn -= np.array([t[0], t[1], t[2]], dtype=float)
                mx -= np.array([t[0], t[1], t[2]], dtype=float)
        return mn, mx

    # -- load + fit -------------------------------------------------------
    def apply(self):
        pending = []
        for job in self.jobs:
            url = self.assets_root + "/" + job["rel"]
            if not self._exists(url):
                self.log(f"NVIDIA_ASSET_MISSING: {job['kind']} {job['rel']}")
                self.stats["skipped"] += 1
                continue
            try:
                UsdGeom.Xform.Define(self.stage, job["path"])
                add_reference_to_stage(
                    usd_path=url, prim_path=job["path"] + "/Asset"
                )
            except Exception as exc:
                self.log(f"NVIDIA_ASSET_REF_ERROR: {job['rel']} -> {exc}")
                self.stats["skipped"] += 1
                self.errors.append(str(exc))
                continue
            ncol = self._disable_physics(job["path"])
            if ncol:
                self.stats["colliders_disabled"] += ncol
            pending.append(job)

        if pending:
            # Let every reference compose once before measuring.
            self.app.update()
            self.app.update()

        for job in pending:
            bb = self._measure(job["path"], job["parent"])
            if bb is None:
                self.log(f"NVIDIA_ASSET_EMPTY_BBOX: {job['kind']} {job['rel']}")
                try:
                    self.stage.RemovePrim(job["path"])
                except Exception:
                    pass
                self.stats["skipped"] += 1
                continue
            mn, mx = bb
            nat = mx - mn
            if nat[0] <= 1e-6 or nat[1] <= 1e-6 or (
                job["mode"] != "floor" and nat[2] <= 1e-6
            ):
                self.log(
                    f"NVIDIA_ASSET_DEGENERATE: {job['kind']} {job['rel']} "
                    f"size={np.round(nat, 4).tolist()}"
                )
                try:
                    self.stage.RemovePrim(job["path"])
                except Exception:
                    pass
                self.stats["skipped"] += 1
                continue

            ctr = (mn + mx) * 0.5
            tgt = np.array(job["size"], dtype=float)
            mode = job["mode"]
            if mode == "uniform":
                s = min(float(tgt[i] / nat[i]) for i in range(3))
                scl = np.array([s, s, s])
            elif mode == "uniform_xy":
                s = min(float(tgt[0] / nat[0]), float(tgt[1] / nat[1]))
                scl = np.array([s, s, s])
            elif mode == "exact":
                scl = tgt / nat
            elif mode == "floor":
                s = min(float(tgt[0] / nat[0]), float(tgt[1] / nat[1]))
                zs = s if nat[2] <= 1e-6 else min(s, 0.010 / nat[2])
                scl = np.array([s, s, zs])
            else:  # natural
                s = 1.0
                big = float(np.max(nat))
                if job["cap"] > 0.0 and big > job["cap"]:
                    s = job["cap"] / big
                scl = np.array([s, s, s])

            yaw = math.radians(job["yaw"])
            c, sn = math.cos(yaw), math.sin(yaw)
            cs = np.array([scl[0] * ctr[0], scl[1] * ctr[1]])
            cx = c * cs[0] - sn * cs[1]
            cy = sn * cs[0] + c * cs[1]
            px, py, pz = job["pos"]
            tx, ty = px - cx, py - cy
            if job["anchor"] == "top":
                tz = pz - scl[2] * mx[2]
            elif job["anchor"] == "center":
                tz = pz - scl[2] * ctr[2]
            else:
                tz = pz - scl[2] * mn[2]

            try:
                xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(job["path"]))
                # T * R * S: scale in the asset's local frame, then rotate.
                xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
                if abs(job["yaw"]) > 1e-6:
                    xf.AddRotateZOp().Set(float(job["yaw"]))
                xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scl]))
            except Exception as exc:
                self.log(f"NVIDIA_ASSET_XFORM_ERROR: {job['rel']} -> {exc}")
                self.stats["skipped"] += 1
                continue

            for hp in job["hide"]:
                hprim = self.stage.GetPrimAtPath(hp)
                if hprim and hprim.IsValid():
                    UsdGeom.Imageable(hprim).MakeInvisible()

            self.applied.append(job["path"])
            self.stats[job["kind"]] += 1
            self.log(
                f"NVIDIA_ASSET_USED: {job['kind']} {job['rel']} -> {job['path']} "
                f"natural {nat[0]:.3f}x{nat[1]:.3f}x{nat[2]:.3f} m "
                f"fitted {scl[0] * nat[0]:.3f}x{scl[1] * nat[1]:.3f}x"
                f"{scl[2] * nat[2]:.3f} m"
            )

        self.log(
            "NVIDIA_ASSETS: "
            + " ".join(f"{k}={v}" for k, v in sorted(self.stats.items()))
        )
        return self.applied

    # -- worker poses -----------------------------------------------------
    def author_worker_poses(self, worker_paths):
        """Freeze one RL-rig pose on each worker (render only)."""
        char_joints = None
        for path in worker_paths:
            if path not in self.applied:
                continue
            root = self.stage.GetPrimAtPath(path + "/Asset")
            if not root or not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdSkel.Skeleton):
                    char_joints = list(
                        UsdSkel.Skeleton(prim).GetJointsAttr().Get() or []
                    )
                    break
            if char_joints:
                break
        if not char_joints:
            self.log("NVIDIA_WORKER_POSE: unavailable (no skeleton)")
            return

        scratch = "/World/_WorkerClip"
        pose_of = {}
        for i, (rel, frame, tag) in enumerate(_POSE_CLIPS):
            url = self.assets_root + "/" + rel
            if not self._exists(url):
                self.log(f"NVIDIA_WORKER_CLIP_MISSING: {rel}")
                continue
            try:
                self.stage.DefinePrim(scratch)
                add_reference_to_stage(usd_path=url, prim_path=scratch)
                prim = self.stage.GetPrimAtPath(scratch)
                if not UsdSkel.IsSkelAnimationPrim(prim):
                    self.log(f"NVIDIA_WORKER_CLIP_NOT_ANIM: {rel}")
                    self.stage.RemovePrim(scratch)
                    continue
                src = UsdSkel.Animation(prim)
                joints = list(src.GetJointsAttr().Get() or [])
                norm = [
                    ("RL_BoneRoot" + j[5:]) if j.startswith("_Root") else j
                    for j in joints
                ]
                if norm != char_joints:
                    self.log(
                        f"NVIDIA_WORKER_CLIP_RIG_MISMATCH: {rel} "
                        f"{len(joints)} vs {len(char_joints)}"
                    )
                    self.stage.RemovePrim(scratch)
                    continue
                tc = Usd.TimeCode(frame)
                tr = src.GetTranslationsAttr()
                ro = src.GetRotationsAttr()
                sc = src.GetScalesAttr()
                v_tr = tr.Get(tc) if tr is not None and tr.HasValue() else None
                v_ro = ro.Get(tc) if ro is not None and ro.HasValue() else None
                v_sc = sc.Get(tc) if sc is not None and sc.HasValue() else None
                if v_tr is None or v_ro is None:
                    self.stage.RemovePrim(scratch)
                    continue
                out = UsdSkel.Animation.Define(
                    self.stage, f"/World/WorkerPose{i}"
                )
                out.CreateJointsAttr().Set(char_joints)
                out.CreateTranslationsAttr().Set(v_tr)
                out.CreateRotationsAttr().Set(v_ro)
                if v_sc is not None:
                    out.CreateScalesAttr().Set(v_sc)
                pose_of[i] = (f"/World/WorkerPose{i}", tag)
                self.stage.RemovePrim(scratch)
            except Exception as exc:
                self.log(f"NVIDIA_WORKER_CLIP_ERROR: {rel} -> {exc}")
                try:
                    self.stage.RemovePrim(scratch)
                except Exception:
                    pass

        bound = 0
        for i, (_x, _y, _yaw, _ci, pose_idx) in enumerate(WORKERS):
            path = f"/World/NVIDIA/Worker{i + 1}"
            if path not in self.applied or pose_idx not in pose_of:
                continue
            try:
                root = None
                for prim in Usd.PrimRange(
                    self.stage.GetPrimAtPath(path + "/Asset")
                ):
                    if prim.IsA(UsdSkel.Root):
                        root = prim
                        break
                if root is None:
                    continue
                UsdSkel.BindingAPI.Apply(root)
                UsdSkel.BindingAPI(root).CreateAnimationSourceRel().SetTargets(
                    [pose_of[pose_idx][0]]
                )
                bound += 1
            except Exception as exc:
                self.log(f"NVIDIA_WORKER_POSE_ERROR: {path} -> {exc}")
        self.log(f"NVIDIA_WORKER_POSE: bound={bound}/{len(WORKERS)}")


def build_nvidia_site(layer, cfg, stage):
    """Queue the industrial set dressing from config + module tables."""
    world = cfg["world"]
    b = world["bounds"]
    width = b["x_max"] - b["x_min"]
    depth = b["y_max"] - b["y_min"]
    cx = (b["x_max"] + b["x_min"]) * 0.5
    cy = (b["y_max"] + b["y_min"]) * 0.5

    # Concrete floor slab, sitting just above the dark physics base.
    layer.add(
        "floor", FLOOR_USD, "/World/NVIDIA/Floor",
        (cx, cy, 0.006), (width, depth, 0.0),
        mode="floor", anchor="top",
    )

    # Wall panels tile the four perimeter cubes (4 panels per long wall).
    for tag, wcx, wcy, rot, plen in (
        ("north", cx, b["y_max"] + 0.08, 0.0, width / 4.0),
        ("south", cx, b["y_min"] - 0.08, 0.0, width / 4.0),
        ("east", b["x_max"] + 0.08, cy, 90.0, depth / 4.0),
        ("west", b["x_min"] - 0.08, cy, 90.0, depth / 4.0),
    ):
        for i in range(4):
            if tag in ("north", "south"):
                off = b["x_min"] + plen * (i + 0.5)
                pos = (off, wcy, 0.0)
                size = (plen, 0.44, 3.05)
                yaw = rot
            else:
                off = b["y_min"] + plen * (i + 0.5)
                pos = (wcx, off, 0.0)
                size = (0.44, plen, 3.05)
                yaw = rot
            layer.add(
                "wall", WALL_USD, f"/World/NVIDIA/Wall_{tag}{i}",
                pos, size, mode="exact", anchor="floor", yaw=yaw,
            )

    # Real rack frames and real stock on both decks.
    for rack in cfg["scene"]["racks"]:
        x, y = float(rack["x"]), float(rack["y"])
        sx, sy, sh = (float(v) for v in rack["size"])
        yaw = float(rack.get("yaw", 0.0))
        tag = rack["name"]
        layer.add(
            "rack", RACK_USD, f"/World/NVIDIA/{tag}",
            (x, y, 0.0), (sx, sy, sh), mode="uniform", anchor="floor",
            yaw=math.degrees(yaw),
        )
        c, s = math.cos(yaw), math.sin(yaw)
        for deck_i, deck_z in enumerate((0.46, 1.13)):
            for k, dx in enumerate((-0.9, 0.0, 0.9)):
                rel = STOCK_USDS[(deck_i * 3 + k) % len(STOCK_USDS)]
                wx = x + dx * c
                wy = y + dx * s
                layer.add(
                    "stock", rel, f"/World/NVIDIA/{tag}_stock{deck_i}{k}",
                    (wx, wy, deck_z), (0.62, 0.46, 0.30),
                    mode="exact", anchor="floor",
                    yaw=math.degrees(yaw),
                )

    # Pallets with cartons on top.
    for i, pallet in enumerate(cfg["scene"].get("pallets", [])):
        x, y = float(pallet["x"]), float(pallet["y"])
        yaw = float(pallet.get("yaw", 0.0))
        sx, sy, sz = (float(v) for v in pallet["size"])
        layer.add(
            "pallet", PALLET_USD, f"/World/NVIDIA/{pallet['name']}",
            (x, y, 0.0), (sx, sy, sz), mode="exact", anchor="floor",
            yaw=math.degrees(yaw),
        )
        if pallet.get("empty"):
            continue
        for k, dz in enumerate((sz + 0.16, sz + 0.50)):
            layer.add(
                "carton", CARTON_USDS[(i + k) % len(CARTON_USDS)],
                f"/World/NVIDIA/{pallet['name']}_carton{k}",
                (x, y, dz), (0.58, 0.46, 0.33),
                mode="exact", anchor="floor",
                yaw=math.degrees(yaw) + (8.0 if k else -5.0),
            )

    # Conveyors behind the stations and the two H1 handoff stations.
    for i, conv in enumerate(cfg["scene"].get("conveyors", [])):
        x, y = float(conv["x"]), float(conv["y"])
        yaw = float(conv.get("yaw", 0.0))
        sx, sy, sz = (float(v) for v in conv.get("size", (2.4, 0.8, 1.166)))
        # The A08 asset is authored along +Y.  Rotate it so the long axis is
        # along X, and fit the swapped local dimensions: after the 90-degree
        # rotation the world footprint is exactly sx (X) by sy (Y).
        layer.add(
            "conveyor", CONVEYOR_USDS[i % len(CONVEYOR_USDS)],
            f"/World/NVIDIA/{conv['name']}",
            (x, y, 0.0), (sy, sx, sz), mode="exact", anchor="floor",
            yaw=math.degrees(yaw) + 90.0,
        )

    # Safety / facilities dressing from the NVIDIA DigitalTwin library.
    for i, (kind, x, y, yaw) in enumerate(PROPS):
        rel = PROP_ASSETS.get(kind)
        if rel is None:
            continue
        layer.add(
            "prop", rel, f"/World/NVIDIA/Prop{i}_{kind}",
            (x, y, 0.0), mode="natural", anchor="floor",
            yaw=yaw, cap=2.4,
        )

    # Render-only crew.  Their bodies have no physics colliders.
    for i, (_x, _y, yaw, ci, _pose) in enumerate(WORKERS):
        layer.add(
            "worker", WORKER_USDS[ci], f"/World/NVIDIA/Worker{i + 1}",
            (_x, _y, 0.0), mode="natural", anchor="floor",
            yaw=yaw, cap=2.1,
        )
