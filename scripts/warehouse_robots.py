#!/usr/bin/env python3
"""Robot models used by the warehouse simulation.

* ``AMR``          - Nova Carter differential-drive base + 2-D PhysX LiDAR
* ``FrankaCell``   - Franka Panda + table, grasp cubes and a tray
* ``HumanoidWalker`` - Unitree H1 driven by Isaac's flat-terrain RL policy

All three are controlled from ROS 2 topics.  The simulation applies the last
received command every physics step, so the 200 Hz physics rate is independent
of the 20-30 Hz ROS 2 command rate.
"""

from __future__ import annotations

import math
import time

import numpy as np

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade


# ---------------------------------------------------------------------------
# Small quaternion helpers (w, x, y, z - Isaac Sim's convention)
# ---------------------------------------------------------------------------
def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    vq = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, vq), quat_inv(q))[1:]


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------
# AMR
# ---------------------------------------------------------------------------
class AMR:
    WHEEL_RADIUS = 0.14
    WHEEL_SEPARATION = 0.4132

    def __init__(self, name, spec, cfg, assets_root, visual=True):
        self.name = name
        self.spec = spec
        self.cfg = cfg
        self.assets_root = assets_root
        self.visual = visual
        self.robot_path = f"/World/AMRs/{name}"
        self.chassis_path = f"{self.robot_path}/chassis_link"
        self.lidar_path = f"{self.chassis_path}/lidar"
        self.laser_frame = f"{name}_laser"
        self.art = None
        self.lidar = None
        self.wheel_idx = None
        self.payload_path = f"{self.chassis_path}/Payload"
        self.payload_visible = False
        self._last_visual_pos = None
        self._near_station = False
        self._left_station = False

    # -- authoring (before world.play) ------------------------------------
    def create(self):
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

        add_reference_to_stage(
            self.assets_root
            + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
            self.robot_path,
        )
        # The referenced articulation composes at the origin.  Put it at its
        # spawn before world.reset(), otherwise it can be initialised inside a
        # rack collider and PhysX will produce NaN poses.
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.robot_path)
        if prim and prim.IsValid():
            spawn = self.spec["spawn"]
            set_prim_pose(
                prim,
                (spawn["x"], spawn["y"], spawn.get("z", 0.10)),
                float(spawn.get("yaw", 0.0)),
            )
        # Give the reference a moment to compose before attaching a sensor.
        return True

    def create_lidar(self, app):
        from isaacsim.sensors.physx import RotatingLidarPhysX

        mount = self.spec.get("lidar_mount", [-0.06, 0.0, 0.38])
        self.lidar = RotatingLidarPhysX(
            prim_path=self.lidar_path,
            name=f"{self.name}_lidar",
            rotation_frequency=0.0,
            translation=np.array(mount, dtype=float),
            fov=(360.0, 1.0),
            resolution=(1.0, 1.0),
            valid_range=(0.25, 30.0),
        )
        # Force a true 2-D scan (the schema default is high LOD / 16 rows).
        self.lidar.prim.GetAttribute("highLod").Set(False)
        self.lidar.initialize()
        if self.visual:
            try:
                self.lidar.enable_visualization(
                    high_lod=False, draw_points=True, draw_lines=False
                )
            except Exception:
                pass
        app.update()

    # -- carried-tote visual ---------------------------------------------
    def create_payload(self, layer, materials=None, visual=True):
        """Author the render-only tote carried on the AMR deck.

        The wrapper lives under ``chassis_link`` so it follows the robot for
        free.  It is hidden until the AMR leaves a station, which makes the
        logical transport tasks legible on screen.
        """
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        if stage.GetPrimAtPath(self.payload_path):
            return
        wrapper = UsdGeom.Xform.Define(stage, self.payload_path).GetPrim()
        xf = UsdGeom.Xformable(wrapper)
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.30))
        if visual and layer is not None:
            from warehouse_nvidia import PAYLOAD_USD

            layer.add(
                "amr_payload", PAYLOAD_USD, self.payload_path + "/Visual",
                (0.0, 0.0, 0.0), (0.50, 0.40, 0.28),
                mode="exact", anchor="floor", parent=self.payload_path,
            )
        elif materials is not None:
            from warehouse_scene import add_box

            add_box(
                stage, self.payload_path + "/Visual",
                (0.50, 0.40, 0.28), (0.0, 0.0, 0.14),
                materials.tint(f"amr_{self.name}", (0.20, 0.55, 0.95)),
                collision=False,
            )
        self.set_payload_visible(False)

    def set_payload_visible(self, visible):
        from isaacsim.core.utils.stage import get_current_stage

        prim = get_current_stage().GetPrimAtPath(self.payload_path)
        if prim and prim.IsValid():
            if visible:
                UsdGeom.Imageable(prim).MakeVisible()
            else:
                UsdGeom.Imageable(prim).MakeInvisible()
            self.payload_visible = bool(visible)

    def update_visual(self, dt, stations, parking=()):
        """Show the tote in transit and hide it while dwelling at a pad."""
        if not self.payload_path or self.art is None:
            return
        pos, _ = self.pose()
        if self._last_visual_pos is None:
            self._last_visual_pos = pos
            return
        speed = float(np.linalg.norm(pos - self._last_visual_pos)) / max(dt, 1e-6)
        self._last_visual_pos = pos

        near_station = any(
            math.hypot(pos[0] - float(s["x"]), pos[1] - float(s["y"])) < 0.95
            for s in stations
        )
        near_parking = any(
            math.hypot(pos[0] - float(p["x"]), pos[1] - float(p["y"])) < 1.00
            for p in parking
        )
        if near_station or near_parking:
            if self.payload_visible:
                self.set_payload_visible(False)
            if near_station:
                self._near_station = True
                self._left_station = False
            if near_parking:
                self._left_station = False
            return
        if self._near_station:
            self._near_station = False
            self._left_station = True
        if self._left_station and not self.payload_visible and speed > 0.10:
            self.set_payload_visible(True)

    # -- initialisation (after world.play) --------------------------------
    def initialize(self):
        from isaacsim.core.prims import SingleArticulation

        self.art = SingleArticulation(self.robot_path)
        self.art.initialize()
        spawn = self.spec["spawn"]
        self.art.set_world_pose(
            position=np.array(
                [spawn["x"], spawn["y"], float(spawn.get("z", 0.10))]
            ),
            orientation=yaw_quat(float(spawn.get("yaw", 0.0))),
        )
        try:
            self.art.set_world_velocity(np.zeros(6))
        except Exception:
            pass
        names = list(self.art.dof_names)
        self.wheel_idx = [
            names.index("joint_wheel_left"),
            names.index("joint_wheel_right"),
        ]
        self.apply_twist(0.0, 0.0)

    def apply_twist(self, vx, wz):
        from isaacsim.core.utils.types import ArticulationAction

        if self.art is None:
            return
        vx = float(np.clip(vx, -1.0, 1.0))
        wz = float(np.clip(wz, -2.0, 2.0))
        vl = (vx - wz * self.WHEEL_SEPARATION / 2.0) / self.WHEEL_RADIUS
        vr = (vx + wz * self.WHEEL_SEPARATION / 2.0) / self.WHEEL_RADIUS
        self.art.apply_action(
            ArticulationAction(
                joint_indices=self.wheel_idx, joint_velocities=[vl, vr]
            )
        )

    def pose(self):
        if self.art is None:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        pos, quat = self.art.get_world_pose()
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)


# ---------------------------------------------------------------------------
# Franka pick-and-place cell
# ---------------------------------------------------------------------------
class FrankaCell:
    def __init__(self, spec, cfg, assets_root, materials, visual=True):
        self.spec = spec
        self.cfg = cfg
        self.assets_root = assets_root
        self.materials = materials
        self.visual = visual
        self.robot_path = "/World/FrankaCell/Franka"
        self.cube_size = float(spec.get("cube_size", 0.05))
        self.art = None
        self.hand = None
        self.cubes = {}
        self.cube_home = {}
        self.attached = None
        self.attach_offset = None
        self.attach_quat = None
        self.last_command = None
        self.last_command_time = 0.0
        self.gripper_target = 0.04
        self.released_at = None
        self.close_logged = False
        self.reset_after = float(spec.get("reset_after", 6.0))
        self.placement_center = np.array(
            [spec["tray"]["x"], spec["tray"]["y"], spec["tray"]["z"]], dtype=float
        )

    # -- authoring --------------------------------------------------------
    def create(self):
        from isaacsim.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(
            self.assets_root
            + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            self.robot_path,
        )
        # Move the Franka off the origin before world.reset(); the origin is
        # inside the rack row and an articulation initialised there corrupts
        # the whole PhysX scene.
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.robot_path)
        if prim and prim.IsValid():
            base = self.spec["base"]
            set_prim_pose(
                prim,
                (base["x"], base["y"], base.get("z", 0.0)),
                float(base.get("yaw", 0.0)),
            )
        self._build_props()

    def _build_props(self):
        # Use the same primitive helpers as the warehouse, but author under the
        # cell root so the whole cell can be moved as one unit.
        from isaacsim.core.utils.stage import get_current_stage

        from warehouse_scene import add_box

        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, "/World/FrankaCell")

        table = self.spec.get("table", None)
        if table is None:
            t = self.cfg["scene"]["static_obstacles"]
            table = next(o for o in t if o["name"] == "franka_table")
        sx, sy, sz = table["size"]
        # Table top + legs are authored by the static scene; only add the tray.
        tray = self.spec["tray"]
        tx, ty, tz = tray["x"], tray["y"], tray["z"]
        tw, td, th = tray["size"]
        add_box(
            stage, "/World/FrankaCell/Tray_Base", (tw, td, th),
            (tx, ty, tz), self.materials.get("table"), collision=True,
        )
        wall = 0.025
        for name, size, pos in (
            ("North", (tw, wall, 0.05), (tx, ty + td / 2 - wall / 2, tz + 0.025)),
            ("South", (tw, wall, 0.05), (tx, ty - td / 2 + wall / 2, tz + 0.025)),
            ("East", (wall, td, 0.05), (tx + tw / 2 - wall / 2, ty, tz + 0.025)),
            ("West", (wall, td, 0.05), (tx - tw / 2 + wall / 2, ty, tz + 0.025)),
        ):
            add_box(
                stage, f"/World/FrankaCell/Tray_{name}", size, pos,
                self.materials.get("safety_yellow"), collision=True,
            )

        # Dynamic grasp cubes.
        for cube in self.spec["cubes"]:
            path = f"/World/FrankaCell/{cube['name']}"
            prim = add_box(
                stage, path,
                (self.cube_size, self.cube_size, self.cube_size),
                (cube["x"], cube["y"], cube["z"]),
                self.materials.tint(cube["name"], (0.95, 0.35, 0.05)),
                collision=True,
            )
            body = UsdPhysics.RigidBodyAPI.Apply(prim)
            body.CreateKinematicEnabledAttr(False)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(0.08)
            self.cube_home[cube["name"]] = np.array(
                [cube["x"], cube["y"], cube["z"]], dtype=float
            )

    # -- initialisation ---------------------------------------------------
    def initialize(self):
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim

        self.art = SingleArticulation(self.robot_path)
        self.art.initialize()
        base = self.spec["base"]
        self.art.set_world_pose(
            position=np.array([base["x"], base["y"], base.get("z", 0.0)]),
            orientation=yaw_quat(float(base.get("yaw", 0.0))),
        )
        self.hand = SingleRigidPrim(
            f"{self.robot_path}/{self.spec.get('eef_frame', 'panda_hand')}"
        )
        self.hand.initialize()
        for name in self.cube_home:
            rigid = SingleRigidPrim(f"/World/FrankaCell/{name}")
            rigid.initialize()
            self.cubes[name] = rigid

    # -- ROS command ------------------------------------------------------
    def apply_command(self, names_positions):
        """Apply a JointState command received from /franka/joint_command."""
        from isaacsim.core.utils.types import ArticulationAction

        if self.art is None or not names_positions:
            return
        names, positions = names_positions
        dof = list(self.art.dof_names)
        current = self.art.get_joint_positions()
        target = np.array(current, dtype=float, copy=True)
        mapping = dict(zip(names, positions))
        for i, name in enumerate(dof):
            if name in mapping:
                target[i] = mapping[name]
        self.last_command = target.copy()
        self.last_command_time = time.time()
        for finger in ("panda_finger_joint1", "panda_finger_joint2"):
            if finger in mapping:
                self.gripper_target = float(mapping[finger])
        self.art.apply_action(ArticulationAction(joint_positions=target))

    def _hand_pose(self):
        pos, quat = self.hand.get_world_pose()
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)

    # -- grasp state machine ---------------------------------------------
    def update(self, dt):
        if self.art is None:
            return
        # Hold the last position target between ROS messages.
        if self.last_command is not None:
            from isaacsim.core.utils.types import ArticulationAction

            self.art.apply_action(
                ArticulationAction(joint_positions=self.last_command)
            )

        hand_pos, hand_quat = self._hand_pose()

        closing = self.gripper_target < 0.005
        if self.attached is None and closing:
            best, best_d = None, 1e9
            for name, rigid in self.cubes.items():
                cp, _ = rigid.get_world_pose()
                d = float(np.linalg.norm(np.asarray(cp) - hand_pos))
                if d < best_d:
                    best, best_d = name, d
            if best is not None and best_d < 0.17:
                self.attached = best
                cp, cq = self.cubes[best].get_world_pose()
                self.attach_offset = quat_rotate(
                    quat_inv(hand_quat), np.asarray(cp) - hand_pos
                )
                self.attach_quat = quat_mul(quat_inv(hand_quat), np.asarray(cq))
                body = UsdPhysics.RigidBodyAPI.Apply(self.cubes[best].prim)
                body.CreateKinematicEnabledAttr().Set(True)
                print(
                    f"[franka] attached {best} (hand {np.round(hand_pos, 3).tolist()}, "
                    f"distance {best_d:.3f} m)", flush=True,
                )
            elif not self.close_logged:
                self.close_logged = True

        elif self.attached is not None and self.gripper_target > 0.025:
            rigid = self.cubes[self.attached]
            body = UsdPhysics.RigidBodyAPI.Apply(rigid.prim)
            body.CreateKinematicEnabledAttr().Set(False)
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
            print(f"[franka] released {self.attached}", flush=True)
            self.attached = None
            self.attach_offset = None
            self.attach_quat = None
            self.released_at = time.time()

        if not closing:
            self.close_logged = False

        if self.attached is not None:
            target_pos = hand_pos + quat_rotate(hand_quat, self.attach_offset)
            target_quat = quat_mul(hand_quat, self.attach_quat)
            self.cubes[self.attached].set_world_pose(target_pos, target_quat)

        self._maybe_reset_cubes()

    def _maybe_reset_cubes(self):
        """Recycle the demo cubes once both have been placed in the tray."""
        if self.attached is not None:
            return
        in_tray = 0
        for name, rigid in self.cubes.items():
            cp, _ = rigid.get_world_pose()
            cp = np.asarray(cp)
            if np.linalg.norm(cp - self.placement_center) < 0.30:
                in_tray += 1
        if in_tray == len(self.cubes) and self.released_at is not None:
            if time.time() - self.released_at > self.reset_after:
                for name, rigid in self.cubes.items():
                    body = UsdPhysics.RigidBodyAPI.Apply(rigid.prim)
                    body.CreateKinematicEnabledAttr().Set(True)
                    rigid.set_world_pose(
                        self.cube_home[name], np.array([1.0, 0.0, 0.0, 0.0])
                    )
                    body.CreateKinematicEnabledAttr().Set(False)
                    try:
                        rigid.set_linear_velocity(np.zeros(3))
                        rigid.set_angular_velocity(np.zeros(3))
                    except Exception:
                        pass
                print("[franka] both cubes placed - reset for next cycle", flush=True)
                self.released_at = None




def set_prim_pose(prim, position, yaw):
    """Set an existing referenced prim's translate/orient without duplicating ops."""
    xf = UsdGeom.Xformable(prim)
    t = xf.GetTranslateOp()
    if t is None:
        t = xf.AddTranslateOp()
    t.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    orient = xf.GetOrientOp()
    if orient is not None:
        q = yaw_quat(float(yaw))
        orient.Set(
            Gf.Quatd(
                float(q[0]),
                Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])),
            )
        )
        return
    rot = xf.GetRotateXYZOp()
    if rot is None:
        rot = xf.AddRotateXYZOp()
    rot.Set(Gf.Vec3f(0.0, 0.0, math.degrees(float(yaw))))


# ---------------------------------------------------------------------------
# Dynamic forklift (NVIDIA asset, kinematic route)
# ---------------------------------------------------------------------------
class Forklift:
    """Kinematic forklift that crosses the AMR station aisle.

    The root is an Xform with a kinematic RigidBodyAPI and one box collider, so
    PhysX LiDAR sees it and the fleet planner can react to its odometry.  The
    NVIDIA Forklift_A01 mesh is a render-only child of that root.
    """

    def __init__(self, spec, visual=True):
        self.spec = spec
        self.visual = visual
        self.name = str(spec.get("name", "forklift"))
        self.root_path = f"/World/{self.name.capitalize()}"
        self.body_path = f"/World/{self.name.capitalize()}Body"
        self.lane_x = float(spec.get("lane_x", -3.8))
        self.y_min = float(spec.get("y_min", -5.3))
        self.y_max = float(spec.get("y_max", 2.4))
        self.speed = float(spec.get("speed", 0.42))
        self.pause = float(spec.get("pause", 3.0))
        self.safety_radius = float(spec.get("safety_radius", 2.1))
        self.y = self.y_min
        self.translate_op = None
        self.body_translate_op = None
        self._hold_time = 0.0
        self._travel = max(0.1, self.y_max - self.y_min)
        self._travel_time = self._travel / max(0.05, self.speed)

    def create(self, layer):
        from pxr import Gf, UsdGeom, UsdPhysics

        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        root = UsdGeom.Xform.Define(stage, self.root_path)
        xf = UsdGeom.Xformable(root)
        self.translate_op = xf.AddTranslateOp()
        self.translate_op.Set(Gf.Vec3d(self.lane_x, self.y_min, 0.0))

        if layer is not None and self.visual:
            from warehouse_nvidia import FORKLIFT_USD

            layer.add(
                "forklift_visual", FORKLIFT_USD,
                self.root_path + "/Visual",
                (0.0, -0.20, 0.0), (1.16, 2.23, 1.61),
                mode="uniform_xy", anchor="floor", parent=self.root_path,
            )

    def activate(self):
        """Author the kinematic body after world.reset().

        PhysX is much happier with a kinematic body added to an already
        initialised scene than with one that is reset along with the AMRs.
        The visual child was measured before reset; only the collider and the
        rigid-body marker are added here.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        if not stage.GetPrimAtPath(self.root_path):
            return
        if stage.GetPrimAtPath(self.body_path):
            return
        box = UsdGeom.Cube.Define(stage, self.body_path)
        box.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(box.GetPrim())
        self.body_translate_op = bxf.AddTranslateOp()
        self.body_translate_op.Set(Gf.Vec3d(self.lane_x, self.y, 0.45))
        bxf.AddScaleOp().Set(Gf.Vec3f(0.90, 1.60, 0.70))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        body = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
        body.CreateKinematicEnabledAttr(True)

    def debug_dump(self, tag):
        """Print the authored root transform and the visual world bounds."""
        import os
        if not os.environ.get("FORKLIFT_DEBUG"):
            return
        from pxr import Usd, UsdGeom
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        root = stage.GetPrimAtPath(self.root_path)
        if not root:
            print(f"[forkdbg] {tag} no root", flush=True)
            return
        xf = UsdGeom.Xformable(root)
        print(
            f"[forkdbg] {tag} root_y={self.y:.2f} xf={xf.GetLocalTransformation()}",
            flush=True,
        )
        vis = stage.GetPrimAtPath(self.root_path + "/Visual")
        if vis:
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True
            )
            rng = cache.ComputeWorldBound(vis).ComputeAlignedRange()
            print(
                f"[forkdbg] {tag} vis_min={tuple(round(v, 2) for v in rng.GetMin())} "
                f"vis_max={tuple(round(v, 2) for v in rng.GetMax())}",
                flush=True,
            )

    def update(self, dt, t, amr_poses):
        """Ping-pong route with an AMR-proximity safety hold.

        Motion is integrated incrementally, so a hold never turns into a
        teleport when the obstacle clears.
        """
        if self.translate_op is None:
            return
        cycle = 2.0 * (self._travel_time + self.pause)
        tt = t % max(1.0, cycle)
        if tt < self._travel_time:
            target, direction = self.y_max, 1.0
        elif tt < self._travel_time + self.pause:
            target, direction = self.y_max, 0.0
        elif tt < 2.0 * self._travel_time + self.pause:
            target, direction = self.y_min, -1.0
        else:
            target, direction = self.y_min, 0.0

        step = min(abs(target - self.y), self.speed * max(dt, 0.0))
        next_y = self.y + direction * step

        # Hold for robots in the swept corridor ahead, and for any robot that
        # is already alongside the mast.  The kinematic forklift must never
        # push an AMR, so the check is deliberately conservative and covers
        # the current, midpoint and next body poses.
        def blocked_at(y):
            for px, py in amr_poses:
                if px is None or py is None:
                    continue
                lateral = abs(px - self.lane_x)
                ahead = (py - y) * direction if direction else 0.0
                in_corridor = (
                    lateral < 1.45
                    and -0.35 < ahead < self.safety_radius
                )
                alongside = math.hypot(px - self.lane_x, py - y) < 1.35
                if in_corridor or alongside:
                    return True
            return False

        blocked = (
            blocked_at(self.y)
            or blocked_at((self.y + next_y) * 0.5)
            or blocked_at(next_y)
        )
        if blocked:
            self._hold_time += dt
        else:
            self._hold_time = 0.0
            self.y = next_y
        self.translate_op.Set(Gf.Vec3d(self.lane_x, self.y, 0.0))
        if self.body_translate_op is not None:
            self.body_translate_op.Set(Gf.Vec3d(self.lane_x, self.y, 0.45))

    def pose(self):
        return (self.lane_x, self.y), 0.0


# ---------------------------------------------------------------------------
# Unitree H1 humanoid
# ---------------------------------------------------------------------------
class HumanoidWalker:
    def __init__(self, spec, visual=True):
        self.spec = spec
        self.visual = visual
        self.name = spec.get("name", "humanoid")
        self.prim_path = f"/World/{self.name.capitalize()}"
        self.policy = None
        self.command = np.zeros(3, dtype=float)
        self.spawn = spec["spawn"]
        self.fall_resets = 0

        task = spec.get("task") or {}
        self.pickup = task.get("pickup") or {"x": -1.0, "y": 4.05}
        self.dropoff = task.get("dropoff") or {"x": 6.8, "y": 4.05}
        self.carrying = False
        self.payload_root = "/World/HumanoidPayload"
        self.pickup_box = "/World/H1PickupBox"
        self.delivered_box = "/World/H1DeliveredBox"
        self._payload_visible = False
        self._pickup_visible = True
        self._delivered_visible = False
        self._pickup_timer = 0.0
        self._drop_timer = 0.0
        self.loads = 0
        self._lane_clamps = 0

    def _lane_clamp(self, pos, quat):
        """Keep the H1 centred in the handoff lane near a conveyor.

        The flat-terrain policy drifts a little when it turns in place.  At a
        handoff station that drift used to walk the robot into the conveyor
        collider.  The lane is only clamped within 1.9 m of a station, so the
        normal walking path is untouched.
        """
        near = any(
            abs(float(pos[0]) - float(station["x"])) < 1.9
            for station in (self.pickup, self.dropoff)
        )
        if not near:
            return pos, quat
        lane_y = float(self.pickup["y"])
        if abs(float(pos[1]) - lane_y) <= 0.10:
            return pos, quat
        corrected = np.array(pos, dtype=float, copy=True)
        corrected[1] = lane_y
        self.policy.robot.set_world_pose(
            position=corrected, orientation=np.asarray(quat, dtype=float)
        )
        self.policy.robot.set_world_velocity(np.zeros(6))
        self._lane_clamps += 1
        if self._lane_clamps <= 4:
            print(
                f"[humanoid] lane clamp y {float(pos[1]):.2f} -> {lane_y:.2f}",
                flush=True,
            )
        return corrected, quat

    def create(self):
        from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy

        self.policy = H1FlatTerrainPolicy(
            prim_path=self.prim_path,
            name=self.name,
            position=np.array(
                [self.spawn["x"], self.spawn["y"], 1.05], dtype=float
            ),
            orientation=yaw_quat(float(self.spawn.get("yaw", 0.0))),
        )

    def initialize(self):
        self.policy.initialize()

    def set_command(self, vx, vy, wz):
        self.command = np.array(
            [
                float(np.clip(vx, -1.0, 1.0)),
                float(np.clip(vy, -0.5, 0.5)),
                float(np.clip(wz, -1.0, 1.0)),
            ],
            dtype=float,
        )

    def create_payload(self, layer):
        """Add the carried tote and the two conveyor totes (NVIDIA visuals)."""
        from pxr import Gf, UsdGeom

        from isaacsim.core.utils.stage import get_current_stage
        from warehouse_nvidia import PAYLOAD_USD

        stage = get_current_stage()
        root = UsdGeom.Xform.Define(stage, self.payload_root)
        xf = UsdGeom.Xformable(root)
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
        xf.AddRotateZOp().Set(0.0)

        layer.add(
            "h1_payload", PAYLOAD_USD, self.payload_root + "/Visual",
            (0.0, 0.0, 0.0), (0.42, 0.32, 0.28),
            mode="exact", anchor="center", parent=self.payload_root,
        )
        # A08 roller-bed top after fitting (see warehouse_process).  The H1
        # handoff totes must rest on the belt, not hover 15 cm above it.
        belt_top = 0.769
        layer.add(
            "h1_pickup_box", PAYLOAD_USD, self.pickup_box,
            (float(self.pickup["x"]), float(self.pickup["y"]) + 0.70, belt_top),
            (0.42, 0.32, 0.28), mode="exact", anchor="floor",
        )
        layer.add(
            "h1_delivered_box", PAYLOAD_USD, self.delivered_box,
            (float(self.dropoff["x"]), float(self.dropoff["y"]) + 0.70, belt_top),
            (0.42, 0.32, 0.28), mode="exact", anchor="floor",
        )

    # -- visibility helpers ----------------------------------------------
    def _set_visible(self, path, visible):
        from pxr import UsdGeom

        from isaacsim.core.utils.stage import get_current_stage

        prim = get_current_stage().GetPrimAtPath(path)
        if prim and prim.IsValid():
            if visible:
                UsdGeom.Imageable(prim).MakeVisible()
            else:
                UsdGeom.Imageable(prim).MakeInvisible()

    def after_assets(self):
        self._set_visible(self.payload_root, False)
        self._set_visible(self.pickup_box, True)
        self._set_visible(self.delivered_box, False)

    def _move_payload(self, pos, quat):
        from pxr import Gf, UsdGeom

        from isaacsim.core.utils.stage import get_current_stage

        fwd = quat_rotate(quat, np.array([0.54, 0.0, 0.0]))
        target = np.asarray(pos, dtype=float) + fwd + np.array([0.0, 0.0, 0.08])
        prim = get_current_stage().GetPrimAtPath(self.payload_root)
        if not prim or not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        xf.GetTranslateOp().Set(Gf.Vec3d(*[float(v) for v in target]))
        xf.GetRotateZOp().Set(float(quat_yaw(quat)))

    def update(self, dt):
        if self.policy is None:
            return
        self.policy.forward(dt, self.command)
        pos, quat = self.policy.robot.get_world_pose()
        pos = np.asarray(pos, dtype=float)
        quat = np.asarray(quat, dtype=float)
        pos, quat = self._lane_clamp(pos, quat)

        if float(pos[2]) < 0.35:
            # The flat-terrain policy can lose balance after a hard shove.
            # Recover deterministically instead of leaving a fallen robot.
            self.fall_resets += 1
            self.policy.robot.set_world_pose(
                position=np.array(
                    [self.spawn["x"], self.spawn["y"], 1.05], dtype=float
                ),
                orientation=yaw_quat(float(self.spawn.get("yaw", 0.0))),
            )
            self.policy.robot.set_world_velocity(np.zeros(6))
            self.command = np.zeros(3)
            if self.carrying:
                self.carrying = False
                self._set_visible(self.payload_root, False)
                self._set_visible(self.pickup_box, True)
            pos, quat = self.policy.robot.get_world_pose()
            pos = np.asarray(pos, dtype=float)
            quat = np.asarray(quat, dtype=float)

        if self.carrying:
            self._move_payload(pos, quat)

        # Proximity handoff.  The H1 node stops at the stations, so a small
        # dwell is enough to make the transfer read on screen.
        near = math.hypot(
            pos[0] - float(self.pickup["x"]), pos[1] - float(self.pickup["y"])
        )
        near_drop = math.hypot(
            pos[0] - float(self.dropoff["x"]), pos[1] - float(self.dropoff["y"])
        )
        stopped = abs(float(self.command[0])) < 0.08
        if not self.carrying and near < 0.85 and stopped:
            self._pickup_timer += dt
            if self._pickup_timer > 0.9:
                self.carrying = True
                self.loads += 1
                self._pickup_timer = 0.0
                self._drop_timer = 0.0
                self._set_visible(self.payload_root, True)
                self._set_visible(self.pickup_box, False)
                self._set_visible(self.delivered_box, False)
                self._move_payload(pos, quat)
                print(f"[humanoid] picked tote #{self.loads}", flush=True)
        elif not self.carrying:
            self._pickup_timer = 0.0

        if self.carrying and near_drop < 0.85 and stopped:
            self._drop_timer += dt
            if self._drop_timer > 0.9:
                self.carrying = False
                self._drop_timer = 0.0
                self._set_visible(self.payload_root, False)
                self._set_visible(self.delivered_box, True)
                print(f"[humanoid] delivered tote #{self.loads}", flush=True)
        elif self.carrying:
            self._drop_timer = 0.0

    def pose(self):
        if self.policy is None:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        pos, quat = self.policy.robot.get_world_pose()
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
