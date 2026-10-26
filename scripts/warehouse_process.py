#!/usr/bin/env python3
"""Industrial material-handling process for the warehouse demo.

This module adds the *process* layer that turns the moving robots into a
warehouse operation:

* a Universal Robots UR10 arm palletises totes from the staging conveyor onto
  an output pallet (NVIDIA UR10 + NVIDIA tote/pallet assets),
* totes visibly travel along the three AMR conveyors,
* each AMR carries an NVIDIA tote between stations (shown while it is in
  transit and hidden while it dwells on a station pad).

The arm motion and grasp are deterministic and self-contained (it uses the
UR10 RMPflow pick-place controller and a render-only attached tote).  Nothing
here changes the fleet planner's task model, so the ROS 2 autonomy stack keeps
running unchanged.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom

from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

from warehouse_nvidia import PAYLOAD_USD
from warehouse_robots import set_prim_pose

UR10_USD = "Isaac/Robots/UniversalRobots/ur10/ur10.usd"


# ---------------------------------------------------------------------------
# small USD helpers
# ---------------------------------------------------------------------------
def _define_xform(stage, path, position):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )
    return prim


def _set_xform_position(path, position):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    op = xf.GetTranslateOp()
    if op is None:
        op = xf.AddTranslateOp()
    op.Set(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )


def _set_visible(path, visible):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    if visible:
        UsdGeom.Imageable(prim).MakeVisible()
    else:
        UsdGeom.Imageable(prim).MakeInvisible()


def _disable_collisions(root_path):
    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    n = 0
    from pxr import UsdPhysics

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            n += 1
    return n


def _add_visual(layer, kind, rel, wrapper, size, materials, color,
                visual=True, anchor="center"):
    """Add an NVIDIA visual under ``wrapper`` or a procedural box fallback.

    The wrapper is the moving Xform.  The visual is always measured in the
    wrapper's local frame, so ``anchor="center"`` means the wrapper marks the
    tote centre and ``anchor="floor"`` means it marks the tote bottom.
    """
    if visual and layer is not None:
        layer.add(
            kind, rel, wrapper + "/Visual", (0.0, 0.0, 0.0), size,
            mode="exact", anchor=anchor, parent=wrapper,
        )
        return
    from warehouse_scene import add_box

    stage = get_current_stage()
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in kind)
    mat = materials.tint(safe, color)
    z = 0.0 if anchor == "center" else size[2] / 2.0
    add_box(
        stage, wrapper + "/Visual", size, (0.0, 0.0, z),
        mat, collision=False,
    )


# ---------------------------------------------------------------------------
# Animated totes on the AMR conveyors
# ---------------------------------------------------------------------------
# Top of the A08 roller bed after it is fitted to the configured conveyor
# footprint.  Loads must sit on this surface, not a guessed clearance above
# the floor; this is the single source of truth for belt height.
CONVEYOR_BELT_TOP = 0.769


class ConveyorTotes:
    """Render-only totes that travel along the three AMR conveyors."""

    def __init__(self, cfg, materials, visual=True, count_per_belt=2):
        self.cfg = cfg
        self.materials = materials
        self.visual = visual
        self.count = max(1, int(count_per_belt))
        self.belts = []
        self.items = []
        self.speed = 0.16

    def create(self, layer):
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, "/World/Process")
        conveyors = self.cfg["scene"].get("conveyors", [])
        for i, conv in enumerate(conveyors):
            # Leave the two humanoid handoff conveyors for the H1 transfer;
            # they already have their own static pickup / delivery totes.
            # The staging conveyor belongs to the UR10 cell: the pick tote is
            # the load the arm actually grasps, so do not animate a second
            # tote through that pick point.
            if (conv["name"].startswith("humanoid_conveyor")
                    or conv["name"] == "conveyor_staging"):
                continue
            x = float(conv["x"])
            y = float(conv["y"])
            length = float(conv["size"][0])
            belt_y = y
            self.belts.append((x, belt_y, length))
            for k in range(self.count):
                path = f"/World/Process/Belt{i}Tote{k}"
                start_x = x - length / 2.0 + 0.25
                _define_xform(stage, path, (start_x, belt_y, CONVEYOR_BELT_TOP))
                _add_visual(
                    layer, "belt_tote", PAYLOAD_USD, path,
                    (0.34, 0.26, 0.24), self.materials,
                    (0.18, 0.50, 0.95), self.visual, anchor="floor",
                )
                self.items.append(
                    {
                        "path": path,
                        "belt": i,
                        "phase": k / float(self.count),
                    }
                )
        return self

    def update(self, dt, t):
        if not self.items:
            return
        for item in self.items:
            x0, y, length = self.belts[item["belt"]]
            span = max(0.2, length - 0.50)
            u = (t * self.speed / span + item["phase"]) % 1.0
            x = x0 - span / 2.0 + u * span
            _set_xform_position(item["path"], (x, y, CONVEYOR_BELT_TOP))


# ---------------------------------------------------------------------------
# UR10 palletising cell
# ---------------------------------------------------------------------------
class IndustrialCell:
    """UR10 picks totes from the staging conveyor and stacks them on a pallet."""

    def __init__(self, spec, materials, assets_root, layer, visual=True):
        self.spec = spec
        self.materials = materials
        self.assets_root = assets_root
        self.layer = layer
        self.visual = visual
        self.enabled = bool(spec.get("enabled", True))

        self.cell_path = "/World/IndustrialCell"
        self.robot_path = self.cell_path + "/UR10"
        self.articulation_path = self.robot_path
        self.base = spec.get(
            "base", {"x": 4.2, "y": 3.0, "z": 0.90, "yaw": math.pi}
        )
        self.pick = np.array(
            [spec.get("pick", {}).get("x", 3.35),
             spec.get("pick", {}).get("y", 3.0),
             spec.get("pick", {}).get("z", 0.909)],
            dtype=float,
        )
        self.place = np.array(
            [spec.get("place", {}).get("x", 4.6),
             spec.get("place", {}).get("y", 2.3),
             spec.get("place", {}).get("z", 0.29)],
            dtype=float,
        )
        self.tote_size = tuple(spec.get("tote_size", (0.42, 0.32, 0.28)))
        self.stack_size = int(spec.get("stack_size", 3))
        self.stack_step = float(spec.get("stack_step", 0.30))
        self.dwell = float(spec.get("dwell", 2.5))
        self.reset_delay = float(spec.get("reset_delay", 4.0))
        self.initial_height = float(spec.get("initial_height", 1.45))

        self.manip = None
        self.ctrl = None
        self.ac = None
        self.attached = False
        self.attach_offset = np.zeros(3)
        self.event = -1
        self.state = "pick"
        self.state_t = 0.0
        self.stack_index = 0
        self.cycles = 0
        self.last_tote_pos = None
        self.pick_tote_path = self.cell_path + "/PickTote"
        self.output_paths = [
            self.cell_path + f"/OutputTote{i}" for i in range(self.stack_size)
        ]
        self.output_used = [False] * self.stack_size
        self._last_status = ""

    # -- authoring (before world.reset) -----------------------------------
    def create(self):
        if not self.enabled:
            return
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, self.cell_path)

        # Pick tote and the output stack wrappers.  The NVIDIA visual is a
        # child of each wrapper, so moving the wrapper is a single translate
        # update (safe while the sim runs).
        _define_xform(stage, self.pick_tote_path, self.pick)
        _add_visual(
            self.layer, "pick_tote", PAYLOAD_USD, self.pick_tote_path,
            self.tote_size, self.materials, (0.95, 0.62, 0.12),
            self.visual, anchor="center",
        )
        for i, path in enumerate(self.output_paths):
            _define_xform(
                stage, path,
                (self.place[0], self.place[1], self.place[2] + i * self.stack_step),
            )
            _add_visual(
                self.layer, f"output_tote{i}", PAYLOAD_USD, path,
                self.tote_size, self.materials, (0.95, 0.62, 0.12),
                self.visual, anchor="center",
            )

        # UR10 on its pedestal, posed before world.reset().
        add_reference_to_stage(
            usd_path=self.assets_root + "/" + UR10_USD,
            prim_path=self.robot_path,
        ).GetVariantSet("Gripper").SetVariantSelection("Short_Suction")
        prim = stage.GetPrimAtPath(self.robot_path)
        if prim and prim.IsValid():
            set_prim_pose(
                prim,
                (self.base["x"], self.base["y"], self.base.get("z", 0.90)),
                float(self.base.get("yaw", math.pi)),
            )
        # The UR10 asset puts its ArticulationRootAPI on the fixed root joint,
        # not on the wrapper Xform.  OmniGraph sensors need the actual root.
        from pxr import UsdPhysics

        self.articulation_path = self.robot_path
        if prim and prim.IsValid():
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                    self.articulation_path = str(child.GetPath())
                    break
        print(
            f"[ur10] articulation root: {self.articulation_path}",
            flush=True,
        )
        # The arm is position-controlled and must not jam against the conveyor
        # collider while it reaches over the belt.
        _disable_collisions(self.robot_path)
        return self

    def after_assets(self):
        if not self.enabled:
            return
        _set_visible(self.pick_tote_path, True)
        for path in self.output_paths:
            _set_visible(path, False)

    # -- initialisation (after world.reset) -------------------------------
    def initialize(self):
        if not self.enabled:
            return
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import SurfaceGripper
        from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import (
            PickPlaceController,
        )

        class _VisualSuctionGripper(SurfaceGripper):
            """Surface gripper whose state is driven by the visual tote.

            The totes in this demo are render-only, so the PhysX surface
            gripper never finds a collider to attach to and logs
            "gripper didn't close successfully".  The grasp itself is
            represented by the tote following the end effector, so report a
            clean close/open state to the controller instead.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._visual_closed = False

            def close(self):
                self._visual_closed = True

            def open(self):
                self._visual_closed = False

            def is_closed(self):
                return bool(self._visual_closed)

            def is_open(self):
                return not bool(self._visual_closed)

        gripper = _VisualSuctionGripper(
            end_effector_prim_path=self.robot_path + "/ee_link",
            surface_gripper_path=self.robot_path + "/ee_link/SurfaceGripper",
        )
        self.manip = SingleManipulator(
            prim_path=self.robot_path,
            name="ur10",
            end_effector_prim_path=self.robot_path + "/ee_link",
            gripper=gripper,
        )
        self.manip.initialize()
        from pxr import UsdPhysics
        prim = get_current_stage().GetPrimAtPath(self.robot_path)
        # Park in the well-conditioned elbow-up configuration used by the
        # UR10 example before the RMPflow controller takes over.
        try:
            self.manip.set_joint_positions(
                positions=np.array(
                    [-math.pi / 2, -math.pi / 2, -math.pi / 2,
                     -math.pi / 2, math.pi / 2, 0.0]
                )
            )
            self.manip.set_joint_velocities(np.zeros(6))
        except Exception:
            pass
        self.manip.gripper.set_default_state(opened=True)
        self.ctrl = PickPlaceController(
            name="ur10_palletize",
            gripper=self.manip.gripper,
            robot_articulation=self.manip,
        )
        self.ctrl._h1 = self.initial_height
        self.ac = self.manip.get_articulation_controller()
        self._start_pick()

    # -- state machine ----------------------------------------------------
    def _start_pick(self):
        self.state = "pick"
        self.state_t = 0.0
        self.event = -1
        self.attached = False
        self.attach_offset = np.zeros(3)
        if self.ctrl is not None:
            self.ctrl.reset()
            self.ctrl._h1 = self.initial_height
        _set_xform_position(self.pick_tote_path, self.pick)
        _set_visible(self.pick_tote_path, True)
        if self.stack_index >= self.stack_size:
            self._clear_stack()

    def _clear_stack(self):
        for i, path in enumerate(self.output_paths):
            _set_visible(path, False)
            _set_xform_position(
                path,
                (self.place[0], self.place[1],
                 self.place[2] + i * self.stack_step),
            )
            self.output_used[i] = False
        self.stack_index = 0

    def _ee_pose(self):
        if self.manip is None:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        pos, quat = self.manip.end_effector.get_world_pose()
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)

    def _place_target(self):
        z = self.place[2] + self.stack_index * self.stack_step
        return np.array([self.place[0], self.place[1], z], dtype=float)

    def pre_step(self, dt, t):
        """Run the pick-place controller before the physics step."""
        if not self.enabled or self.manip is None:
            return
        self.state_t += dt
        if self.state == "pick":
            if self.ctrl.is_done():
                self.state = "dwell"
                self.state_t = 0.0
                return
            place = self._place_target()
            # `pick`/`place` are the centres of the totes.  The controller
            # drives the end effector to the *top* of each tote, so the
            # suction cup lands on the lid instead of descending into the
            # middle of the load.
            half_h = float(self.tote_size[2]) * 0.5
            try:
                action = self.ctrl.forward(
                    picking_position=self.pick + np.array([0.0, 0.0, half_h]),
                    placing_position=place + np.array([0.0, 0.0, half_h]),
                    current_joint_positions=self.manip.get_joint_positions(),
                    end_effector_offset=np.array([0.0, 0.0, 0.06]),
                )
            except Exception as exc:
                self._log_status(f"controller error: {exc}")
                self.state = "dwell"
                self.state_t = 0.0
                return
            self.ac.apply_action(action)
            ev = int(self.ctrl.get_current_event())
            if ev != self.event:
                if ev >= 3 and not self.attached and ev < 7:
                    # Fixed geometric offset: the end effector is driven to
                    # the tote lid, so the tote centre is half a tote plus
                    # the controller's 6 cm clearance below it.  Capturing
                    # (pick - ee) here was sensitive to RMPflow lag and made
                    # the carried tote/placed stack end up in mid-air.
                    self.attach_offset = np.array(
                        [0.0, 0.0, -0.5 * float(self.tote_size[2]) - 0.06]
                    )
                    self.attached = True
                    _set_visible(self.pick_tote_path, True)
                    self._log_status("grasp")
                if ev >= 7 and self.attached:
                    self._release()
                self.event = ev
        elif self.state == "dwell":
            if self.state_t >= self.dwell:
                self.stack_index += 1
                if self.stack_index >= self.stack_size:
                    self.state = "full"
                    self.state_t = 0.0
                else:
                    self._start_pick()
        elif self.state == "full":
            if self.state_t >= self.reset_delay:
                self._clear_stack()
                self.cycles += 1
                self._start_pick()

    def _release(self):
        # Snap the finished tote to the pallet stack rather than to the
        # controller's instantaneous end-effector pose.  RMPflow can still be
        # a few centimetres short of the placing target on the release tick,
        # and using that pose left totes suspended in the air.
        tote_pos = self._place_target()
        idx = min(self.stack_index, self.stack_size - 1)
        path = self.output_paths[idx]
        _set_xform_position(path, tote_pos)
        _set_visible(path, True)
        self.output_used[idx] = True
        _set_visible(self.pick_tote_path, False)
        self.attached = False
        self.last_tote_pos = None
        self._log_status(f"placed stack {idx + 1}/{self.stack_size}")

    def post_step(self, dt, t):
        """Move the carried tote with the gripper and update status."""
        if not self.enabled or self.manip is None:
            return
        if self.attached:
            ee, _ = self._ee_pose()
            self.last_tote_pos = ee + self.attach_offset
            _set_xform_position(self.pick_tote_path, self.last_tote_pos)
            _set_visible(self.pick_tote_path, True)

    def _log_status(self, tag):
        if tag != self._last_status:
            self._last_status = tag
            print(f"[ur10] {tag}", flush=True)

    def status(self):
        if not self.enabled:
            return "off"
        if self.manip is None:
            return "loading"
        return (
            f"{self.state} stack {len([u for u in self.output_used if u])}/"
            f"{self.stack_size} cycles {self.cycles}"
        )
