#!/usr/bin/env python3
"""Central fleet manager: task allocation, A* planning and traffic control.

This node is deliberately small and dependency-light (rclpy + numpy + yaml).
It does not use Nav2 or a map server: the static map is generated from the
same YAML the simulator builds, and every AMR receives a ``nav_msgs/Path``.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray

from .config_utils import load_config, stations_by_name
from .grid_map import GridMap, wrap_angle, yaw_from_quaternion

STATIONARY = ("PICKING", "DROPPING")
STATE_COLORS = {
    "IDLE": (0.55, 0.55, 0.55),
    "PARKING": (0.55, 0.40, 0.95),
    "TO_PICKUP": (0.15, 0.65, 1.0),
    "PICKING": (0.10, 0.90, 0.35),
    "TO_DROPOFF": (1.0, 0.65, 0.10),
    "DROPPING": (1.0, 0.30, 0.15),
}


def yaw_quat(yaw):
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


class ManagedAMR:
    def __init__(self, spec):
        self.name = spec["name"]
        self.spec = spec
        self.role = str(spec.get("role", "warehouse AMR"))
        self.preferred = [str(v) for v in spec.get("preferred_templates", [])]
        self.right_of_way = int(spec.get("right_of_way", 99))
        self.max_speed = float(spec.get("max_speed", 0.65))
        self.pose = None
        self.yaw = 0.0
        self.odom_time = 0.0
        self.path = []
        self.goal = None
        self.path_time = 0.0
        self.task = None
        self.state = "IDLE"
        self.state_since = time.time()
        self.blocked = False
        self.remaining = 0.0
        self.nav_time = 0.0
        self.hold = False
        self.hold_reason = ""
        self.hold_since = 0.0
        self.last_swap = 0.0
        self.forced_hold_until = 0.0
        self.last_plan = 0.0
        self.last_status = None
        self.park = None
        self.waiting_for = None
        self.wait_goal = None
        self.wait_goal_station = None
        self.loaded = False
        self.pub_path = None
        self.pub_hold = None
        self.pub_scale = None
        self.pub_load = None
        self.pub_mode = None
        self.last_hold_pub = None
        self.last_load_pub = None
        self.agent_mode = "AUTO"

    @property
    def fresh(self):
        return self.pose is not None and (time.time() - self.odom_time) < 2.0

    def set_hold(self, reason):
        if not self.hold:
            self.hold_since = time.time()
        self.hold = True
        self.hold_reason = reason


class FleetManager(Node):
    def __init__(self):
        super().__init__("fleet_manager")
        self.declare_parameter("config", "")
        self.declare_parameter("publish_markers", True)
        cfg = load_config(self.get_parameter("config").value)
        self.cfg = cfg
        self.stations = stations_by_name(cfg)
        self.grid = GridMap(cfg)

        fleet = cfg["fleet"]
        self.replan_period = float(fleet.get("replan_period", 2.0))
        self.pickup_dwell = float(fleet.get("pickup_dwell", 2.5))
        self.dropoff_dwell = float(fleet.get("dropoff_dwell", 2.5))
        self.arrival_tol = float(fleet.get("arrival_tolerance", 0.35))
        self.hold_distance = float(fleet.get("hold_distance", 1.45))
        self.release_distance = float(fleet.get("release_distance", 2.10))
        self.max_hold = float(fleet.get("max_hold_time", 9.0))
        self.max_active = max(1, int(fleet.get("max_active_amrs", 1)))
        self.status_period = float(fleet.get("status_period", 1.0))
        self.station_claim_radius = float(fleet.get("station_claim_radius", 3.0))
        self.station_wait_distance = float(fleet.get("station_wait_distance", 1.9))

        self.amrs = [ManagedAMR(spec) for spec in cfg["amrs"]]
        self.by_name = {a.name: a for a in self.amrs}
        # Only one robot may dwell on a station pad at a time.  Names map to
        # the robot that currently owns the pad.
        self.station_owner = {}
        self.station_release_pending = {}
        self.dynamic_obstacles = {}
        parking = cfg["scene"].get("parking", [])
        for i, amr in enumerate(self.amrs):
            amr.park = (
                (float(parking[i]["x"]), float(parking[i]["y"]))
                if i < len(parking) else None
            )

        task_cfg = cfg["tasks"]
        self.extra_counter = 0
        self.managed = bool(task_cfg.get("managed", False))
        self.task_templates = [dict(item) for item in task_cfg.get("list", [])]
        self.tasks = []
        if not self.managed:
            for item in self.task_templates:
                self.tasks.append(self._new_task(item, manual=False))
        else:
            self.initial_inventory = {
                str(k): int(v)
                for k, v in task_cfg.get("initial_inventory", {}).items()
            }
            self.capacity = {
                str(k): int(v)
                for k, v in task_cfg.get("capacity", {}).items()
            }
            # Every discovered stock location gets a capacity entry so the
            # dashboard and dispatcher agree even if a key was omitted.
            for item in self.task_templates:
                for key in list(item.get("requires", {})) + list(
                    item.get("produces", {})
                ):
                    self.initial_inventory.setdefault(str(key), 0)
                    self.capacity.setdefault(str(key), 99)
            self.inventory = dict(self.initial_inventory)
            self.incoming = {k: 0 for k in self.capacity}
            self.max_pending = max(1, int(task_cfg.get("max_pending", 8)))
            self.supplier_period = max(1.0, float(task_cfg.get("supplier_period", 14.0)))
            self.shipping_period = max(1.0, float(task_cfg.get("shipping_period", 10.0)))
            self.last_supply = time.time()
            self.last_ship = time.time()
            self.flow = {
                "received": 0,
                "shipped": 0,
                "completed": 0,
                "replenished": 0,
            }
            self.task_counter = 0
        if not self.managed:
            self.initial_inventory = {}
            self.capacity = {}
            self.inventory = {}
            self.incoming = {}
            self.max_pending = 0
            self.supplier_period = 0.0
            self.shipping_period = 0.0
            self.last_supply = 0.0
            self.last_ship = 0.0
            self.flow = {"received": 0, "shipped": 0, "completed": 0, "replenished": 0}
            self.task_counter = 0
        self.loop_tasks = bool(task_cfg.get("loop", True))
        self.cycle = 0
        self.completed = 0
        self.all_done_since = None
        self.extra_counter = 0

        # Global operating mode and per-robot teleoperation ownership.
        self.mode = "AUTO"           # AUTO | PAUSED | ESTOP
        self.teleop = set()
        self.last_agent_mode_sync = 0.0

        self.pub_status = self.create_publisher(String, "fleet/status", 10)
        self.pub_events = self.create_publisher(String, "fleet/task_events", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "fleet/markers", 10)
        self.pub_map = self.create_publisher(OccupancyGrid, "fleet/map", 1)
        self.pub_mode = self.create_publisher(String, "fleet/mode", 10)
        self.create_subscription(String, "fleet/submit_task", self.on_submit_task, 10)
        self.create_subscription(String, "fleet/control", self.on_control, 20)
        self.create_subscription(
            Odometry, "forklift/odom",
            lambda msg: self.on_dynamic_odom("forklift", msg), 10,
        )
        self.create_subscription(
            Odometry, "humanoid/odom",
            lambda msg: self.on_dynamic_odom("humanoid", msg), 10,
        )

        for amr in self.amrs:
            amr.pub_path = self.create_publisher(Path, f"{amr.name}/path", 10)
            amr.pub_hold = self.create_publisher(Bool, f"{amr.name}/hold", 10)
            amr.pub_scale = self.create_publisher(Float32, f"{amr.name}/speed_scale", 10)
            amr.pub_load = self.create_publisher(Twist, f"{amr.name}/load_state", 10)
            amr.pub_mode = self.create_publisher(String, f"{amr.name}/mode", 10)
            self.create_subscription(
                Odometry, f"{amr.name}/odom",
                lambda msg, a=amr: self.on_odom(a, msg), 10,
            )
            self.create_subscription(
                String, f"{amr.name}/nav_status",
                lambda msg, a=amr: self.on_nav_status(a, msg), 10,
            )

        self.last_status = 0.0
        self.last_markers = 0.0
        self.last_map = 0.0
        self.create_timer(0.1, self.tick)
        if self.managed:
            self._dispatch_tasks(time.time())
        self._publish_map()
        self._publish_mode()
        self.get_logger().info(
            f"Fleet manager ready: {len(self.amrs)} AMRs, "
            f"{len(self.tasks)} queued tasks, "
            f"mode={self.mode}, managed={'on' if self.managed else 'off'}, "
            f"map {self.grid.nx}x{self.grid.ny} @ {self.grid.res:.2f} m"
        )
        self.publish_event(
            "fleet_started",
            {
                "amrs": len(self.amrs),
                "managed": self.managed,
                "mode": self.mode,
            },
        )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------
    def on_odom(self, amr, msg):
        p = msg.pose.pose.position
        amr.pose = (float(p.x), float(p.y))
        amr.yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        amr.odom_time = time.time()

    def on_dynamic_odom(self, name, msg):
        p = msg.pose.pose.position
        self.dynamic_obstacles[name] = (
            (float(p.x), float(p.y)),
            time.time(),
        )

    def on_nav_status(self, amr, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        amr.blocked = bool(data.get("blocked", False))
        amr.remaining = float(data.get("remaining", 0.0))
        amr.nav_time = time.time()

    def _new_task(self, item, manual=False, task_id=None):
        template = None if manual else str(item.get("id", ""))
        return {
            "id": task_id or str(item.get("id", f"M{self.extra_counter}")),
            "template": template,
            "pickup": item["pickup"],
            "dropoff": item["dropoff"],
            "priority": int(item.get("priority", 50)),
            "cargo": str(item.get("cargo", "tote")),
            "description": str(item.get("description", "")),
            "requires": dict(item.get("requires", {})),
            "produces": dict(item.get("produces", {})),
            "manual": bool(manual),
            "reserved": False,
            "status": "PENDING",
            "assigned": None,
            "created": time.time(),
            "completed": None,
        }

    def on_submit_task(self, msg):
        try:
            if msg.data.strip().startswith("{"):
                data = json.loads(msg.data)
            else:
                parts = msg.data.split()
                data = {"pickup": parts[0], "dropoff": parts[1]}
            pickup = data["pickup"]
            dropoff = data["dropoff"]
        except Exception as exc:
            self.get_logger().warn(f"submit_task ignored: {exc}")
            return
        if pickup not in self.stations or dropoff not in self.stations:
            self.get_logger().warn(
                f"submit_task ignored: unknown station ({pickup} -> {dropoff})"
            )
            return
        self.extra_counter += 1
        task = self._new_task(
            {
                "pickup": pickup,
                "dropoff": dropoff,
                "priority": int(data.get("priority", 50)),
                "cargo": data.get("cargo", "operator_tote"),
                "description": data.get("description", "Operator work order"),
            },
            manual=True,
            task_id=f"X{self.extra_counter}",
        )
        self.tasks.append(task)
        self.get_logger().info(f"Task {task['id']} submitted: {pickup} -> {dropoff}")
        self.publish_event("task_submitted", task)

    def on_control(self, msg):
        """Accept dashboard / operator commands on ``fleet/control``.

        JSON actions:
          {"action": "set_mode", "mode": "AUTO|PAUSED|ESTOP"}
          {"action": "set_teleop", "robot": "amr1", "enabled": true}
          {"action": "cancel_task", "robot": "amr1"}
        """
        try:
            data = json.loads(msg.data)
            action = str(data.get("action", "")).lower()
        except Exception as exc:
            self.get_logger().warn(f"fleet/control ignored: {exc}")
            return
        if action == "set_mode":
            self._set_mode(str(data.get("mode", "AUTO")).upper())
        elif action in ("estop", "stop"):
            self._set_mode("ESTOP")
        elif action in ("clear_estop", "resume"):
            self._set_mode("AUTO")
        elif action == "pause":
            self._set_mode("PAUSED")
        elif action in ("set_teleop", "teleop"):
            robot = str(data.get("robot", "")).strip("/")
            amr = self.by_name.get(robot)
            if amr is None:
                self.get_logger().warn(f"teleop ignored: unknown robot {robot}")
                return
            self._set_teleop(amr, bool(data.get("enabled", True)))
        elif action == "cancel_task":
            robot = str(data.get("robot", "")).strip("/")
            amr = self.by_name.get(robot)
            if amr is not None:
                self._cancel_task(amr, "operator cancel")
        else:
            self.get_logger().warn(f"fleet/control ignored action: {action}")

    def _set_mode(self, mode):
        if mode not in ("AUTO", "PAUSED", "ESTOP"):
            self.get_logger().warn(f"unknown fleet mode: {mode}")
            return
        if mode == self.mode:
            return
        previous = self.mode
        self.mode = mode
        if mode in ("ESTOP", "PAUSED"):
            for amr in self.amrs:
                amr.set_hold(f"fleet {mode.lower()}")
            if mode == "ESTOP":
                # E-stop is an unconditional safe state.  The operator must
                # explicitly leave teleop before driving again.
                self.teleop.clear()
        else:
            for amr in self.amrs:
                if amr.name not in self.teleop:
                    amr.hold = False
                    amr.hold_reason = ""
        for amr in self.amrs:
            self._publish_hold(amr, mode in ("ESTOP", "PAUSED"))
        self._sync_agent_modes(force=True)
        self._publish_mode()
        self.publish_event(
            "mode_changed", {"previous": previous, "mode": mode}
        )
        self.get_logger().warn(f"Fleet mode {previous} -> {mode}")

    def _set_teleop(self, amr, enabled):
        if enabled:
            if self.mode != "AUTO":
                self.get_logger().warn(
                    f"teleop for {amr.name} refused while fleet is {self.mode}"
                )
                self.publish_event(
                    "teleop_refused",
                    {"robot": amr.name, "mode": self.mode},
                )
                return
            if amr.name not in self.teleop:
                self.teleop.add(amr.name)
                self._cancel_task(amr, "teleop takeover")
                amr.state = "TELEOP"
                amr.state_since = time.time()
                amr.path = []
                amr.goal = None
                amr.hold = False
                self._publish_path(amr, [])
                self._publish_hold(amr, False)
                self.get_logger().warn(f"{amr.name} teleoperation enabled")
                self.publish_event("teleop_enabled", {"robot": amr.name})
        else:
            if amr.name in self.teleop:
                self.teleop.discard(amr.name)
                if amr.state == "TELEOP":
                    amr.state = "IDLE"
                    amr.state_since = time.time()
                amr.hold = False
                self._publish_path(amr, [])
                self.get_logger().info(f"{amr.name} teleoperation released")
                self.publish_event("teleop_disabled", {"robot": amr.name})
        self._sync_agent_modes(force=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def tick(self):
        now = time.time()
        self._sync_agent_modes()
        if self.managed:
            self._replenish(now)
            self._dispatch_tasks(now)

        if self.mode == "AUTO":
            self._assign_tasks(now)
            for amr in self.amrs:
                if amr.name not in self.teleop:
                    self._advance_state(amr, now)
            self._traffic_control(now)
        else:
            for amr in self.amrs:
                amr.set_hold(f"fleet {self.mode.lower()}")

        for amr in self.amrs:
            hold = (
                (amr.hold or amr.forced_hold_until > now or self.mode != "AUTO")
                and amr.name not in self.teleop
            )
            if hold and amr.forced_hold_until > now:
                amr.hold_reason = "deadlock swap"
            self._publish_hold(amr, hold)

        if self.mode == "AUTO":
            self._maybe_requeue(now)
            self._publish_loads()

        if now - self.last_status >= self.status_period:
            self.last_status = now
            self._publish_status(now)
        if now - self.last_map >= 5.0:
            self.last_map = now
            self._publish_map()
        if now - self.last_markers >= 1.0 and self.get_parameter("publish_markers").value:
            self.last_markers = now
            self._publish_markers(now)

    def _assign_tasks(self, now):
        if self.mode != "AUTO":
            return
        # Let AMRs that are still driving to a parking spot clear the work
        # area first.  Handing a new task to the first robot that parks can
        # send it back through the arm dock / aisle while the others are
        # still parking, which creates avoidable close-quarters deadlocks.
        if any(a.state == "PARKING" for a in self.amrs):
            return
        idle = [
            a for a in self.amrs
            if a.task is None and a.state == "IDLE" and a.fresh
            and a.name not in self.teleop
        ]
        pending = [t for t in self.tasks if t["status"] == "PENDING"]
        if not idle or not pending:
            return
        # Keep the demo's narrow station aisle predictable: only one AMR
        # executes a task at a time.  Three robots were constantly meeting
        # head-on in the 2 m corridor, and LiDAR then froze two of them
        # point-blank.  The other Carters stay visibly parked and hand over
        # as soon as the active one has parked again.
        active = [a for a in self.amrs if a.task is not None]
        slots = self.max_active - len(active)
        if slots <= 0:
            return
        for amr in idle[:slots]:
            choices = [t for t in pending if t["status"] == "PENDING"]
            if not choices:
                break
            # Travel distance dominates; role preference and priority break
            # ties.  The operator can always submit a manual work order.
            choices.sort(
                key=lambda t: (
                    0 if t.get("template") in amr.preferred else 1,
                    self._dist(amr.pose, self._station(t["pickup"]))
                    + 0.5 * int(t.get("priority", 50)),
                    int(t.get("priority", 50)),
                )
            )
            task = None
            for candidate in choices:
                if self._try_reserve(candidate):
                    task = candidate
                    break
            if task is None:
                break
            task["status"] = "ASSIGNED"
            task["assigned"] = amr.name
            task["assigned_at"] = now
            amr.task = task
            amr.state = "TO_PICKUP"
            amr.state_since = now
            amr.hold = False
            amr.loaded = False
            self._publish_load(amr)
            amr.waiting_for = None
            amr.wait_goal = None
            amr.wait_goal_station = None
            self.get_logger().info(
                f"ASSIGN {task['id']} -> {amr.name}  "
                f"{task['pickup']} -> {task['dropoff']}  "
                f"({task.get('cargo', 'tote')})"
            )
            self.publish_event(
                "task_assigned",
                {
                    "id": task["id"],
                    "robot": amr.name,
                    "pickup": task["pickup"],
                    "dropoff": task["dropoff"],
                    "cargo": task.get("cargo", "tote"),
                    "template": task.get("template"),
                },
            )
            self._plan(amr, self._station_goal(amr, task["pickup"]), now)

    def _advance_state(self, amr, now):
        if amr.name in self.teleop or self.mode != "AUTO":
            return
        if amr.task is None and amr.state != "PARKING":
            if amr.state not in ("IDLE",):
                amr.state = "IDLE"
                amr.state_since = now
            return

        if amr.state in ("TO_PICKUP", "TO_DROPOFF", "PARKING"):
            if amr.pose is None:
                return
            dest_name = None
            if amr.state == "PARKING":
                goal = amr.park
                if goal is None:
                    amr.state = "IDLE"
                    amr.state_since = now
                    return
            else:
                dest_name = (
                    amr.task["pickup"] if amr.state == "TO_PICKUP"
                    else amr.task["dropoff"]
                )
                goal = self._station_goal(amr, dest_name)
            dist = self._dist(amr.pose, goal)
            if dist <= self.arrival_tol:
                if amr.state == "PARKING":
                    amr.state = "IDLE"
                    amr.state_since = now
                    amr.path = []
                    amr.goal = None
                    self._publish_path(amr, [])
                    self.get_logger().info(f"{amr.name} parked")
                    return
                if amr.waiting_for == dest_name:
                    # Holding outside a pad that another robot is using.
                    amr.set_hold(f"waiting for {dest_name}")
                    return
                self.station_owner[dest_name] = amr.name
                amr.waiting_for = None
                amr.wait_goal = None
                amr.wait_goal_station = None
                amr.state = "PICKING" if amr.state == "TO_PICKUP" else "DROPPING"
                amr.state_since = now
                amr.hold = True
                amr.hold_since = now
                self.publish_event(
                    "arrived",
                    {"id": amr.task["id"], "robot": amr.name, "state": amr.state},
                )
                self.get_logger().info(
                    f"{amr.name} arrived at {amr.task['pickup'] if amr.state == 'PICKING' else amr.task['dropoff']}"
                )
                return

            # Replan when the local controller is blocked, when the robot has
            # drifted off the planned path, or on the periodic timer.
            need = False
            if not amr.path:
                need = True
            elif amr.blocked and now - amr.last_plan > 1.0:
                need = True
            elif amr.goal != goal:
                need = True
            elif now - amr.last_plan >= self.replan_period:
                if self._dist_to_path(amr.pose, amr.path) > 0.9:
                    need = True
            if need:
                self._plan(amr, goal, now)

        elif amr.state == "PICKING":
            amr.hold = True
            if now - amr.state_since >= self.pickup_dwell:
                self._release_station(amr, amr.task["pickup"])
                amr.task["status"] = "DELIVERING"
                amr.loaded = True
                self._publish_load(amr)
                amr.state = "TO_DROPOFF"
                amr.state_since = now
                amr.hold = False
                self.publish_event(
                    "payload_picked",
                    {
                        "id": amr.task["id"],
                        "robot": amr.name,
                        "station": amr.task["pickup"],
                        "cargo": amr.task.get("cargo", "tote"),
                    },
                )
                self._plan(
                    amr,
                    self._station_goal(amr, amr.task["dropoff"]),
                    now,
                )

        elif amr.state == "DROPPING":
            amr.hold = True
            if now - amr.state_since >= self.dropoff_dwell:
                self._release_station(amr, amr.task["dropoff"])
                task = amr.task
                amr.loaded = False
                self._publish_load(amr)
                self._complete_task(task, amr, now)
                amr.task = None
                amr.path = []
                amr.goal = None
                # If more work is queued, stay "hot" at the drop-off pad and
                # take the next assignment immediately.  Driving 11 m back to
                # park and out again between every task dominated the demo
                # cycle time.  Park only once the queue is empty.
                more_work = any(
                    t["status"] == "PENDING" for t in self.tasks
                )
                if amr.park is not None and not more_work:
                    amr.state = "PARKING"
                    amr.state_since = now
                    amr.hold = False
                    self._plan(amr, amr.park, now)
                else:
                    amr.state = "IDLE"
                    amr.state_since = now
                    amr.hold = False
                    self._publish_path(amr, [])

    def _traffic_control(self, now):
        # Clear stale holds (stationary and station-waiting holds are
        # re-applied by the state machine).
        for amr in self.amrs:
            if (
                amr.state not in STATIONARY
                and amr.waiting_for is None
                and amr.forced_hold_until <= now
            ):
                amr.hold = False
                amr.hold_reason = ""

        for i in range(len(self.amrs)):
            for j in range(i + 1, len(self.amrs)):
                a, b = self.amrs[i], self.amrs[j]
                if a.name in self.teleop or b.name in self.teleop:
                    continue
                if a.pose is None or b.pose is None:
                    continue
                # Parked Carters are stationary scenery for traffic purposes.
                # Letting them "yield" at their 2 m-spaced charging spots held
                # a returning AMR 1.2 m short of its park forever, which in
                # turn blocked task assignment.
                if a.state == "IDLE" or b.state == "IDLE":
                    continue
                d = self._dist(a.pose, b.pose)
                if d > self.release_distance:
                    continue
                if d > self.hold_distance:
                    # Inside the hysteresis band: keep the current decision.
                    continue
                # A robot waiting outside an occupied pad is already
                # stationary; the robot leaving the pad must be free to go.
                if a.waiting_for is not None or b.waiting_for is not None:
                    waiting = a if a.waiting_for is not None else b
                    other = b if waiting is a else a
                    waiting.set_hold(f"waiting for {waiting.waiting_for}")
                    if other.forced_hold_until <= now:
                        other.hold = False
                        other.hold_reason = ""
                    continue

                yielder = None
                if a.state in STATIONARY and b.state not in STATIONARY:
                    yielder = b
                elif b.state in STATIONARY and a.state not in STATIONARY:
                    yielder = a
                elif a.state not in STATIONARY and b.state not in STATIONARY:
                    yielder = a if a.right_of_way > b.right_of_way else b
                if yielder is None:
                    continue
                other = b if yielder is a else a
                # Only yield to something that is actually in front of you.
                # Two robots passing side by side (or one that has already
                # moved behind) must not freeze each other.
                if not self._is_ahead(yielder, other):
                    if self._is_ahead(other, yielder):
                        yielder, other = other, yielder
                    else:
                        continue
                yielder.set_hold(f"yield to {other.name}")
                # Deadlock guard: if the same robot has yielded too long,
                # swap the right of way once and let the other one wait.
                if (
                    now - yielder.hold_since > self.max_hold
                    and now - yielder.last_swap > 2.0
                    and other.state not in STATIONARY
                ):
                    yielder.last_swap = now
                    yielder.forced_hold_until = now
                    other.forced_hold_until = now + 5.0
                    yielder.hold = False
                    yielder.hold_since = now
                    self.get_logger().warn(
                        f"traffic deadlock {a.name}/{b.name}: swapping right of way"
                    )
                    self.publish_event(
                        "traffic_swap", {"a": a.name, "b": b.name}
                    )

    def _plan(self, amr, goal, now):
        other_points = [o.pose for o in self.amrs if o is not amr and o.pose]
        other_points.extend(self._dynamic_points(now))
        path = self.grid.astar(amr.pose, goal, soft_points=other_points)
        if not path:
            path = self.grid.astar(amr.pose, goal)
        amr.last_plan = now
        if not path:
            amr.path = []
            amr.goal = goal
            amr.set_hold("no path")
            self.get_logger().warn(
                f"{amr.name}: no path from {tuple(round(v, 2) for v in amr.pose)} "
                f"to {tuple(round(v, 2) for v in goal)}"
            )
            self._publish_path(amr, [])
            return
        amr.path = path
        amr.goal = goal
        amr.path_time = now
        self._publish_path(amr, path)
        self.get_logger().info(
            f"{amr.name}: planned {len(path)} waypoints to "
            f"({goal[0]:.2f}, {goal[1]:.2f})"
        )

    # ------------------------------------------------------------------
    # Managed warehouse flow
    # ------------------------------------------------------------------
    def _dispatch_tasks(self, now):
        if not self.managed:
            return
        live = {"PENDING", "ASSIGNED", "DELIVERING"}
        active = [t for t in self.tasks if t["status"] in live]
        if len(active) >= self.max_pending:
            return
        for template in self.task_templates:
            tid = str(template.get("id", ""))
            if any(t.get("template") == tid and t["status"] in live
                   for t in self.tasks):
                continue
            if not self._can_generate(template):
                continue
            self.task_counter += 1
            task = self._new_task(
                template,
                manual=False,
                task_id=f"{tid}-{self.task_counter:04d}",
            )
            self.tasks.append(task)
            self.publish_event(
                "work_order_created",
                {
                    "id": task["id"],
                    "template": tid,
                    "pickup": task["pickup"],
                    "dropoff": task["dropoff"],
                    "cargo": task["cargo"],
                    "description": task["description"],
                },
            )
            active.append(task)
            if len(active) >= self.max_pending:
                break

    def _can_generate(self, template):
        requires = template.get("requires", {}) or {}
        produces = template.get("produces", {}) or {}
        for key, amount in requires.items():
            if self.inventory.get(str(key), 0) < int(amount):
                return False
        for key, amount in produces.items():
            key = str(key)
            if (
                self.inventory.get(key, 0) + self.incoming.get(key, 0)
                + int(amount) > self.capacity.get(key, 99)
            ):
                return False
        return True

    def _try_reserve(self, task):
        if task.get("reserved"):
            return True
        if task.get("manual"):
            return True
        requires = task.get("requires", {}) or {}
        produces = task.get("produces", {}) or {}
        for key, amount in requires.items():
            if self.inventory.get(str(key), 0) < int(amount):
                return False
        for key, amount in produces.items():
            key = str(key)
            if (
                self.inventory.get(key, 0) + self.incoming.get(key, 0)
                + int(amount) > self.capacity.get(key, 99)
            ):
                return False
        for key, amount in requires.items():
            self.inventory[str(key)] = max(
                0, self.inventory.get(str(key), 0) - int(amount)
            )
        for key, amount in produces.items():
            key = str(key)
            self.incoming[key] = self.incoming.get(key, 0) + int(amount)
        task["reserved"] = True
        return True

    def _complete_task(self, task, amr, now):
        task["status"] = "DONE"
        task["completed"] = now
        task["assigned"] = amr.name
        self.completed += 1
        if task.get("reserved"):
            for key, amount in (task.get("produces", {}) or {}).items():
                key = str(key)
                self.inventory[key] = min(
                    self.capacity.get(key, 99),
                    self.inventory.get(key, 0) + int(amount),
                )
                self.incoming[key] = max(
                    0, self.incoming.get(key, 0) - int(amount)
                )
            task["reserved"] = False
        if self.managed:
            self.flow["completed"] = self.flow.get("completed", 0) + 1
        self.get_logger().info(
            f"COMPLETE {task['id']} by {amr.name} "
            f"({task['pickup']} -> {task['dropoff']})"
        )
        self.publish_event(
            "task_completed",
            {
                "id": task["id"],
                "robot": amr.name,
                "pickup": task["pickup"],
                "dropoff": task["dropoff"],
                "cargo": task.get("cargo", "tote"),
                "cycle": self.cycle,
                "inventory": dict(self.inventory),
            },
        )

    def _cancel_task(self, amr, reason):
        task = amr.task
        if task is None:
            return
        if task.get("reserved"):
            for key, amount in (task.get("requires", {}) or {}).items():
                key = str(key)
                self.inventory[key] = min(
                    self.capacity.get(key, 99),
                    self.inventory.get(key, 0) + int(amount),
                )
            for key, amount in (task.get("produces", {}) or {}).items():
                key = str(key)
                self.incoming[key] = max(
                    0, self.incoming.get(key, 0) - int(amount)
                )
            task["reserved"] = False
        for station in (task["pickup"], task["dropoff"]):
            if self.station_owner.get(station) == amr.name:
                self.station_owner.pop(station, None)
                self.station_release_pending.pop(station, None)
        task["status"] = "PENDING"
        task["assigned"] = None
        amr.task = None
        amr.path = []
        amr.goal = None
        amr.loaded = False
        amr.hold = False
        amr.waiting_for = None
        amr.wait_goal = None
        amr.wait_goal_station = None
        self._publish_path(amr, [])
        self._publish_load(amr)
        self.get_logger().warn(
            f"{amr.name}: task {task['id']} requeued ({reason})"
        )
        self.publish_event(
            "task_requeued",
            {"id": task["id"], "robot": amr.name, "reason": reason},
        )

    def _replenish(self, now):
        if not self.managed:
            return
        if now - self.last_supply >= self.supplier_period:
            self.last_supply = now
            if self.inventory.get("inbound", 0) < self.capacity.get("inbound", 99):
                self.inventory["inbound"] = self.inventory.get("inbound", 0) + 1
                self.flow["received"] = self.flow.get("received", 0) + 1
                self.publish_event(
                    "inbound_received",
                    {"inbound": self.inventory["inbound"]},
                )
        if now - self.last_ship >= self.shipping_period:
            self.last_ship = now
            if self.inventory.get("outbound", 0) > 0:
                self.inventory["outbound"] -= 1
                self.inventory["empty"] = min(
                    self.capacity.get("empty", 99),
                    self.inventory.get("empty", 0) + 1,
                )
                self.flow["shipped"] = self.flow.get("shipped", 0) + 1
                self.publish_event(
                    "outbound_shipped",
                    {
                        "shipped": self.flow["shipped"],
                        "empty": self.inventory["empty"],
                    },
                )

    # ------------------------------------------------------------------
    # Mode / teleop plumbing
    # ------------------------------------------------------------------
    def _desired_agent_mode(self, amr):
        if self.mode in ("PAUSED", "ESTOP"):
            return "STOPPED"
        if amr.name in self.teleop:
            return "TELEOP"
        return "AUTO"

    def _sync_agent_modes(self, force=False):
        for amr in self.amrs:
            desired = self._desired_agent_mode(amr)
            if force or desired != amr.agent_mode:
                amr.agent_mode = desired
                if amr.pub_mode is not None:
                    amr.pub_mode.publish(String(data=desired))
                if desired != "AUTO":
                    self._publish_path(amr, [])

    def _publish_mode(self):
        data = {
            "mode": self.mode,
            "teleop": sorted(self.teleop),
        }
        self.pub_mode.publish(String(data=json.dumps(data)))

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _publish_load(self, amr):
        msg = Twist()
        msg.linear.x = 1.0 if amr.loaded else 0.0
        amr.pub_load.publish(msg)
        amr.last_load_pub = amr.loaded

    def _publish_loads(self):
        for amr in self.amrs:
            if amr.last_load_pub != amr.loaded:
                self._publish_load(amr)

    def _publish_map(self):
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(self.grid.res)
        msg.info.width = int(self.grid.nx)
        msg.info.height = int(self.grid.ny)
        msg.info.origin.position.x = float(self.grid.x_min)
        msg.info.origin.position.y = float(self.grid.y_min)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        cells = np.where(self.grid.blocked, 100, 0).astype(np.int8)
        msg.data = cells.flatten(order="F").tolist()
        self.pub_map.publish(msg)

    def _publish_path(self, amr, points):
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()
        sampled = self.grid.resample(points, 0.12) if points else []
        for i, (x, y) in enumerate(sampled):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            if i + 1 < len(sampled):
                yaw = math.atan2(sampled[i + 1][1] - y, sampled[i + 1][0] - x)
            elif i > 0:
                yaw = math.atan2(y - sampled[i - 1][1], x - sampled[i - 1][0])
            else:
                yaw = amr.yaw
            ps.pose.orientation = yaw_quat(yaw)
            msg.poses.append(ps)
        amr.pub_path.publish(msg)

    def _publish_hold(self, amr, hold):
        if amr.last_hold_pub != hold:
            amr.pub_hold.publish(Bool(data=bool(hold)))
            amr.last_hold_pub = hold
        # Scale is 0 when held, otherwise 1.0.  Kept as a separate topic so a
        # future local planner can shape it without changing the hold logic.
        amr.pub_scale.publish(Float32(data=0.0 if hold else 1.0))

    def _publish_status(self, now):
        data = {
            "time": round(now, 3),
            "cycle": self.cycle,
            "completed": self.completed,
            "mode": self.mode,
            "teleop": sorted(self.teleop),
            "managed": self.managed,
            "inventory": dict(self.inventory),
            "incoming": dict(self.incoming),
            "capacity": dict(self.capacity),
            "flow": dict(self.flow),
            "map": {
                "frame_id": "map",
                "width": int(self.grid.nx),
                "height": int(self.grid.ny),
                "resolution": float(self.grid.res),
                "origin": [float(self.grid.x_min), float(self.grid.y_min)],
                "blocked_cells": int(self.grid.blocked.sum()),
            },
            "amrs": [
                {
                    "name": a.name,
                    "role": a.role,
                    "state": a.state,
                    "agent_mode": a.agent_mode,
                    "pose": None if a.pose is None else [
                        round(a.pose[0], 3), round(a.pose[1], 3)
                    ],
                    "yaw": round(a.yaw, 3),
                    "task": None if a.task is None else a.task["id"],
                    "cargo": None if a.task is None else a.task.get("cargo"),
                    "loaded": bool(a.loaded),
                    "hold": bool(a.hold),
                    "hold_reason": a.hold_reason,
                    "blocked": bool(a.blocked),
                    "remaining": round(a.remaining, 2),
                    "path_points": len(a.path),
                    "path": [
                        [round(p[0], 2), round(p[1], 2)]
                        for p in (a.path[::max(1, len(a.path) // 40)]
                                  if a.path else [])
                    ],
                    "preferred_templates": list(a.preferred),
                }
                for a in self.amrs
            ],
            "tasks": [
                {
                    "id": t["id"],
                    "template": t.get("template"),
                    "pickup": t["pickup"],
                    "dropoff": t["dropoff"],
                    "priority": int(t.get("priority", 50)),
                    "cargo": t.get("cargo", "tote"),
                    "description": t.get("description", ""),
                    "status": t["status"],
                    "assigned": t["assigned"],
                    "manual": bool(t.get("manual", False)),
                    "age": round(now - float(t.get("created", now)), 1),
                }
                for t in self.tasks
            ],
            "dynamic_obstacles": [
                {"name": name, "pose": [round(p[0], 3), round(p[1], 3)]}
                for name, (p, stamp) in self.dynamic_obstacles.items()
                if now - stamp < 2.0
            ],
        }
        self.pub_status.publish(String(data=json.dumps(data)))
        if int(now) % 5 == 0:
            states = " ".join(f"{a.name}:{a.state}" for a in self.amrs)
            inventory = " ".join(
                f"{k}={v}" for k, v in sorted(self.inventory.items())
            ) or "fixed-cycle"
            self.get_logger().info(
                f"mode {self.mode} completed {self.completed} | {states} | {inventory}"
            )

    def _publish_markers(self, now):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, amr in enumerate(self.amrs):
            if amr.pose is None:
                continue
            color = STATE_COLORS.get(amr.state, (1.0, 1.0, 1.0))
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "fleet"
            m.id = i + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(amr.pose[0])
            m.pose.position.y = float(amr.pose[1])
            m.pose.position.z = 1.15
            m.scale.x = m.scale.y = m.scale.z = 0.32
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.95
            arr.markers.append(m)

            t = Marker()
            t.header = m.header
            t.ns = "fleet_labels"
            t.id = 100 + i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(amr.pose[0])
            t.pose.position.y = float(amr.pose[1])
            t.pose.position.z = 1.65
            t.scale.z = 0.28
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 0.95
            task = amr.task["id"] if amr.task else "-"
            t.text = f"{amr.name}  {amr.state}  {task}"
            arr.markers.append(t)
        self.pub_markers.publish(arr)

    def publish_event(self, kind, payload):
        msg = {"event": kind, "time": round(time.time(), 3), **payload}
        self.pub_events.publish(String(data=json.dumps(msg)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _station_goal(self, amr, name):
        """Return the pad when it is ours/free, otherwise our wait point.

        A robot claims a free pad as soon as it is within
        ``station_claim_radius`` of it.  Robots that arrive while somebody
        else owns the pad stop ``station_wait_distance`` outside it and are
        released automatically when the owner moves on.
        """
        station = self._station(name)
        if amr.pose is None:
            return station

        owner = self.station_owner.get(name)
        pending = self.station_release_pending.get(name)
        if pending is not None:
            pending_amr = self.by_name.get(pending)
            if (pending_amr is None or pending_amr.pose is None
                    or self._dist(pending_amr.pose, station)
                    > self.station_wait_distance):
                self.station_release_pending.pop(name, None)
            else:
                # The previous user is still driving away.  Keep the pad
                # reserved so the next robot waits outside instead of
                # colliding with the departing robot.
                owner = pending
        if owner is not None and owner != amr.name:
            owner_amr = self.by_name.get(owner)
            if owner_amr is None:
                self.station_owner.pop(name, None)
                self.station_release_pending.pop(name, None)
                owner = None
            elif not self._uses_station(owner_amr, name):
                # A robot that has released the pad may still be inside the
                # departure clearance.  Keep the lock until it has actually
                # driven out of the whole departure lane.  (Clearing it as
                # soon as the task changed caused two AMRs to meet head-on
                # on the staging pad.)
                if self._dist(owner_amr.pose, station) > self.station_wait_distance:
                    self.station_owner.pop(name, None)
                    self.station_release_pending.pop(name, None)
                    owner = None

        if owner is None:
            if amr.waiting_for == name:
                self.get_logger().info(f"{amr.name}: {name} released - proceeding")
                amr.waiting_for = None
                amr.wait_goal = None
                amr.wait_goal_station = None
            if self._dist(amr.pose, station) <= self.station_claim_radius:
                self.station_owner[name] = amr.name
                amr.waiting_for = None
                amr.wait_goal = None
                amr.wait_goal_station = None
                self.get_logger().info(f"{amr.name}: claimed station {name}")
            return station

        if owner == amr.name:
            amr.waiting_for = None
            amr.wait_goal = None
            amr.wait_goal_station = None
            return station

        if amr.waiting_for != name:
            amr.waiting_for = name
            amr.wait_goal = None
            amr.wait_goal_station = None
            self.get_logger().info(
                f"{amr.name}: {name} occupied by {owner} - waiting outside"
            )
        return self._wait_goal(amr, name)

    def _uses_station(self, amr, name):
        if amr.task is None:
            return False
        if amr.state in ("TO_PICKUP", "PICKING"):
            return amr.task["pickup"] == name
        if amr.state in ("TO_DROPOFF", "DROPPING"):
            return amr.task["dropoff"] == name
        return False

    def _release_station(self, amr, name):
        if self.station_owner.get(name) == amr.name:
            # Keep the pad reserved while this robot drives out of the
            # departure lane; otherwise the next AMR claims it immediately
            # and the two meet head-on in the station approach.
            self.station_release_pending[name] = amr.name
            amr.waiting_for = None
            amr.wait_goal = None
            amr.wait_goal_station = None
            self.get_logger().info(f"{amr.name}: released station {name}")

    def _wait_goal(self, amr, name):
        if amr.wait_goal_station == name and amr.wait_goal is not None:
            return amr.wait_goal

        sx, sy = self._station(name)
        px, py = amr.pose
        base = math.atan2(py - sy, px - sx)
        occupied = [
            o.wait_goal for o in self.amrs
            if o is not amr and o.waiting_for == name and o.wait_goal
        ]

        goal = None
        # Prefer the innermost ring: waiting 1.9 m out keeps the aisle clear
        # and lets the robot claim the pad quickly once it is released.  Only
        # expand the radius when the closer ring is blocked.
        for scale in (1.0, 1.35, 1.8):
            radius = self.station_wait_distance * scale
            candidates = []
            for k in range(8):
                ang = base + k * (math.pi / 4.0)
                point = (sx + radius * math.cos(ang), sy + radius * math.sin(ang))
                if not self.grid.is_free_xy(*point):
                    continue
                if not self.grid.line_clear((sx, sy), point):
                    continue
                if any(self._dist(point, q) < 1.3 for q in occupied):
                    continue
                candidates.append((self._dist(amr.pose, point), point))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                goal = candidates[0][1]
                break

        if goal is None:
            # Last-resort point on the robot's side; A* will find the nearest
            # free cell if this lands inside the inflated margin.
            d = self.station_wait_distance
            goal = (sx + d * math.cos(base), sy + d * math.sin(base))

        amr.wait_goal = goal
        amr.wait_goal_station = name
        return goal

    def _station(self, name):
        s = self.stations[name]
        return (float(s["x"]), float(s["y"]))

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _dynamic_points(self, now):
        out = []
        for pose, stamp in self.dynamic_obstacles.values():
            if now - stamp < 2.0:
                out.append(pose)
        return out

    def _is_ahead(self, amr, other):
        """Is ``other`` inside the front half-plane of ``amr``?"""
        if amr.pose is None or other.pose is None:
            return True
        dx = other.pose[0] - amr.pose[0]
        dy = other.pose[1] - amr.pose[1]
        if math.hypot(dx, dy) < 0.9:
            return True
        return abs(wrap_angle(math.atan2(dy, dx) - amr.yaw)) < math.radians(100.0)

    @staticmethod
    def _dist_to_path(pose, path):
        return min((math.hypot(pose[0] - x, pose[1] - y) for x, y in path),
                   default=0.0)

    def _maybe_requeue(self, now):
        if self.managed:
            # Managed flow creates fresh work orders from live inventory.
            # Prune old completed history so a long-running demo stays small.
            done = [t for t in self.tasks if t["status"] == "DONE"
                    and now - float(t.get("completed") or now) > 180.0]
            if done:
                done_ids = {t["id"] for t in done}
                self.tasks = [t for t in self.tasks if t["id"] not in done_ids]
            return
        active = any(t["status"] != "DONE" for t in self.tasks)
        idle = all(a.task is None and a.state == "IDLE" for a in self.amrs)
        if active or not idle:
            self.all_done_since = None
            return
        if self.all_done_since is None:
            self.all_done_since = now
            return
        if now - self.all_done_since < 5.0:
            return
        if not self.loop_tasks:
            return
        for t in self.tasks:
            t["status"] = "PENDING"
            t["assigned"] = None
            t["completed"] = None
        self.cycle += 1
        self.all_done_since = None
        self.get_logger().info(
            f"All tasks complete - starting cycle {self.cycle}"
        )
        self.publish_event("cycle_started", {"cycle": self.cycle})


def main(args=None):
    rclpy.init(args=args)
    node = FleetManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # A SIGINT can land while rclpy is rebuilding its wait set; the
        # context is already gone in that case.  Re-raise genuine failures.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
