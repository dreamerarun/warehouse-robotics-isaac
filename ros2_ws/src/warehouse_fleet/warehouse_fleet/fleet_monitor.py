#!/usr/bin/env python3
"""Console dashboard for the warehouse fleet.

Subscribes to ``/fleet/status`` and ``/fleet/task_events`` and prints a compact
table.  It is deliberately read-only: stopping it never affects the robots.
"""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class FleetMonitor(Node):
    def __init__(self):
        super().__init__("fleet_monitor")
        self.declare_parameter("period", 5.0)
        self.declare_parameter("events", True)
        self.period = max(1.0, float(self.get_parameter("period").value))
        self.show_events = bool(self.get_parameter("events").value)

        self.status = None
        self.status_time = 0.0
        self.humanoid = None
        self.humanoid_time = 0.0
        self.last_print = 0.0
        self.events = 0

        self.create_subscription(String, "/fleet/status", self.on_status, 10)
        self.create_subscription(
            String, "/humanoid/status", self.on_humanoid_status, 10
        )
        if self.show_events:
            self.create_subscription(
                String, "/fleet/task_events", self.on_event, 50
            )
        self.create_timer(1.0, self.tick)
        self.get_logger().info("Fleet monitor ready (Ctrl+C to stop)")

    # ------------------------------------------------------------------
    def on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
            self.status_time = time.time()
        except Exception as exc:
            self.get_logger().warn(f"bad fleet status: {exc}")

    def on_humanoid_status(self, msg):
        try:
            self.humanoid = json.loads(msg.data)
            self.humanoid_time = time.time()
        except Exception:
            pass

    def on_event(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        kind = data.get("event", "?")
        self.events += 1
        if kind == "task_assigned":
            line = (
                f"ASSIGN   {data.get('id')} -> {data.get('robot')}  "
                f"{data.get('pickup')} -> {data.get('dropoff')}"
            )
        elif kind == "task_completed":
            line = f"DONE     {data.get('id')} by {data.get('robot')}"
        elif kind == "cycle_started":
            line = f"CYCLE    {data.get('cycle')} started"
        elif kind == "traffic_swap":
            line = f"TRAFFIC  {data.get('a')} / {data.get('b')} swapped"
        elif kind == "task_submitted":
            line = (
                f"SUBMIT   {data.get('id')}  "
                f"{data.get('pickup')} -> {data.get('dropoff')}"
            )
        else:
            line = kind.upper()
        print(f"  [{time.strftime('%H:%M:%S')}] {line}", flush=True)

    # ------------------------------------------------------------------
    def tick(self):
        now = time.time()
        if self.status is None:
            if now - self.last_print > 10.0:
                self.last_print = now
                print("  waiting for /fleet/status ...", flush=True)
            return
        if now - self.last_print < self.period:
            return
        self.last_print = now
        self._print_table()

    def _print_table(self):
        data = self.status
        age = time.time() - self.status_time
        tasks = data.get("tasks", [])
        counts = {}
        for task in tasks:
            counts[task.get("status", "?")] = counts.get(task.get("status", "?"), 0) + 1
        done = sum(1 for t in tasks if t.get("status") == "DONE")

        print()
        print(
            f"=== WAREHOUSE FLEET  sim t={data.get('time', 0):8.1f}s  "
            f"cycle {data.get('cycle', 0)}  tasks {done}/{len(tasks)} complete "
            f"=== status age {age:.1f}s"
        )
        print(
            f"{'ROBOT':<6} {'STATE':<11} {'TASK':<5} {'POSE':<17} "
            f"{'REM':>5}  NOTE"
        )
        for amr in data.get("amrs", []):
            pose = amr.get("pose")
            pose_txt = "-" if not pose else f"({pose[0]:+6.2f},{pose[1]:+6.2f})"
            task = amr.get("task") or "-"
            note = ""
            if amr.get("hold_reason"):
                note = amr["hold_reason"]
            elif amr.get("blocked"):
                note = "LIDAR BLOCKED"
            print(
                f"{amr.get('name', '?'):<6} {amr.get('state', '?'):<11} "
                f"{task:<5} {pose_txt:<17} "
                f"{amr.get('remaining', 0.0):5.2f}  {note}"
            )

        if self.humanoid and time.time() - self.humanoid_time < 3.0:
            h = self.humanoid
            pose = h.get("pose", [0.0, 0.0, 0.0])
            carry = "CARRYING" if h.get("carrying") else "empty"
            print(
                f"{'H1':<6} {h.get('state', '?'):<11} {'tote':<5} "
                f"({pose[0]:+6.2f},{pose[1]:+6.2f}) "
                f"{h.get('distance', 0.0):5.2f}  {carry}  loads={h.get('loads', 0)}"
            )
        obstacles = data.get("dynamic_obstacles", [])
        if obstacles:
            text = "  ".join(
                f"{o['name']}=({o['pose'][0]:+.1f},{o['pose'][1]:+.1f})"
                for o in obstacles
            )
            print(f"dynamic obstacles: {text}")

        order = ["PENDING", "ASSIGNED", "DELIVERING", "DONE"]
        summary = "  ".join(f"{k.lower()}={counts.get(k, 0)}" for k in order)
        print(f"tasks: {summary}   (events seen: {self.events})")
        print(flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = FleetMonitor()
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
