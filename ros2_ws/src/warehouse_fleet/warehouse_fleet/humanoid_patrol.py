#!/usr/bin/env python3
"""Unitree H1 material-handling controller.

The humanoid is driven by Isaac Sim's flat-terrain walking policy, which takes
a velocity command ``[vx, vy, wz]``.  This node runs a real task loop:

    pickup conveyor -> carry tote -> packing conveyor -> return -> repeat

The simulator shows the tote in the H1's arms and puts a delivered tote on the
packing conveyor.  The node closes the loop on ``/humanoid/odom`` and reports
task state on ``/humanoid/status``.  If no ``humanoid.task`` block is present
in the config, the original patrol route is used instead.

Topics::

    in   /humanoid/odom      nav_msgs/Odometry    from Isaac Sim
    out  /humanoid/cmd_vel   geometry_msgs/Twist  to Isaac Sim
    out  /humanoid/status    std_msgs/String      JSON status
"""

from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .config_utils import load_config
from .grid_map import wrap_angle, yaw_from_quaternion


class HumanoidPatrol(Node):
    def __init__(self):
        super().__init__("humanoid_patrol")
        self.declare_parameter("config", "")
        cfg = load_config(self.get_parameter("config").value)
        spec = cfg["humanoid"]
        self.name = str(spec.get("name", "humanoid"))
        self.speed = float(spec.get("speed", 0.55))
        self.turn_gain = float(spec.get("turn_gain", 1.4))
        self.tolerance = float(spec.get("waypoint_tolerance", 0.45))
        self.task_cfg = spec.get("task")
        if self.task_cfg:
            self.pickup = self._point(self.task_cfg["pickup"])
            self.dropoff = self._point(self.task_cfg["dropoff"])
            self.home = self._point(self.task_cfg.get("home", self.task_cfg["pickup"]))
            self.load_time = float(self.task_cfg.get("load_time", 3.0))
            self.unload_time = float(self.task_cfg.get("unload_time", 3.0))
        else:
            self.pickup = self.dropoff = self.home = None
            self.load_time = self.unload_time = 0.0
            self.waypoints = [
                (float(w["x"]), float(w["y"])) for w in spec["patrol"]
            ]
            self.index = 0

        self.pose = None
        self.odom_time = 0.0
        self.last_pose = None
        self.last_pose_time = 0.0
        self.cmd = (0.0, 0.0)
        self.last_status_pub = 0.0

        # Task state
        self.state = "TO_PICKUP" if self.task_cfg else "PATROL"
        self.state_since = time.time()
        self.carrying = False
        self.loads = 0
        self.best_distance = float("inf")
        self.best_time = time.time()
        self.settle_until = time.time() + 2.5
        self.target = self.pickup
        self.face_yaw = self.pickup[2] if self.pickup else 0.0

        self.pub_cmd = self.create_publisher(Twist, f"/{self.name}/cmd_vel", 10)
        self.pub_status = self.create_publisher(
            String, f"/{self.name}/status", 10
        )
        self.create_subscription(
            Odometry, f"/{self.name}/odom", self.on_odom, 10
        )
        self.create_timer(1.0 / 20.0, self.tick)
        if self.task_cfg:
            self.get_logger().info(
                "H1 material handler ready: "
                f"pickup=({self.pickup[0]:.1f},{self.pickup[1]:.1f}) "
                f"dropoff=({self.dropoff[0]:.1f},{self.dropoff[1]:.1f})"
            )
        else:
            self.get_logger().info(
                f"H1 patrol ready: {len(self.waypoints)} waypoints at "
                f"{self.speed:.2f} m/s"
            )

    @staticmethod
    def _point(item):
        return (
            float(item["x"]),
            float(item["y"]),
            float(item.get("yaw", 0.0)),
        )

    # ------------------------------------------------------------------
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        now = time.time()
        pose = (float(p.x), float(p.y), yaw_from_quaternion(msg.pose.pose.orientation))

        # The simulator teleports the H1 back to its spawn if the policy falls
        # over.  Treat a large jump as a reset instead of walking back across
        # the warehouse.
        if self.last_pose is not None and now - self.last_pose_time < 0.75:
            jump = math.hypot(pose[0] - self.last_pose[0], pose[1] - self.last_pose[1])
            if jump > 1.5 and now >= self.settle_until:
                self.get_logger().warn("H1 position jump detected - restarting task")
                self.state = "TO_PICKUP" if self.task_cfg else "PATROL"
                self.state_since = now
                self.carrying = False
                self.index = 0
                self.best_distance = float("inf")
                self.best_time = now
                self.settle_until = now + 1.5
        self.pose = pose
        self.odom_time = now
        self.last_pose = pose
        self.last_pose_time = now

    # ------------------------------------------------------------------
    def tick(self):
        now = time.time()
        if self.pose is None or now - self.odom_time > 1.0:
            self._publish(0.0, 0.0)
            return
        if self.task_cfg:
            self._tick_task(now)
        else:
            self._tick_patrol(now)
        self._publish_status(now)

    def _tick_task(self, now):
        x, y, yaw = self.pose
        if self.state in ("LOADING", "UNLOADING"):
            # Hold position and face the conveyor while the transfer happens.
            error = wrap_angle(self.face_yaw - yaw)
            wz = max(-0.8, min(0.8, self.turn_gain * error))
            if abs(error) < 0.08:
                wz = 0.0
            self._publish(0.0, wz)
            self.cmd = (0.0, wz)
            dwell = self.load_time if self.state == "LOADING" else self.unload_time
            if now - self.state_since >= dwell:
                if self.state == "LOADING":
                    self.carrying = True
                    self.loads += 1
                    self.state = "TO_DROPOFF"
                    self.target = self.dropoff
                    self.face_yaw = self.dropoff[2]
                    self.get_logger().info(
                        f"tote loaded - carrying to dropoff ({self.loads} total)"
                    )
                else:
                    self.carrying = False
                    self.state = "TO_PICKUP"
                    self.target = self.pickup
                    self.face_yaw = self.pickup[2]
                    self.get_logger().info("tote delivered - returning to pickup")
                self.state_since = now
                self.best_distance = float("inf")
                self.best_time = now
            return

        if self.target is None:
            self.target = self.pickup
        tx, ty, tyaw = self.target
        dist = self._drive_to(tx, ty, now)
        self.face_yaw = tyaw
        if dist <= self.tolerance:
            self._publish(0.0, 0.0)
            self.cmd = (0.0, 0.0)
            self.state_since = now
            self.best_distance = float("inf")
            self.best_time = now
            if self.state == "TO_PICKUP":
                self.state = "LOADING"
                self.get_logger().info("arrived at pickup conveyor - loading")
            elif self.state == "TO_DROPOFF":
                self.state = "UNLOADING"
                self.get_logger().info("arrived at packing conveyor - unloading")

    def _drive_to(self, tx, ty, now):
        x, y, yaw = self.pose
        dx, dy = tx - x, ty - y
        distance = math.hypot(dx, dy)
        error = wrap_angle(math.atan2(dy, dx) - yaw)
        wz = max(-0.8, min(0.8, self.turn_gain * error))
        alignment = math.cos(error)
        if alignment > 0.45:
            vx = self.speed * alignment
        else:
            vx = 0.0
        vx *= min(1.0, max(0.35, distance / 1.0))

        if distance < self.best_distance - 0.10:
            self.best_distance = distance
            self.best_time = now
        elif now - self.best_time > 15.0:
            self.get_logger().warn(
                f"no progress towards ({tx:.1f}, {ty:.1f}); retrying"
            )
            self.best_distance = float("inf")
            self.best_time = now
            vx = 0.0
            wz = 0.6

        self._publish(vx, wz)
        self.cmd = (vx, wz)
        return distance

    def _tick_patrol(self, now):
        x, y, yaw = self.pose
        tx, ty = self.waypoints[self.index]
        dx, dy = tx - x, ty - y
        distance = math.hypot(dx, dy)

        if distance <= self.tolerance:
            self.index = (self.index + 1) % len(self.waypoints)
            self.best_distance = float("inf")
            self.best_time = now
            tx, ty = self.waypoints[self.index]
            dx, dy = tx - x, ty - y
            distance = math.hypot(dx, dy)

        if distance < self.best_distance - 0.10:
            self.best_distance = distance
            self.best_time = now
        elif now - self.best_time > 12.0:
            self.index = (self.index + 1) % len(self.waypoints)
            self.best_distance = float("inf")
            self.best_time = now
            tx, ty = self.waypoints[self.index]
            dx, dy = tx - x, ty - y
            distance = math.hypot(dx, dy)

        error = wrap_angle(math.atan2(dy, dx) - yaw)
        wz = max(-0.8, min(0.8, self.turn_gain * error))
        alignment = math.cos(error)
        vx = self.speed * alignment if alignment > 0.45 else 0.0
        vx *= min(1.0, max(0.35, distance / 1.0))
        self._publish(vx, wz)
        self.cmd = (vx, wz)

    # ------------------------------------------------------------------
    def _publish_status(self, now):
        if now - self.last_status_pub < 0.5:
            return
        self.last_status_pub = now
        x, y, yaw = self.pose
        target = self.target or self.pickup or (x, y, 0.0)
        data = {
            "state": self.state,
            "task": "tote_transfer" if self.task_cfg else "patrol",
            "carrying": self.carrying,
            "loads": self.loads,
            "target": [round(target[0], 3), round(target[1], 3)],
            "distance": round(
                math.hypot(target[0] - x, target[1] - y), 3
            ),
            "pose": [round(x, 3), round(y, 3), round(yaw, 3)],
            "cmd": [round(self.cmd[0], 3), round(self.cmd[1], 3)],
        }
        self.pub_status.publish(String(data=json.dumps(data)))

    def _publish(self, vx, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanoidPatrol()
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
        try:
            node.pub_cmd.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
