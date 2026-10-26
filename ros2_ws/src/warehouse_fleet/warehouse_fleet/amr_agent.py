#!/usr/bin/env python3
"""Per-AMR motion controller.

The fleet manager owns *what* each robot should do (task, path, right of way).
This node owns *how* the robot follows the path:

* pure-pursuit steering with a configurable look-ahead,
* speed shaping on heading error and distance-to-goal,
* a 2-D LiDAR safety envelope that stops the robot for obstacles that are not
  on the static map (other robots, the humanoid, misplaced crates),
* ``/<ns>/hold`` and ``/<ns>/speed_scale`` gating from the traffic manager.

Topics (namespace = robot name, e.g. ``/amr1``)::

    in   <ns>/path         nav_msgs/Path        from fleet_manager
    in   <ns>/hold         std_msgs/Bool        from fleet_manager
    in   <ns>/speed_scale  std_msgs/Float32     from fleet_manager
    in   <ns>/odom         nav_msgs/Odometry    from Isaac Sim
    in   <ns>/scan         sensor_msgs/LaserScan from Isaac Sim
    out  <ns>/cmd_vel      geometry_msgs/Twist   to Isaac Sim
    out  <ns>/nav_status   std_msgs/String       JSON progress report
"""

from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String

from .config_utils import load_config
from .grid_map import wrap_angle, yaw_from_quaternion


class AmrAgent(Node):
    def __init__(self):
        super().__init__("amr_agent")
        self.declare_parameter("robot", "amr1")
        self.declare_parameter("config", "")
        self.declare_parameter("max_speed", 0.0)
        self.declare_parameter("control_rate", 20.0)

        self.name = str(self.get_parameter("robot").value).strip("/")
        cfg = load_config(self.get_parameter("config").value)
        self.cfg = cfg
        fleet = cfg["fleet"]
        spec = next(
            (a for a in cfg["amrs"] if a["name"] == self.name),
            {"name": self.name, "max_speed": 0.65},
        )

        configured = float(self.get_parameter("max_speed").value)
        self.max_speed = configured if configured > 0.0 else float(
            spec.get("max_speed", 0.65)
        )
        self.lookahead = float(fleet.get("path_lookahead", 0.55))
        self.safety_stop = float(fleet.get("scan_safety_stop", 0.55))
        self.safety_slow = float(fleet.get("scan_safety_slow", 1.30))
        self.safety_side = float(fleet.get("scan_safety_side", 0.52))
        self.arrival_tol = float(fleet.get("arrival_tolerance", 0.35))

        self.pose = None          # (x, y, yaw)
        self.path = []
        self.path_stamp = 0.0
        self.hold = False
        self.scale = 1.0
        self.scan = None
        self.scan_time = 0.0
        self.blocked_since = None
        self.min_range = float("inf")
        self.min_angle = 0.0
        self.last_block_log = 0.0
        self.last_index = 0
        self.last_control = 0.0
        self.last_turn = 0.0

        # Obstacles the static map already knows about.  The LiDAR safety
        # envelope must ignore these: the planner is responsible for keeping
        # the path clear of racks and walls, and stopping for them here only
        # lets the robot get pinned against a shelf during a recovery.
        self.static_rects = []
        for key in ("racks", "static_obstacles", "pallets"):
            for item in cfg["scene"].get(key, []):
                self.static_rects.append(
                    (
                        float(item["x"]),
                        float(item["y"]),
                        float(item.get("yaw", 0.0)),
                        float(item["size"][0]) / 2.0,
                        float(item["size"][1]) / 2.0,
                    )
                )
        self.bounds = cfg["world"]["bounds"]

        # Recovery state (back away when a dynamic obstacle or a wedge keeps
        # the robot from making progress).
        self.recover_until = 0.0
        self.recover_speed = 0.0
        self.recover_turn = 0.0
        self.last_recover = 0.0
        self.last_progress_pose = None
        self.last_progress_time = 0.0

        self.pub_cmd = self.create_publisher(Twist, f"{self.name}/cmd_vel", 10)
        self.pub_status = self.create_publisher(
            String, f"{self.name}/nav_status", 10
        )
        self.create_subscription(Path, f"{self.name}/path", self.on_path, 10)
        self.create_subscription(Bool, f"{self.name}/hold", self.on_hold, 10)
        self.create_subscription(
            Float32, f"{self.name}/speed_scale", self.on_scale, 10
        )
        self.create_subscription(Odometry, f"{self.name}/odom", self.on_odom, 10)
        self.create_subscription(LaserScan, f"{self.name}/scan", self.on_scan, 10)

        rate = max(5.0, float(self.get_parameter("control_rate").value))
        self.create_timer(1.0 / rate, self.control)
        self.get_logger().info(
            f"{self.name} agent ready (speed {self.max_speed:.2f} m/s, "
            f"lookahead {self.lookahead:.2f} m)"
        )

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def on_path(self, msg: Path):
        self.path = [(float(p.pose.position.x), float(p.pose.position.y))
                     for p in msg.poses]
        self.path_stamp = time.time()
        self.last_index = 0

    def on_hold(self, msg: Bool):
        self.hold = bool(msg.data)

    def on_scale(self, msg: Float32):
        self.scale = max(0.0, min(1.0, float(msg.data)))

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.pose = (
            float(p.x),
            float(p.y),
            yaw_from_quaternion(msg.pose.pose.orientation),
        )

    def on_scan(self, msg: LaserScan):
        self.scan = msg
        self.scan_time = time.time()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def control(self):
        now = time.time()
        v, w = 0.0, 0.0
        remaining = 0.0
        safety = 1.0
        status = "idle"

        if self.pose is not None and self.path:
            remaining = math.hypot(
                self.path[-1][0] - self.pose[0], self.path[-1][1] - self.pose[1]
            )

        if self.pose is None:
            status = "no_odom"
        elif not self.path:
            status = "no_path"
        elif self.hold or self.scale <= 0.0:
            status = "hold"
        elif remaining <= max(0.18, self.arrival_tol * 0.6):
            status = "arrived"
        elif now < self.recover_until:
            v, w = self.recover_speed, self.recover_turn
            status = "reversing"
        else:
            v, w, safety, status = self._follow(remaining)

        if status == "blocked":
            v = 0.0
            w = 0.0
            if self.blocked_since is not None and now - self.blocked_since > 1.0:
                # Safety stop is not a dead end: rotate away from the nearest
                # obstacle so the forward cone clears, then resume pursuit.
                if abs(self.min_angle) > 0.10:
                    w = -0.60 if self.min_angle > 0.0 else 0.60
                else:
                    # Dead ahead: turn towards the path if possible.
                    alpha = self._path_heading_error()
                    w = 0.60 if alpha >= 0.0 else -0.60
                status = "avoiding"
                # Rotating in place does not help when the robot is wedged
                # (for example between two robots).  After a couple of
                # seconds, reverse briefly to open the gap.
                if (
                    now - self.blocked_since > 2.4
                    and now - self.last_recover > 2.0
                ):
                    self._begin_recovery(now, "lidar block")
                    v, w = self.recover_speed, self.recover_turn
                    status = "reversing"
                elif now - self.last_block_log > 2.0:
                    self.last_block_log = now
                    self.get_logger().warn(
                        f"{self.name}: obstacle at {self.min_range:.2f} m, "
                        f"{math.degrees(self.min_angle):+.0f} deg - turning away"
                    )
        elif safety < 1.0:
            v *= safety

        # Progress watchdog: if we keep commanding forward motion but the
        # base does not move, back away.  This catches physical wedges that
        # the forward LiDAR cone cannot see.
        if status in ("following", "turning") and self.pose is not None and v > 0.08:
            if self.last_progress_pose is None:
                self.last_progress_pose = self.pose
                self.last_progress_time = now
            elif (
                math.hypot(
                    self.pose[0] - self.last_progress_pose[0],
                    self.pose[1] - self.last_progress_pose[1],
                )
                > 0.06
            ):
                self.last_progress_pose = self.pose
                self.last_progress_time = now
            elif now - self.last_progress_time > 2.5 and now - self.last_recover > 3.0:
                self._begin_recovery(now, "no progress")
                v, w = self.recover_speed, self.recover_turn
                status = "reversing"
        else:
            self.last_progress_pose = self.pose
            self.last_progress_time = now

        # Do not publish faster than the controller really is.
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.pub_cmd.publish(twist)

        if now - self.last_control >= 0.5:
            self.last_control = now
            self._publish_status(status, remaining, v, w, safety)

    def _follow(self, remaining):
        x, y, yaw = self.pose
        path = self.path

        # Closest path point, searched around the previous index first so a
        # path that crosses itself cannot make the robot jump backwards.
        window = 40
        lo = max(0, self.last_index - 5)
        hi = min(len(path), self.last_index + window)
        best_i = min(
            range(lo, hi),
            key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2,
        )
        # If we are nowhere near the local window (teleport / new path), do a
        # full search once.
        if math.hypot(path[best_i][0] - x, path[best_i][1] - y) > 1.5:
            best_i = min(
                range(len(path)),
                key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2,
            )
        self.last_index = best_i

        # Pure-pursuit look-ahead point.
        target = path[-1]
        travelled = 0.0
        for i in range(best_i, len(path) - 1):
            travelled += math.hypot(
                path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
            )
            if travelled >= self.lookahead:
                target = path[i + 1]
                break

        dx = target[0] - x
        dy = target[1] - y
        dist = max(1e-6, math.hypot(dx, dy))
        alpha = wrap_angle(math.atan2(dy, dx) - yaw)

        speed_scale, blocked = self._scan_clearance()
        if blocked:
            return 0.0, 0.0, 0.0, "blocked"

        # Rotate in place when the heading error is large, otherwise chase the
        # look-ahead point with a curvature command.
        if abs(alpha) > 1.15:
            if abs(alpha) > 2.85 and abs(self.last_turn) > 0.1:
                # At (almost) 180 degrees both directions are valid; keep the
                # previous one so sensor noise cannot make the robot dither.
                w = 0.80 * math.copysign(1.0, self.last_turn)
            else:
                w = math.copysign(0.85, alpha)
                w *= max(0.35, 1.0 - abs(alpha) / math.pi)
            self.last_turn = w
            return 0.0, float(w), speed_scale, "turning"

        v = self.max_speed * self.scale
        v *= max(0.20, math.cos(alpha))
        # Ease into the final approach; the fleet manager switches state at
        # arrival_tolerance, this keeps the base from overshooting the pad.
        v *= min(1.0, max(0.10, remaining / 0.70))
        w = 2.0 * v * math.sin(alpha) / max(0.20, dist)
        w = max(-1.1, min(1.1, w))
        self.last_turn = w
        return float(v), float(w), speed_scale, "following"

    def _scan_clearance(self):
        """Return (speed_scale, blocked) for *dynamic* obstacles ahead.

        Points that fall inside a rack, pallet, table or the perimeter are
        ignored: those are already represented in the planner's occupancy
        grid, and reacting to them here made robots freeze whenever they
        brushed a shelf.  Only things that are not on the static map (other
        robots, the humanoid, a stray crate) trigger the safety envelope.
        """
        if self.scan is None or time.time() - self.scan_time > 1.5:
            self.blocked_since = None
            return 1.0, False
        scan = self.scan
        ranges = scan.ranges
        if not ranges:
            return 1.0, False
        half_cone = math.radians(42.0)
        closest = float("inf")
        closest_angle = 0.0
        closest_any = float("inf")
        for i, value in enumerate(ranges):
            if not math.isfinite(value) or value <= 0.06:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if self.pose is not None:
                wx = self.pose[0] + value * math.cos(self.pose[2] + angle)
                wy = self.pose[1] + value * math.sin(self.pose[2] + angle)
                if self._static_point(wx, wy):
                    continue
            if value < closest_any:
                closest_any = value
            if abs(wrap_angle(angle)) > half_cone:
                continue
            if value < closest:
                closest = value
                closest_angle = angle
        self.min_range = closest
        self.min_angle = wrap_angle(closest_angle)

        # Optional 360-degree close-range guard.  It is disabled by default:
        # the LiDAR can see its own chassis at short range, and the forklift
        # aisle is already reserved in the planner.  Set scan_safety_side > 0
        # to enable it on a platform whose self-hits are filtered.
        if (self.safety_side > 0.0 and 0.20 < closest_any < self.safety_side):
            if self.blocked_since is None:
                self.blocked_since = time.time()
            return 0.0, True

        if closest < self.safety_stop:
            if self.blocked_since is None:
                self.blocked_since = time.time()
            return 0.0, True
        self.blocked_since = None
        if closest < self.safety_slow:
            span = max(1e-3, self.safety_slow - self.safety_stop)
            return max(0.0, (closest - self.safety_stop) / span), False
        return 1.0, False

    def _static_point(self, x, y):
        """True if a world point belongs to a known static obstacle."""
        for cx, cy, yaw, hx, hy in self.static_rects:
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = x - cx, y - cy
            lx = dx * c + dy * s
            ly = -dx * s + dy * c
            if abs(lx) <= hx + 0.10 and abs(ly) <= hy + 0.10:
                return True
        b = self.bounds
        return (
            x < b["x_min"] + 0.20
            or x > b["x_max"] - 0.20
            or y < b["y_min"] + 0.20
            or y > b["y_max"] - 0.20
        )

    def _rear_clearance(self):
        """Closest range behind the robot (for safe reverse recovery)."""
        if self.scan is None:
            return float("inf")
        closest = float("inf")
        half_cone = math.radians(55.0)
        for i, value in enumerate(self.scan.ranges):
            if not math.isfinite(value) or value <= 0.06:
                continue
            angle = self.scan.angle_min + i * self.scan.angle_increment
            if abs(wrap_angle(angle - math.pi)) > half_cone:
                continue
            closest = min(closest, value)
        return closest

    def _begin_recovery(self, now, reason):
        """Back out of a wedge, or spin hard if there is no room behind."""
        self.last_recover = now
        self.blocked_since = None
        rear = self._rear_clearance()
        if rear < 0.45:
            self.recover_speed = 0.0
            self.recover_turn = 0.9 if self.min_angle <= 0.0 else -0.9
            self.recover_until = now + 0.7
        else:
            self.recover_speed = -0.22
            self.recover_turn = 0.35 if self.min_angle > 0.0 else -0.35
            self.recover_until = now + 0.9
        self.last_progress_pose = self.pose
        self.last_progress_time = now
        self.get_logger().warn(
            f"{self.name}: {reason} - recovery "
            f"({'backing up' if self.recover_speed < 0 else 'spinning'})"
        )

    def _path_heading_error(self):
        if not self.path or self.pose is None:
            return 0.0
        x, y, yaw = self.pose
        target = self.path[min(len(self.path) - 1, self.last_index + 4)]
        return wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw)

    def _publish_status(self, status, remaining, v, w, safety):
        blocked = self.blocked_since is not None
        data = {
            "robot": self.name,
            "state": status,
            "pose": None if self.pose is None else [
                round(self.pose[0], 3), round(self.pose[1], 3), round(self.pose[2], 3)
            ],
            "remaining": round(remaining, 3),
            "blocked": blocked,
            "hold": bool(self.hold),
            "speed": round(v, 3),
            "turn": round(w, 3),
            "safety_scale": round(safety, 3),
            "min_range": None if not math.isfinite(self.min_range) else round(self.min_range, 3),
            "min_angle": round(self.min_angle, 3),
            "path_points": len(self.path),
            "path_age": round(time.time() - self.path_stamp, 2) if self.path else 0.0,
        }
        self.pub_status.publish(String(data=json.dumps(data)))


def main(args=None):
    rclpy.init(args=args)
    node = AmrAgent()
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
