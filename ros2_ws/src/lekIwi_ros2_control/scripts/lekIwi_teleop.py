#!/usr/bin/env python3
# ============================================================================
# lekIwi_teleop.py -- keyboard teleoperation for the LeKiwi (base + 6DOF arm)
# ----------------------------------------------------------------------------
# This node is a PURE ROS 2 node.  It never touches Isaac Sim directly:
#
#   base : publishes geometry_msgs/Twist on /cmd_vel
#            -> lekIwi_base_controller (IK + safety)
#            -> ros2_control base_controller
#            -> Isaac Sim
#   arm  : publishes trajectory_msgs/JointTrajectory on
#          /arm_controller/joint_trajectory
#            -> ros2_control joint_trajectory_controller
#            -> Isaac Sim
#
# CONTROLS
#   BASE                        ARM
#     W  forward (linear.x +)     1 / 2   J1  - / +
#     S  backward (linear.x -)    3 / 4   J2  - / +
#     A  rotate left (wz +)       5 / 6   J3  - / +
#     D  rotate right (wz -)      7 / 8   J4  - / +
#     Q  strafe left  (y +)       9 / 0   J5  - / +
#     E  strafe right (y -)       - / =   J6  - / +
#     SPACE stop (zero twist)     R       reset arm to home pose
#                                 ESC     exit (zero base, stop publishing)
#
# SMOOTHNESS: this node publishes the *requested* twist at a fixed rate.
# Acceleration limiting (slew-rate ramp) is done once, centrally, in
# `lekIwi_base_controller`, so every /cmd_vel source is smoothed and clamped
# identically.
#
# KEY TIMEOUT: terminals only report key *presses* (with OS auto-repeat), never
# releases.  If no motion key arrives within `key_timeout` seconds the node
# releases the target to zero.  Raise it if the base stutters when you first
# press a key (the Linux auto-repeat delay is ~660 ms by default); you can make
# the keyboard snappier with:  xset r rate 300 30
# ============================================================================

import math
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINT_NAMES = [
    "STS3215_03a_v1_Revolute_45",
    "STS3215_03a_v1_1_Revolute_49",
    "STS3215_03a_v1_2_Revolute_51",
    "STS3215_03a_v1_3_Revolute_53",
    "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "STS3215_03a_v1_4_Revolute_57",
]

HELP = """
+------------------------------------------------------------------+
|  LeKiwi teleoperation  (ROS 2 Humble -> Isaac Sim 5.1)            |
+------------------------------------------------------------------+
|  BASE                                                          |
|    W / S   forward / backward                                  |
|    A / D   rotate left / right                                 |
|    Q / E   strafe left / right                                 |
|    SPACE   stop                                                |
|  ARM                                                             |
|    1 / 2   J1  - / +                                           |
|    3 / 4   J2  - / +                                           |
|    5 / 6   J3  - / +                                           |
|    7 / 8   J4  - / +                                           |
|    9 / 0   J5  - / +                                           |
|    - / =   J6  - / +                                           |
|    R       reset arm to home pose                              |
|    ESC     exit                                                |
+------------------------------------------------------------------+
"""


class LeKiwiTeleop(Node):
    def __init__(self):
        super().__init__("lekIwi_teleop")

        # ---------------- parameters ------------------------------------
        self.declare_parameter("linear_speed", 0.30)
        self.declare_parameter("strafe_speed", 0.30)
        self.declare_parameter("angular_speed", 0.80)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("key_timeout", 1.0)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("arm_command_topic", "/arm_controller/joint_trajectory")
        self.declare_parameter("arm_step", 0.05)
        self.declare_parameter("arm_goal_time", 0.20)
        self.declare_parameter("arm_joint_names", ARM_JOINT_NAMES)
        self.declare_parameter("arm_position_min",
                               [-3.14159265, -3.14159265, -3.14159265,
                                -3.14159265, -3.14159265, -1.57079633])
        self.declare_parameter("arm_position_max",
                               [3.14159265, 3.14159265, 3.14159265,
                                3.14159265, 3.14159265, 1.57079633])
        self.declare_parameter("arm_home_positions",
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        gp = lambda n: self.get_parameter(n).value
        self.lin_speed = float(gp("linear_speed"))
        self.str_speed = float(gp("strafe_speed"))
        self.ang_speed = float(gp("angular_speed"))
        self.key_timeout = float(gp("key_timeout"))
        self.arm_step = float(gp("arm_step"))
        self.arm_goal_time = float(gp("arm_goal_time"))
        self.arm_names = list(gp("arm_joint_names"))
        self.arm_min = np.array(gp("arm_position_min"), dtype=float)
        self.arm_max = np.array(gp("arm_position_max"), dtype=float)
        self.arm_home = np.array(gp("arm_home_positions"), dtype=float)

        if len(self.arm_names) != 6:
            raise ValueError("arm_joint_names must have exactly 6 entries")

        # ---------------- state ------------------------------------------
        self._twist_target = np.zeros(3)     # [vx, vy, wz]
        self._last_motion_key = 0.0
        self._arm_targets = np.array(self.arm_home, dtype=float)
        self._arm_dirty = False
        self._running = True
        self._estop = False

        # ---------------- I/O --------------------------------------------
        self._cmd_pub = self.create_publisher(
            Twist, gp("cmd_vel_topic"), 10
        )
        self._arm_pub = self.create_publisher(
            JointTrajectory, gp("arm_command_topic"), 10
        )

        rate = max(1.0, float(gp("publish_rate")))
        self._timer = self.create_timer(1.0 / rate, self._on_timer)

        self.get_logger().info(
            f"teleop: linear={self.lin_speed} m/s  strafe={self.str_speed} m/s "
            f"angular={self.ang_speed} rad/s  key_timeout={self.key_timeout}s "
            f"arm_step={self.arm_step} rad"
        )

    # ==================================================================
    #  key handling (runs in a background thread)
    # ==================================================================
    def handle_key(self, k: str):
        if not k:
            return
        now = time.monotonic()

        # ---------------- base ------------------------------------------
        if k in ("w", "W"):
            self._twist_target = np.array([self.lin_speed, 0.0, 0.0])
            self._last_motion_key = now
            return
        if k in ("s", "S"):
            self._twist_target = np.array([-self.lin_speed, 0.0, 0.0])
            self._last_motion_key = now
            return
        if k in ("a", "A"):
            self._twist_target = np.array([0.0, 0.0, self.ang_speed])
            self._last_motion_key = now
            return
        if k in ("d", "D"):
            self._twist_target = np.array([0.0, 0.0, -self.ang_speed])
            self._last_motion_key = now
            return
        if k in ("q", "Q"):
            self._twist_target = np.array([0.0, self.str_speed, 0.0])
            self._last_motion_key = now
            return
        if k in ("e", "E"):
            self._twist_target = np.array([0.0, -self.str_speed, 0.0])
            self._last_motion_key = now
            return
        if k == " ":
            self._twist_target = np.zeros(3)
            self._last_motion_key = now
            return

        # ---------------- arm -------------------------------------------
        arm_keys = {
            "1": (0, -1), "2": (0, +1),
            "3": (1, -1), "4": (1, +1),
            "5": (2, -1), "6": (2, +1),
            "7": (3, -1), "8": (3, +1),
            "9": (4, -1), "0": (4, +1),
            "-": (5, -1), "_": (5, -1),
            "=": (5, +1), "+": (5, +1),
        }
        if k in arm_keys:
            idx, sign = arm_keys[k]
            new = self._arm_targets.copy()
            new[idx] = new[idx] + sign * self.arm_step
            new = np.clip(new, self.arm_min, self.arm_max)
            if not np.allclose(new, self._arm_targets):
                self._arm_targets = new
                self._arm_dirty = True
            return

        if k in ("r", "R"):
            self._arm_targets = np.array(self.arm_home, dtype=float)
            self._arm_dirty = True
            self.get_logger().info(
                "arm reset -> " + np.array2string(self._arm_targets, precision=3)
            )
            return

        if k == "\x1b":                     # ESC
            self.get_logger().info("ESC received, shutting down")
            self._running = False

    # ==================================================================
    def _on_timer(self):
        if not self._running:
            return

        # ---- base: release the target if the keyboard goes quiet --------
        if (time.monotonic() - self._last_motion_key) > self.key_timeout:
            self._twist_target = np.zeros(3)

        twist = Twist()
        if not self._estop:
            twist.linear.x = float(self._twist_target[0])
            twist.linear.y = float(self._twist_target[1])
            twist.angular.z = float(self._twist_target[2])
        self._cmd_pub.publish(twist)

        # ---- arm: publish only when the target changed -------------------
        if self._arm_dirty:
            self._arm_dirty = False
            self._publish_arm_trajectory()

    def _publish_arm_trajectory(self):
        jt = JointTrajectory()
        jt.joint_names = list(self.arm_names)
        jt.header.stamp = self.get_clock().now().to_msg()

        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in self._arm_targets]
        pt.time_from_start = Duration(seconds=self.arm_goal_time).to_msg()

        jt.points = [pt]
        self._arm_pub.publish(jt)
        self.get_logger().info(
            "arm cmd -> " + np.array2string(self._arm_targets, precision=3),
            throttle_duration_sec=0.5,
        )

    # ==================================================================
    def _stop(self):
        self._twist_target = np.zeros(3)
        self._estop = True
        for _ in range(5):
            self._cmd_pub.publish(Twist())
            time.sleep(0.02)


def _read_keys(node: LeKiwiTeleop, stop_event: threading.Event):
    """Blocking key reader.  Raw tty mode when stdin is a TTY, line mode
    otherwise (so the node can also be fed from a pipe/script)."""
    tty_in = sys.stdin.isatty()
    old_settings = None
    if tty_in:
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
        except Exception as exc:                     # pragma: no cover
            node.get_logger().warn(f"cannot set cbreak mode: {exc}")
            tty_in = False

    try:
        while not stop_event.is_set():
            if tty_in:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                data = sys.stdin.read(1)
            else:
                line = sys.stdin.readline()
                if line == "":                      # EOF
                    time.sleep(0.1)
                    continue
                data = line
            for ch in data:
                node.handle_key(ch)
                if not node._running:
                    return
    finally:
        if old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = LeKiwiTeleop()
    print(HELP)

    stop_event = threading.Event()
    key_thread = threading.Thread(
        target=_read_keys, args=(node, stop_event), daemon=True
    )
    key_thread.start()

    try:
        while rclpy.ok() and node._running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            node._stop()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        print("teleop stopped")


if __name__ == "__main__":
    main()
