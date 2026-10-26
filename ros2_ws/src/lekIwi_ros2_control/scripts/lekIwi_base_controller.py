#!/usr/bin/env python3
# ============================================================================
# lekIwi_base_controller.py
# ----------------------------------------------------------------------------
# /cmd_vel (geometry_msgs/Twist)  ->  3-omni-wheel inverse kinematics
#                                  ->  /base_controller/commands
#                                     (std_msgs/Float64MultiArray, 3 velocities)
#
# /base_controller/commands is consumed by the ros2_control controller
# `base_controller` (forward_command_controller/ForwardCommandController), which
# writes the velocity command interfaces of the LeKiwi drive wheels.  The
# ros2_control hardware interface `lekIwi_isaac_hardware/LekIwiIsaacSystem`
# then forwards them to Isaac Sim over ROS 2 topics.
#
# ---------------------------------------------------------------------------
# INVERSE KINEMATICS -- derived from the URDF, NOT assumed
# ---------------------------------------------------------------------------
# Source: /home/arun/Desktop/lekiwi.urdf, forward kinematics to the root link
# "base_plate_layer1_v5" (the robot base frame).
#
#   joint                                 wheel centre p_i [m]     axis a_i
#   ST3215_Servo_Motor_v1_2_Revolute_60   ( 0.000610, -0.119200)  ( 0, -1, 0)
#   ST3215_Servo_Motor_v1_1_Revolute_62   ( 0.086298,  0.050528)  ( .866025, .5, 0)
#   ST3215_Servo_Motor_v1_Revolute_64     (-0.086215,  0.049072)  (-.866025, .5, 0)
#
# All three axes are horizontal.  For a wheel with horizontal spin axis `a`,
# no-slip contact gives
#
#       d_i . (v + w x p_i) = r * qdot_i
#
# where d_i = a_i x ẑ  is the wheel TRACTION direction (omni rollers transmit
# force only along d_i) and r is the wheel radius.
#
# Therefore
#
#   qdot_i = ( d_ix*vx + d_iy*vy + w*(d_iy*p_ix - d_ix*p_iy) ) / r
#          = ( d_i . [vx, vy] + w * c_i ) / r,     c_i = d_iy*p_ix - d_ix*p_iy
#
# Measured traction directions (base frame):
#   d_60 = (-1, 0)                 -> 180 deg
#   d_62 = ( 0.5, -0.866025)       -> -60 deg
#   d_64 = ( 0.5,  0.866025)       -> +60 deg
# i.e. 120 deg apart, which is exactly the LeKiwi 3-wheel omni layout.
#
# Measured moment arms about the base origin:
#   c_60 = -0.119200,  c_62 = -0.100000,  c_64 = -0.099200
# ============================================================================

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool

# ---------------------------------------------------------------------------
# URDF-derived geometry (see header).  Order == order of `wheel_joint_names`
# in config/joint_limits.yaml, which is also the order in which
# forward_command_controller expects the values.
# ---------------------------------------------------------------------------
WHEEL_JOINT_NAMES = [
    "ST3215_Servo_Motor_v1_2_Revolute_60",
    "ST3215_Servo_Motor_v1_1_Revolute_62",
    "ST3215_Servo_Motor_v1_Revolute_64",
]

# traction direction d_i = a_i x ẑ  (unit, horizontal) expressed in base frame
WHEEL_TRACTION = np.array([
    [-1.0,                 0.0],                   # Revolute_60
    [0.5, -0.8660254037844387],                    # Revolute_62
    [0.5,  0.8660254037844387],                    # Revolute_64
])

# wheel centre in the base frame [m]
WHEEL_POSITION = np.array([
    [ 0.000610, -0.119200],
    [ 0.086298,  0.050528],
    [-0.086215,  0.049072],
])

# moment arms c_i = d_iy*p_ix - d_ix*p_iy about the base origin
WHEEL_MOMENT = (
    WHEEL_TRACTION[:, 1] * WHEEL_POSITION[:, 0]
    - WHEEL_TRACTION[:, 0] * WHEEL_POSITION[:, 1]
)
# nominal (symmetric) wheel-base radius -- used to scale the rotational term
MEAN_MOMENT = float(np.mean(np.abs(WHEEL_MOMENT)))       # 0.106133 m


def wheel_velocities(vx, vy, wz, wheel_radius, base_radius, exact=True):
    """Map a base-frame twist [vx, vy, wz] onto the 3 wheel angular rates.

    Returns an array of 3 wheel velocities in rad/s (unclamped).
    """
    moment = WHEEL_MOMENT if exact else np.full(3, -abs(base_radius))
    # scale so that `base_radius` genuinely drives the rotational term
    moment = moment * (float(base_radius) / MEAN_MOMENT)
    lin = WHEEL_TRACTION @ np.array([vx, vy], dtype=float)
    return (lin + moment * wz) / float(wheel_radius)


class LeKiwiBaseController(Node):
    """Safety layer between /cmd_vel and the ros2_control base controller."""

    def __init__(self):
        super().__init__("lekIwi_base_controller")

        # ---- parameters -------------------------------------------------
        self.declare_parameter("wheel_radius", 0.0507845)
        self.declare_parameter("base_radius", 0.106135)
        self.declare_parameter("max_linear_velocity", 0.50)
        self.declare_parameter("max_angular_velocity", 1.50)
        self.declare_parameter("max_wheel_velocity", 30.0)
        self.declare_parameter("linear_accel_limit", 1.00)
        self.declare_parameter("angular_accel_limit", 3.00)
        self.declare_parameter("cmd_vel_timeout", 0.50)
        self.declare_parameter("use_exact_geometry", True)
        self.declare_parameter("update_rate", 20.0)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("command_topic", "/base_controller/commands")
        self.declare_parameter("estop_topic", "/emergency_stop")
        self.declare_parameter("wheel_joint_names", WHEEL_JOINT_NAMES)

        gp = lambda n: self.get_parameter(n).value
        self.wheel_radius = float(gp("wheel_radius"))
        self.base_radius = float(gp("base_radius"))
        self.v_max = float(gp("max_linear_velocity"))
        self.w_max = float(gp("max_angular_velocity"))
        self.qd_max = float(gp("max_wheel_velocity"))
        self.a_lin = float(gp("linear_accel_limit"))
        self.a_ang = float(gp("angular_accel_limit"))
        self.timeout = float(gp("cmd_vel_timeout"))
        self.exact = bool(gp("use_exact_geometry"))
        self.wheel_joint_names = list(gp("wheel_joint_names"))
        if len(self.wheel_joint_names) != 3:
            raise ValueError(
                "wheel_joint_names must contain exactly 3 entries, got "
                f"{len(self.wheel_joint_names)}"
            )

        # ---- state --------------------------------------------------------
        self._target = np.zeros(3)     # requested twist [vx, vy, wz]
        self._current = np.zeros(3)    # ramped twist actually sent
        self._last_cmd_time = None
        self._estop = False
        self._shutting_down = False

        # ---- I/O ----------------------------------------------------------
        # best-effort: always compatible with any publisher QoS
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._pub = self.create_publisher(
            Float64MultiArray, gp("command_topic"), 10
        )
        self._sub = self.create_subscription(
            Twist, gp("cmd_vel_topic"), self._on_cmd_vel, sub_qos
        )
        self._estop_sub = self.create_subscription(
            Bool, gp("estop_topic"), self._on_estop, 10
        )

        rate = max(1.0, float(gp("update_rate")))
        self._dt = 1.0 / rate
        self._timer = self.create_timer(self._dt, self._on_timer)

        self.get_logger().info(
            "lekIwi_base_controller ready | "
            f"wheel_radius={self.wheel_radius:.5f} m  "
            f"base_radius={self.base_radius:.5f} m  "
            f"v_max={self.v_max} m/s  w_max={self.w_max} rad/s  "
            f"timeout={self.timeout:.2f}s  exact_geometry={self.exact}"
        )
        self.get_logger().info(
            "IK: qdot = (d . [vx,vy] + w * c) / r with "
            f"d={np.round(WHEEL_TRACTION, 6).tolist()} "
            f"c={np.round(WHEEL_MOMENT, 6).tolist()}"
        )

    # ------------------------------------------------------------------
    def _now(self):
        # wall clock for the watchdog: it must keep running even if /clock
        # (Isaac Sim) stalls, otherwise a paused sim would never trigger a stop
        import time
        return time.monotonic()

    def _on_cmd_vel(self, msg: Twist):
        raw = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=float)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- velocity limits (component-wise, then magnitude clamp) -------
        lin_mag = math.hypot(raw[0], raw[1])
        if lin_mag > self.v_max and lin_mag > 0.0:
            raw[0:2] *= self.v_max / lin_mag
        raw[2] = float(np.clip(raw[2], -self.w_max, self.w_max))

        self._target = raw
        self._last_cmd_time = self._now()

    def _on_estop(self, msg: Bool):
        if bool(msg.data):
            if not self._estop:
                self.get_logger().warn("EMERGENCY STOP asserted")
            self._estop = True
            self._target[:] = 0.0
        else:
            if self._estop:
                self.get_logger().warn("Emergency stop released")
            self._estop = False

    # ------------------------------------------------------------------
    def _on_timer(self):
        now = self._now()

        # ---- watchdog: no /cmd_vel recently -> stop ----------------------
        timed_out = (
            self._last_cmd_time is None
            or (now - self._last_cmd_time) > self.timeout
        )
        if timed_out:
            target = np.zeros(3)
        elif self._estop:
            target = np.zeros(3)
        else:
            target = self._target

        # ---- acceleration limiting (slew rate) ---------------------------
        dv = target - self._current
        max_dv = self.a_lin * self._dt
        max_dw = self.a_ang * self._dt
        dv[0] = float(np.clip(dv[0], -max_dv, max_dv))
        dv[1] = float(np.clip(dv[1], -max_dv, max_dv))
        dv[2] = float(np.clip(dv[2], -max_dw, max_dw))
        self._current = self._current + dv

        # fully stop as soon as the watchdog trips, otherwise the ramp would
        # keep the base rolling for `v_max / a_lin` seconds after the timeout
        if timed_out or self._estop:
            self._current = target

        # ---- inverse kinematics -----------------------------------------
        vx, vy, wz = (float(self._current[0]),
                      float(self._current[1]),
                      float(self._current[2]))
        qd = wheel_velocities(
            vx, vy, wz, self.wheel_radius, self.base_radius, self.exact
        )
        qd = np.clip(qd, -self.qd_max, self.qd_max)
        qd = np.nan_to_num(qd, nan=0.0, posinf=0.0, neginf=0.0)

        self._publish(qd)

    def _publish(self, qd):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in qd]
        self._pub.publish(msg)

    # ------------------------------------------------------------------
    def _stop_and_shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        self.get_logger().info("Shutting down: commanding zero wheel velocity")
        # publish a burst of zeros so Isaac Sim is sure to receive them
        zero = Float64MultiArray()
        zero.data = [0.0, 0.0, 0.0]
        for _ in range(10):
            self._pub.publish(zero)
            try:
                rclpy.spin_once(self, timeout_sec=0.02)
            except Exception:
                break


def main(args=None):
    rclpy.init(args=args)
    node = LeKiwiBaseController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop_and_shutdown()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
