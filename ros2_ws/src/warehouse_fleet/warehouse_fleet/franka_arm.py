#!/usr/bin/env python3
"""Franka Panda pick-and-place controller.

The joint waypoints in ``config/warehouse.yaml`` were solved offline with Lula
IK for the exact base pose used by the simulator, so this node only has to
interpolate between them and stream ``sensor_msgs/JointState`` messages.  The
simulator mirrors the real cell: it closes the fingers, attaches the nearest
cube to ``panda_hand`` and releases it again when the gripper opens.

Topics::

    in   /franka/joint_states   sensor_msgs/JointState   from Isaac Sim
    out  /franka/joint_command  sensor_msgs/JointState   to Isaac Sim
    out  /franka/status         std_msgs/String          JSON progress
"""

from __future__ import annotations

import json
import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .config_utils import load_config


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class FrankaArm(Node):
    def __init__(self):
        super().__init__("franka_arm")
        self.declare_parameter("config", "")
        self.declare_parameter("loop", True)
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("rest_time", 2.0)

        cfg = load_config(self.get_parameter("config").value)
        spec = cfg["franka"]
        self.joint_names = list(spec["joint_names"])
        # The YAML waypoints only describe the seven arm joints; the two
        # finger joints are driven by the `gripper` field of each step.
        self.arm_joints = [
            name for name in self.joint_names
            if not name.startswith("panda_finger_joint")
        ]
        self.waypoints = spec["waypoints"]
        self.sequence = list(spec["sequence"])
        self.loop = bool(self.get_parameter("loop").value)
        self.speed = max(0.2, float(self.get_parameter("speed").value))
        self.rest_time = max(0.5, float(self.get_parameter("rest_time").value))

        self.feedback = None          # dict name -> position
        self.feedback_time = 0.0
        self.state = "WAIT"
        self.index = 0
        self.segment_start = 0.0
        self.segment_duration = 1.0
        self.start_joints = None
        self.start_gripper = 0.04
        self.target_joints = None
        self.target_gripper = 0.04
        self.wait_until = 0.0
        self.last_label = ""
        self.last_status_pub = 0.0
        self.cycles = 0

        self.pub_cmd = self.create_publisher(
            JointState, "/franka/joint_command", 10
        )
        self.pub_status = self.create_publisher(String, "/franka/status", 10)
        self.pub_events = self.create_publisher(String, "/fleet/task_events", 10)
        self.create_subscription(
            JointState, "/franka/joint_states", self.on_feedback, 10
        )
        self.create_timer(1.0 / 30.0, self.tick)
        self.get_logger().info(
            f"Franka controller ready: {len(self.sequence)} sequence steps, "
            f"loop={'on' if self.loop else 'off'}"
        )

    # ------------------------------------------------------------------
    def on_feedback(self, msg: JointState):
        if not msg.name:
            return
        self.feedback = {
            str(name): float(pos)
            for name, pos in zip(msg.name, msg.position)
            if math.isfinite(float(pos))
        }
        self.feedback_time = time.time()

    # ------------------------------------------------------------------
    def tick(self):
        if self.feedback is None or time.time() - self.feedback_time > 3.0:
            return

        now = time.time()
        if self.state == "WAIT":
            self.start_joints = self._measured()
            self.start_gripper = self._measured_gripper()
            self.index = 0
            self._begin_step(now)
            return

        if self.state == "MOVE":
            t = (now - self.segment_start) / max(1e-3, self.segment_duration)
            s = smoothstep(t)
            joints = [
                a + (b - a) * s
                for a, b in zip(self.start_joints, self.target_joints)
            ]
            gripper = self.start_gripper + (self.target_gripper - self.start_gripper) * s
            self._publish(joints, gripper)
            if t >= 1.0:
                self.state = "DWELL"
                self.segment_start = now
                self.segment_duration = float(self._step().get("dwell", 0.5))
                self.get_logger().info(
                    f"[{self.index + 1}/{len(self.sequence)}] {self.last_label}"
                )
            return

        if self.state == "DWELL":
            self._publish(self.target_joints, self.target_gripper)
            if now - self.segment_start >= self.segment_duration:
                self.index += 1
                if self.index >= len(self.sequence):
                    self.state = "REST"
                    self.segment_start = now
                    self.segment_duration = self.rest_time
                else:
                    self._begin_step(now)
            return

        if self.state == "REST":
            self._publish(self.target_joints, self.target_gripper)
            if now - self.segment_start >= self.segment_duration:
                if self.loop:
                    self.index = 0
                    self.start_joints = self._measured()
                    self.start_gripper = self._measured_gripper()
                    self.cycles += 1
                    self.pub_events.publish(String(data=json.dumps({
                        "event": "franka_cycle",
                        "robot": "franka",
                        "cycles": self.cycles,
                    })))
                    self._begin_step(now)
                    self.get_logger().info("Pick-and-place cycle complete - restarting")
                else:
                    self.get_logger().info("Pick-and-place sequence complete")
                    self.state = "DONE"
            return

    # ------------------------------------------------------------------
    def _step(self):
        return self.sequence[self.index]

    def _begin_step(self, now):
        step = self._step()
        waypoint = self.waypoints[step["waypoint"]]
        target = [float(v) for v in waypoint["joints"]]
        gripper = float(step.get("gripper", waypoint.get("gripper", 0.04)))

        if self.state in ("WAIT", "REST") or self.target_joints is None:
            # First step of a cycle: blend from measured feedback.
            if self.start_joints is None:
                self.start_joints = self._measured()
        else:
            # Following steps continue from the previous target so the
            # interpolation never jumps backwards.
            self.start_joints = list(self.target_joints)
            self.start_gripper = self.target_gripper

        max_delta = max(
            (abs(a - b) for a, b in zip(self.start_joints, target)), default=0.0
        )
        finger_delta = abs(gripper - self.start_gripper) * 8.0
        duration = max(0.55, max(max_delta, finger_delta) / (1.15 * self.speed))
        duration = min(duration, 4.5)

        self.target_joints = target
        self.target_gripper = gripper
        self.segment_start = now
        self.segment_duration = duration
        self.last_label = str(step.get("label", step["waypoint"]))
        self.state = "MOVE"

    def _measured(self):
        fb = self.feedback or {}
        return [float(fb.get(name, 0.0)) for name in self.arm_joints]

    def _measured_gripper(self):
        fb = self.feedback or {}
        value = fb.get("panda_finger_joint1", 0.04)
        return max(0.0, min(0.04, float(value)))

    def _publish(self, joints, gripper):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        arm = dict(zip(self.arm_joints, joints))
        names = []
        positions = []
        for name in self.joint_names:
            names.append(name)
            if name.startswith("panda_finger_joint"):
                positions.append(float(gripper))
            else:
                positions.append(float(arm.get(name, 0.0)))
        msg.name = names
        msg.position = positions
        self.pub_cmd.publish(msg)

        now = time.time()
        if now - self.last_status_pub >= 0.5:
            self.last_status_pub = now
            data = {
                "state": self.state,
                "step": self.index + 1,
                "steps": len(self.sequence),
                "label": self.last_label,
                "gripper": round(float(gripper), 4),
                "cycles": int(self.cycles),
            }
            self.pub_status.publish(String(data=json.dumps(data)))


def main(args=None):
    rclpy.init(args=args)
    node = FrankaArm()
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
