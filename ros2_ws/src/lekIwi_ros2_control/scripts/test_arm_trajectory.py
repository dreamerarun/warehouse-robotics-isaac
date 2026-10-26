#!/usr/bin/env python3
# ============================================================================
# test_arm_trajectory.py -- exercise both arm command interfaces
# ----------------------------------------------------------------------------
#   1) trajectory_msgs/JointTrajectory on /arm_controller/joint_trajectory
#   2) control_msgs/action/FollowJointTrajectory on
#      /arm_controller/follow_joint_trajectory
#
# Usage:
#   ros2 run lekIwi_ros2_control test_arm_trajectory                    # topic
#   ros2 run lekIwi_ros2_control test_arm_trajectory --use-action       # action
#   ros2 run lekIwi_ros2_control test_arm_trajectory --joints-only      # list only
# ============================================================================

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINT_NAMES = [
    "STS3215_03a_v1_Revolute_45",
    "STS3215_03a_v1_1_Revolute_49",
    "STS3215_03a_v1_2_Revolute_51",
    "STS3215_03a_v1_3_Revolute_53",
    "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "STS3215_03a_v1_4_Revolute_57",
]


def send_topic(node, topic):
    pub = node.create_publisher(JointTrajectory, topic, 10)
    time.sleep(1.0)  # let discovery complete

    waypoints = [
        [0.0, -0.5, 0.5, 0.0, 0.0, 0.0],
        [0.4, -0.2, 0.8, 0.3, 0.0, 0.4],
        [-0.4, -0.2, 0.8, -0.3, 0.0, -0.4],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    for i, wp in enumerate(waypoints):
        jt = JointTrajectory()
        jt.joint_names = list(ARM_JOINT_NAMES)
        jt.header.stamp = node.get_clock().now().to_msg()
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in wp]
        pt.time_from_start = Duration(seconds=2.0).to_msg()
        jt.points = [pt]
        pub.publish(jt)
        node.get_logger().info(f"[{i + 1}/{len(waypoints)}] {wp}")
        # wait for the action/goal window to elapse before the next point
        end = time.time() + 2.4
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("topic trajectory test finished")
    return True


def send_action(node, action_name):
    try:
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
    except ImportError as exc:
        node.get_logger().error(
            f"action dependencies missing ({exc}); "
            "install ros-humble-control-msgs"
        )
        return False

    client = ActionClient(node, FollowJointTrajectory, action_name)
    node.get_logger().info(f"waiting for action server '{action_name}' ...")
    if not client.wait_for_server(timeout_sec=10.0):
        node.get_logger().error("action server not available")
        return False

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINT_NAMES)
    pt = JointTrajectoryPoint()
    pt.positions = [0.0, -0.4, 0.6, 0.0, 0.0, 0.3]
    pt.time_from_start = Duration(seconds=3.0).to_msg()
    goal.trajectory.points = [pt]

    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future, timeout_sec=10.0)
    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        node.get_logger().error("goal rejected")
        return False

    node.get_logger().info("goal accepted, waiting for result ...")
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=20.0)
    result = result_future.result()
    node.get_logger().info(f"result: {result.result.error_code} "
                           f"({result.result.error_string})")
    return result.result.error_code == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="LeKiwi arm test")
    parser.add_argument("--topic", default="/arm_controller/joint_trajectory")
    parser.add_argument("--action", default="/arm_controller/follow_joint_trajectory")
    parser.add_argument("--use-action", action="store_true")
    parser.add_argument("--joints-only", action="store_true")
    args = parser.parse_args(argv)

    if args.joints_only:
        print("arm joints:")
        for i, j in enumerate(ARM_JOINT_NAMES, 1):
            print(f"  J{i}: {j}")
        return 0

    rclpy.init()
    node = rclpy.create_node("test_arm_trajectory")
    try:
        ok = send_action(node, args.action) if args.use_action else \
             send_topic(node, args.topic)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
