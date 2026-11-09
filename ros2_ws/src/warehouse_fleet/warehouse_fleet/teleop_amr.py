#!/usr/bin/env python3
"""Keyboard teleop for one AMR (useful when the fleet manager is not running).

    ros2 run warehouse_fleet teleop_amr --ros-args -p robot:=amr2

Keys::

    w / s   forward / reverse        a / d   turn left / right
    x       stop                     q       quit
    + / -   increase / decrease speed

The command is latched: the robot keeps moving until another key is pressed.
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

HELP = """
Warehouse AMR teleop
  w/s : forward / reverse      a/d : turn left / right
  x   : stop                   +/- : speed up / down
  q   : quit
"""


class TeleopAmr(Node):
    def __init__(self):
        super().__init__("teleop_amr")
        self.declare_parameter("robot", "amr1")
        self.declare_parameter("direct", False)
        self.name = str(self.get_parameter("robot").value).strip("/")
        self.direct = bool(self.get_parameter("direct").value)
        self.speed = 0.45
        self.vx = 0.0
        self.wz = 0.0
        topic = f"/{self.name}/cmd_vel" if self.direct else (
            f"/{self.name}/teleop_cmd_vel"
        )
        self.pub = self.create_publisher(Twist, topic, 10)
        self.pub_control = self.create_publisher(String, "/fleet/control", 10)
        self.active = False

    def request_mode(self, enabled):
        if self.direct:
            return
        msg = String()
        msg.data = (
            '{"action":"set_teleop","robot":"%s","enabled":%s}'
            % (self.name, "true" if enabled else "false")
        )
        self.pub_control.publish(msg)
        self.active = bool(enabled)

    def send(self):
        msg = Twist()
        msg.linear.x = float(self.vx)
        msg.angular.z = float(self.wz)
        self.pub.publish(msg)


def read_key(timeout=0.1):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return ""
    return sys.stdin.read(1)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopAmr()
    if not sys.stdin.isatty():
        print("teleop_amr requires an interactive terminal", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(HELP)
    if node.direct:
        print(f"publishing direct to /{node.name}/cmd_vel\n")
    else:
        print(f"requesting fleet teleop and publishing /{node.name}/teleop_cmd_vel\n")
        node.request_mode(True)

    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            key = read_key(0.08)
            if key:
                key = key.lower()
                if key == "w":
                    node.vx, node.wz = node.speed, 0.0
                elif key == "s":
                    node.vx, node.wz = -node.speed, 0.0
                elif key == "a":
                    node.vx, node.wz = 0.0, 0.9
                elif key == "d":
                    node.vx, node.wz = 0.0, -0.9
                elif key == "x" or key == " ":
                    node.vx = node.wz = 0.0
                elif key in ("+", "="):
                    node.speed = min(1.0, node.speed + 0.05)
                    print(f"speed {node.speed:.2f} m/s")
                elif key in ("-", "_"):
                    node.speed = max(0.10, node.speed - 0.05)
                    print(f"speed {node.speed:.2f} m/s")
                elif key == "q":
                    break
                else:
                    continue
                print(
                    f"\r{node.name}: vx={node.vx:+.2f} m/s  wz={node.wz:+.2f} rad/s   ",
                    end="",
                    flush=True,
                )
            node.send()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.vx = node.wz = 0.0
        node.send()
        if node.active:
            node.request_mode(False)
        print("\nstopped")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
