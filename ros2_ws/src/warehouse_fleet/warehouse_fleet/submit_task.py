#!/usr/bin/env python3
"""Submit an extra transport task to the running fleet manager.

Examples::

    ros2 run warehouse_fleet submit_task inbound staging
    ros2 run warehouse_fleet submit_task inbound outbound --priority 1
    ros2 run warehouse_fleet submit_task --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .config_utils import load_config, stations_by_name


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Submit a pickup/drop-off task to /fleet/submit_task"
    )
    parser.add_argument("pickup", nargs="?", help="pickup station name")
    parser.add_argument("dropoff", nargs="?", help="drop-off station name")
    parser.add_argument("--priority", type=int, default=50)
    parser.add_argument("--list", action="store_true", help="list station names")
    parser.add_argument("--config", default="")
    parser.add_argument("--repeat", type=int, default=1,
                        help="publish the task N times (default 1)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config)
    stations = stations_by_name(cfg)

    if args.list:
        print("Stations:")
        for name, spec in stations.items():
            print(
                f"  {name:<10} ({spec['x']:+.2f}, {spec['y']:+.2f})  "
                f"{spec.get('kind', '')}"
            )
        return 0

    if not args.pickup or not args.dropoff:
        print("error: pickup and dropoff are required (try --list)", file=sys.stderr)
        return 2
    for name in (args.pickup, args.dropoff):
        if name not in stations:
            print(f"error: unknown station '{name}' (try --list)", file=sys.stderr)
            return 2
    if args.pickup == args.dropoff:
        print("error: pickup and dropoff must differ", file=sys.stderr)
        return 2

    rclpy.init()
    node = Node("submit_task")
    pub = node.create_publisher(String, "/fleet/submit_task", 10)
    payload = json.dumps(
        {
            "pickup": args.pickup,
            "dropoff": args.dropoff,
            "priority": int(args.priority),
        }
    )
    msg = String(data=payload)

    # Wait briefly for the fleet manager so the message is not lost.
    deadline = time.time() + 3.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    for _ in range(max(1, args.repeat)):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.1)

    if pub.get_subscription_count() == 0:
        print(
            "warning: no subscriber on /fleet/submit_task - is the fleet "
            "manager running?",
            file=sys.stderr,
        )
    print(f"submitted: {args.pickup} -> {args.dropoff} (priority {args.priority})")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
