#!/usr/bin/env python3
"""Browser operations dashboard for the Isaac Sim warehouse.

The node is deliberately self-contained: rclpy for the ROS 2 link and the
Python standard-library HTTP server for the UI.  It exposes the fleet map,
live robot/process state, task submission, autonomy controls and a dead-man
teleoperation channel without adding a web-framework dependency to the demo.

HTTP API
--------
    GET  /                 dashboard
    GET  /api/state        fleet, process, robot, event and map metadata
    GET  /api/map          occupancy grid (base64 int8 cells)
    GET  /api/health       liveness
    POST /api/task         {"pickup": "inbound", "dropoff": "staging"}
    POST /api/control      {"action": "set_mode", "mode": "PAUSED"}
    POST /api/control      {"action": "set_teleop", "robot": "amr1",
                            "enabled": true}
    POST /api/control      {"action": "teleop", "robot": "amr1",
                            "linear": 0.25, "angular": 0.0}
    POST /api/control      {"action": "process", "command": "reset_pallet"}

ROS topics used by the node are documented in README.md.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .config_utils import load_config, stations_by_name


def _json_bytes(value):
    return json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")


class DashboardNode(Node):
    def __init__(self):
        super().__init__("warehouse_dashboard")
        self.declare_parameter("config", "")
        self.declare_parameter("host", "")
        self.declare_parameter("port", 0)

        self.cfg = load_config(self.get_parameter("config").value)
        dash_cfg = self.cfg.get("dashboard", {})
        self.host = str(self.get_parameter("host").value or
                        dash_cfg.get("host", "127.0.0.1"))
        self.port = int(self.get_parameter("port").value or
                        dash_cfg.get("port", 8080))
        self.state_period = float(dash_cfg.get("state_period", 0.5))
        self.event_limit = int(dash_cfg.get("event_history", 120))
        self.stations = stations_by_name(self.cfg)
        self.amr_names = [str(a["name"]) for a in self.cfg.get("amrs", [])]

        self._lock = threading.RLock()
        self._fleet = {}
        self._humanoid = {}
        self._franka = {}
        self._ur10 = {}
        self._process = {}
        self._map_meta = {}
        self._map_data = b""
        self._map_version = 0
        self._map_time = 0.0
        self._seen = {}
        self._events = deque(maxlen=self.event_limit)

        self._scene = self._public_scene()
        self._config = self._public_config()

        self.pub_task = self.create_publisher(String, "fleet/submit_task", 10)
        self.pub_control = self.create_publisher(String, "fleet/control", 20)
        self.pub_ur10 = self.create_publisher(Twist, "ur10/command", 10)
        self._teleop_pubs = {
            name: self.create_publisher(Twist, f"{name}/teleop_cmd_vel", 10)
            for name in self.amr_names
        }

        self.create_subscription(String, "fleet/status", self._on_fleet, 10)
        self.create_subscription(OccupancyGrid, "fleet/map", self._on_map, 1)
        self.create_subscription(String, "fleet/task_events", self._on_event, 30)
        self.create_subscription(String, "humanoid/status", self._on_humanoid, 10)
        self.create_subscription(String, "franka/status", self._on_franka, 10)
        self.create_subscription(String, "ur10/status", self._on_ur10, 10)
        self.create_subscription(
            String, "warehouse/process_status", self._on_process, 10
        )

        self._httpd = None
        self._http_thread = None
        self._static_root = self._find_static_root()
        self.start_http()
        self.get_logger().info(
            f"Dashboard ready at http://{self.host}:{self.port} "
            f"({len(self.amr_names)} AMRs, {len(self.stations)} stations)"
        )

    # ------------------------------------------------------------------
    # Configuration exposed to the UI
    # ------------------------------------------------------------------
    def _public_scene(self):
        scene = self.cfg.get("scene", {})
        keys = (
            "conveyors", "stations", "pallets", "racks", "static_obstacles",
            "parking", "industrial", "forklift", "people", "walls",
        )
        return {
            "world": self.cfg.get("world", {}),
            "scene": {key: scene.get(key, []) for key in keys},
        }

    def _public_config(self):
        fleet = self.cfg.get("fleet", {})
        tasks = self.cfg.get("tasks", {})
        return {
            "amrs": self.cfg.get("amrs", []),
            "fleet": {
                key: fleet.get(key)
                for key in (
                    "max_active_amrs", "map_resolution", "teleop_linear_speed",
                    "teleop_angular_speed", "teleop_timeout", "status_period",
                )
            },
            "tasks": {
                "managed": bool(tasks.get("managed", False)),
                "templates": tasks.get("list", []),
                "initial_inventory": tasks.get("initial_inventory", {}),
                "capacity": tasks.get("capacity", {}),
                "max_pending": tasks.get("max_pending", 8),
            },
            "dashboard": {
                "host": self.host,
                "port": self.port,
            },
            "industrial": self.cfg.get("scene", {}).get("industrial", {}),
        }

    # ------------------------------------------------------------------
    # ROS subscriptions
    # ------------------------------------------------------------------
    def _stamp(self, key):
        self._seen[key] = time.time()

    def _store_json(self, attr, text, key):
        try:
            value = json.loads(text)
        except Exception:
            value = {"raw": text}
        with self._lock:
            setattr(self, attr, value)
            self._seen[key] = time.time()

    def _on_fleet(self, msg):
        self._store_json("_fleet", msg.data, "fleet/status")

    def _on_humanoid(self, msg):
        self._store_json("_humanoid", msg.data, "humanoid/status")

    def _on_franka(self, msg):
        self._store_json("_franka", msg.data, "franka/status")

    def _on_ur10(self, msg):
        self._store_json("_ur10", msg.data, "ur10/status")

    def _on_process(self, msg):
        self._store_json("_process", msg.data, "warehouse/process_status")

    def _on_event(self, msg):
        try:
            event = json.loads(msg.data)
        except Exception:
            event = {"event": "raw", "data": msg.data}
        event.setdefault("received", round(time.time(), 3))
        with self._lock:
            self._events.append(event)
            self._seen["fleet/task_events"] = time.time()

    def _on_map(self, msg):
        data = np.asarray(msg.data, dtype=np.int8).tobytes()
        with self._lock:
            self._map_meta = {
                "frame_id": msg.header.frame_id or "map",
                "width": int(msg.info.width),
                "height": int(msg.info.height),
                "resolution": float(msg.info.resolution),
                "origin": [
                    float(msg.info.origin.position.x),
                    float(msg.info.origin.position.y),
                ],
                "blocked_cells": int(np.count_nonzero(np.asarray(msg.data) > 50)),
                "stamp": {
                    "sec": int(msg.header.stamp.sec),
                    "nanosec": int(msg.header.stamp.nanosec),
                },
            }
            self._map_data = data
            self._map_version += 1
            self._map_time = time.time()
            self._seen["fleet/map"] = self._map_time

    # ------------------------------------------------------------------
    # HTTP API
    # ------------------------------------------------------------------
    def state_snapshot(self):
        now = time.time()
        with self._lock:
            seen = {key: round(now - stamp, 2) for key, stamp in self._seen.items()}
            fleet = dict(self._fleet)
            humanoid = dict(self._humanoid)
            franka = dict(self._franka)
            ur10 = dict(self._ur10)
            process = dict(self._process)
            map_meta = dict(self._map_meta)
            events = list(self._events)[-self.event_limit:]
            map_version = self._map_version
            map_age = round(now - self._map_time, 2) if self._map_time else None
        online = bool(fleet) and seen.get("fleet/status", 999) < 4.0
        return {
            "server_time": round(now, 3),
            "online": online,
            "ros": {
                "topics": seen,
                "map_version": map_version,
                "map_age": map_age,
            },
            "fleet": fleet,
            "humanoid": humanoid,
            "franka": franka,
            "ur10": ur10,
            "process": process,
            "map": map_meta,
            "events": events,
            "scene": self._scene,
            "config": self._config,
        }

    def map_payload(self):
        with self._lock:
            return {
                "version": self._map_version,
                "meta": dict(self._map_meta),
                "encoding": "int8-base64",
                "data": base64.b64encode(self._map_data).decode("ascii"),
            }

    def submit_task(self, data):
        pickup = str(data.get("pickup", "")).strip()
        dropoff = str(data.get("dropoff", "")).strip()
        if pickup not in self.stations or dropoff not in self.stations:
            raise ValueError("unknown pickup/dropoff station")
        if pickup == dropoff:
            raise ValueError("pickup and dropoff must differ")
        task = {
            "pickup": pickup,
            "dropoff": dropoff,
            "priority": max(0, min(99, int(data.get("priority", 20)))),
            "cargo": str(data.get("cargo", "operator_tote")),
            "description": str(data.get("description", "Dashboard work order")),
        }
        self.pub_task.publish(String(data=json.dumps(task)))
        return task

    def control(self, data):
        action = str(data.get("action", "")).strip().lower()
        if action in ("set_mode", "mode", "estop", "pause", "resume"):
            mode = str(data.get("mode", "")).upper()
            if action == "estop":
                mode = "ESTOP"
            elif action == "pause":
                mode = "PAUSED"
            elif action == "resume":
                mode = "AUTO"
            if mode not in ("AUTO", "PAUSED", "ESTOP"):
                raise ValueError("mode must be AUTO, PAUSED or ESTOP")
            self.pub_control.publish(
                String(data=json.dumps({"action": "set_mode", "mode": mode}))
            )
            return {"action": "set_mode", "mode": mode}
        if action in ("set_teleop", "teleop_enable"):
            robot = str(data.get("robot", "")).strip("/")
            if robot not in self.amr_names:
                raise ValueError("unknown robot")
            enabled = bool(data.get("enabled", True))
            self.pub_control.publish(
                String(data=json.dumps({
                    "action": "set_teleop",
                    "robot": robot,
                    "enabled": enabled,
                }))
            )
            if not enabled:
                self.pub_teleop(robot, 0.0, 0.0)
            return {"action": "set_teleop", "robot": robot, "enabled": enabled}
        if action in ("teleop", "drive"):
            robot = str(data.get("robot", "")).strip("/")
            if robot not in self.amr_names:
                raise ValueError("unknown robot")
            linear = max(-1.0, min(1.0, float(data.get("linear", 0.0))))
            angular = max(-1.5, min(1.5, float(data.get("angular", 0.0))))
            self.pub_teleop(robot, linear, angular)
            return {"action": "teleop", "robot": robot,
                    "linear": linear, "angular": angular}
        if action == "cancel_task":
            robot = str(data.get("robot", "")).strip("/")
            if robot not in self.amr_names:
                raise ValueError("unknown robot")
            self.pub_control.publish(
                String(data=json.dumps({"action": "cancel_task", "robot": robot}))
            )
            return {"action": "cancel_task", "robot": robot}
        if action == "process":
            command = str(data.get("command", "")).strip().lower()
            self.pub_process(command)
            return {"action": "process", "command": command}
        raise ValueError("unsupported control action")

    def pub_teleop(self, robot, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._teleop_pubs[robot].publish(msg)

    def pub_process(self, command):
        msg = Twist()
        if command == "pause":
            msg.linear.x = -1.0
        elif command == "resume":
            msg.linear.x = 1.0
        elif command == "reset_pallet":
            msg.linear.y = 1.0
        else:
            raise ValueError("process command must be pause, resume or reset_pallet")

        def pulse():
            self.pub_ur10.publish(msg)
            threading.Timer(0.35, self._zero_process).start()

        pulse()

    def _zero_process(self):
        try:
            self.pub_ur10.publish(Twist())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------
    def _find_static_root(self):
        candidates = []
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, "web"))
        try:
            from ament_index_python.packages import get_package_share_directory

            candidates.append(
                os.path.join(get_package_share_directory("warehouse_fleet"), "web")
            )
        except Exception:
            pass
        for root in candidates:
            if os.path.isfile(os.path.join(root, "index.html")):
                return root
        return candidates[0]

    def start_http(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "WarehouseOps/1.0"

            def log_message(self, fmt, *args):
                return

            def _send(self, status, body, content_type, cache="no-store"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", cache)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass

            def _json(self, status, value):
                self._send(status, _json_bytes(value), "application/json")

            def _read_json(self):
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length < 0 or length > 1_000_000:
                    raise ValueError("request body too large")
                raw = self.rfile.read(length) if length else b"{}"
                return json.loads(raw.decode("utf-8"))

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self):
                path = unquote(urlparse(self.path).path)
                try:
                    if path in ("/", "/index.html"):
                        return self._serve_static("index.html")
                    if path == "/api/state":
                        return self._json(200, node.state_snapshot())
                    if path == "/api/map":
                        return self._json(200, node.map_payload())
                    if path == "/api/health":
                        return self._json(200, {"ok": True, "time": time.time()})
                    if path == "/api/config":
                        return self._json(200, {
                            "scene": node._scene,
                            "config": node._config,
                        })
                    if path.startswith("/static/"):
                        return self._serve_static(path[len("/static/"):])
                    if path == "/favicon.ico":
                        self.send_response(204)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    return self._json(404, {"error": "not found"})
                except Exception as exc:
                    return self._json(500, {"error": str(exc)})

            def _serve_static(self, name):
                safe = os.path.basename(name)
                target = os.path.join(node._static_root, safe)
                if not os.path.isfile(target):
                    return self._json(404, {"error": "not found"})
                ext = os.path.splitext(safe)[1].lower()
                ctype = {
                    ".html": "text/html; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".json": "application/json",
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                }.get(ext, "application/octet-stream")
                with open(target, "rb") as fh:
                    body = fh.read()
                return self._send(200, body, ctype, cache="no-cache")

            def do_POST(self):
                path = unquote(urlparse(self.path).path)
                try:
                    data = self._read_json()
                    if path == "/api/task":
                        return self._json(200, {
                            "ok": True,
                            "task": node.submit_task(data),
                        })
                    if path == "/api/control":
                        return self._json(200, {
                            "ok": True,
                            "result": node.control(data),
                        })
                    return self._json(404, {"error": "not found"})
                except ValueError as exc:
                    return self._json(400, {"ok": False, "error": str(exc)})
                except Exception as exc:
                    return self._json(500, {"ok": False, "error": str(exc)})

        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
            self._httpd.daemon_threads = True
        except OSError as exc:
            self.get_logger().error(
                f"dashboard HTTP server could not bind {self.host}:{self.port}: {exc}"
            )
            self._httpd = None
            return
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, name="warehouse-http", daemon=True
        )
        self._http_thread.start()

    def stop_http(self):
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None

    def destroy_node(self):
        self.stop_http()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
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
