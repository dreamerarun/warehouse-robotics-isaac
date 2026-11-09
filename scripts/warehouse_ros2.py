#!/usr/bin/env python3
"""ROS 2 bridge plumbing for the warehouse simulation.

The simulation never imports host ROS 2 Python.  All topics are created with
OmniGraph nodes from ``isaacsim.ros2.bridge`` so the 3.11 interpreter only ever
uses the bridge's bundled Humble runtime.

Topic contract per AMR (namespace = robot name)::

    /<ns>/cmd_vel   geometry_msgs/Twist            sim <- autonomy
    /<ns>/odom      nav_msgs/Odometry              sim -> autonomy
    /<ns>/scan      sensor_msgs/LaserScan          sim -> autonomy
    /tf             tf2_msgs/TFMessage             sim -> autonomy (all robots)

Franka::

    /franka/joint_command  sensor_msgs/JointState  sim <- autonomy
    /franka/joint_states   sensor_msgs/JointState  sim -> autonomy

Humanoid::

    /humanoid/cmd_vel      geometry_msgs/Twist     sim <- autonomy
    /humanoid/odom         nav_msgs/Odometry       sim -> autonomy
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TwistHandle:
    linear: object = None
    angular: object = None
    last: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    received: bool = False

    def read(self):
        try:
            lin = np.asarray(self.linear.get(), dtype=float).reshape(-1)
            ang = np.asarray(self.angular.get(), dtype=float).reshape(-1)
        except Exception:
            return self.last.copy()
        if lin.size < 3 or ang.size < 3:
            return self.last.copy()
        self.received = True
        self.last = np.array(
            [lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]], dtype=float
        )
        return self.last.copy()


@dataclass
class JointCommandHandle:
    names: object = None
    positions: object = None
    last_names: list = field(default_factory=list)
    last_positions: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def read(self):
        try:
            names = self.names.get() if self.names is not None else None
            pos = self.positions.get() if self.positions is not None else None
        except Exception:
            return None
        if names is None or pos is None:
            return None
        names = [str(n) for n in names]
        pos = np.asarray(pos, dtype=float).reshape(-1)
        if not names or names == ["None"] or pos.size == 0:
            return None
        self.last_names = names
        self.last_positions = pos.copy()
        return names, pos


class RosBridge:
    """Create and poll the OmniGraph action graphs."""

    def __init__(self, app, domain_id=None):
        self.app = app
        self.domain_id = domain_id
        self.enabled = False
        self.errors = []
        self.graphs = []
        self.twists = {}
        self.joint_commands = {}
        self.string_pubs = {}

    # -- lifecycle --------------------------------------------------------
    def start(self):
        try:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.ros2.bridge")
            for _ in range(8):
                self.app.update()
            import omni.kit.app

            manager = omni.kit.app.get_app().get_extension_manager()
            self.enabled = bool(manager.is_extension_enabled("isaacsim.ros2.bridge"))
            if not self.enabled:
                self.errors.append("isaacsim.ros2.bridge did not enable")
        except Exception as exc:  # pragma: no cover - startup diagnostics
            self.errors.append(f"enable extension: {exc}")
            self.enabled = False
        return self.enabled

    def _graph(self, path, nodes, values, connects):
        import omni.graph.core as og

        og.Controller.edit(
            {"graph_path": path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: nodes,
                og.Controller.Keys.SET_VALUES: values,
                og.Controller.Keys.CONNECT: connects,
            },
        )
        self.graphs.append(path)

    # -- graphs -----------------------------------------------------------
    def add_amr(self, name, robot_path, chassis_path, lidar_path, laser_frame,
                lidar_mount):
        """Differential AMR: odom + 2-D scan + tf + cmd_vel input."""
        if not self.enabled:
            return False
        import omni.graph.core as og
        import usdrt

        graph = f"/World/ROS2/{name}"
        try:
            self._graph(
                graph,
                [
                    ("Tick", "omni.graph.action.OnPlaybackTick"),
                    ("Time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ReadPose", "isaacsim.core.nodes.IsaacReadWorldPose"),
                    ("Odom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("PubOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    ("ReadLidar", "isaacsim.sensors.physx.IsaacReadLidarBeams"),
                    ("PubScan", "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
                    ("TfOdom", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    ("TfLaser", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ],
                [
                    ("Odom.inputs:chassisPrim", [usdrt.Sdf.Path(chassis_path)]),
                    ("ReadPose.inputs:prim", [usdrt.Sdf.Path(chassis_path)]),
                    ("PubOdom.inputs:topicName", f"{name}/odom"),
                    ("PubOdom.inputs:odomFrameId", "odom"),
                    ("PubOdom.inputs:chassisFrameId", f"{name}/base_link"),
                    ("PubOdom.inputs:publishRawVelocities", False),
                    ("ReadLidar.inputs:lidarPrim", [usdrt.Sdf.Path(lidar_path)]),
                    ("PubScan.inputs:topicName", f"{name}/scan"),
                    ("PubScan.inputs:frameId", laser_frame),
                    ("TfOdom.inputs:topicName", "tf"),
                    ("TfOdom.inputs:parentFrameId", "odom"),
                    ("TfOdom.inputs:childFrameId", f"{name}/base_link"),
                    ("TfLaser.inputs:topicName", "tf"),
                    ("TfLaser.inputs:parentFrameId", f"{name}/base_link"),
                    ("TfLaser.inputs:childFrameId", laser_frame),
                    ("TfLaser.inputs:translation", [float(v) for v in lidar_mount]),
                    ("TfLaser.inputs:rotation", [0.0, 0.0, 0.0, 1.0]),
                    ("SubTwist.inputs:topicName", f"{name}/cmd_vel"),
                ],
                [
                    ("Tick.outputs:tick", "Odom.inputs:execIn"),
                    ("Odom.outputs:execOut", "PubOdom.inputs:execIn"),
                    ("ReadPose.outputs:translation", "PubOdom.inputs:position"),
                    ("ReadPose.outputs:orientation", "PubOdom.inputs:orientation"),
                    ("Odom.outputs:linearVelocity", "PubOdom.inputs:linearVelocity"),
                    ("Odom.outputs:angularVelocity", "PubOdom.inputs:angularVelocity"),
                    ("Time.outputs:simulationTime", "PubOdom.inputs:timeStamp"),
                    # 2-D lidar (one row) can be connected directly.
                    ("Tick.outputs:tick", "ReadLidar.inputs:execIn"),
                    ("ReadLidar.outputs:execOut", "PubScan.inputs:execIn"),
                    ("ReadLidar.outputs:azimuthRange", "PubScan.inputs:azimuthRange"),
                    ("ReadLidar.outputs:depthRange", "PubScan.inputs:depthRange"),
                    ("ReadLidar.outputs:horizontalFov", "PubScan.inputs:horizontalFov"),
                    ("ReadLidar.outputs:horizontalResolution",
                     "PubScan.inputs:horizontalResolution"),
                    ("ReadLidar.outputs:numCols", "PubScan.inputs:numCols"),
                    ("ReadLidar.outputs:numRows", "PubScan.inputs:numRows"),
                    ("ReadLidar.outputs:linearDepthData",
                     "PubScan.inputs:linearDepthData"),
                    ("ReadLidar.outputs:intensitiesData",
                     "PubScan.inputs:intensitiesData"),
                    ("ReadLidar.outputs:rotationRate", "PubScan.inputs:rotationRate"),
                    ("Time.outputs:simulationTime", "PubScan.inputs:timeStamp"),
                    ("Odom.outputs:execOut", "TfOdom.inputs:execIn"),
                    ("ReadPose.outputs:translation", "TfOdom.inputs:translation"),
                    ("ReadPose.outputs:orientation", "TfOdom.inputs:rotation"),
                    ("Time.outputs:simulationTime", "TfOdom.inputs:timeStamp"),
                    ("Odom.outputs:execOut", "TfLaser.inputs:execIn"),
                    ("Time.outputs:simulationTime", "TfLaser.inputs:timeStamp"),
                    ("Tick.outputs:tick", "SubTwist.inputs:execIn"),
                ],
            )
        except Exception as exc:
            self.errors.append(f"{name} graph: {exc}")
            return False

        h = TwistHandle(
            linear=og.Controller.attribute(graph + "/SubTwist.outputs:linearVelocity"),
            angular=og.Controller.attribute(graph + "/SubTwist.outputs:angularVelocity"),
        )
        self.twists[name] = h
        return True

    def add_twist_input(self, key, topic):
        """Subscribe to a simple geometry_msgs/Twist command channel.

        The load-state and UR10 command channels use a Twist as a compact
        three-float transport: linear.x is the value, and the remaining
        components are reserved for pulses.  This keeps the bridge on the
        well-supported typed OmniGraph nodes.
        """
        if not self.enabled:
            return False
        import omni.graph.core as og

        graph = f"/World/ROS2/input_{key}"
        try:
            self._graph(
                graph,
                [
                    ("Tick", "omni.graph.action.OnPlaybackTick"),
                    ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ],
                [("SubTwist.inputs:topicName", topic)],
                [("Tick.outputs:tick", "SubTwist.inputs:execIn")],
            )
        except Exception as exc:
            self.errors.append(f"{key} input graph: {exc}")
            return False
        self.twists[key] = TwistHandle(
            linear=og.Controller.attribute(
                graph + "/SubTwist.outputs:linearVelocity"
            ),
            angular=og.Controller.attribute(
                graph + "/SubTwist.outputs:angularVelocity"
            ),
        )
        return True

    def add_string_publisher(self, key, topic):
        """Publish JSON/string telemetry through the generic ROS 2 publisher.

        OmniGraph creates the dynamic ``inputs:data`` port only after the
        message type is selected, so the graph is built first, evaluated for a
        frame, then a ConstantToken is connected to that port.  The token value
        is updated from the simulation loop by :meth:`publish_string`.
        """
        if not self.enabled:
            return False
        import omni.graph.core as og

        graph = f"/World/ROS2/pub_{key}"
        try:
            self._graph(
                graph,
                [
                    ("Tick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("Pub", "isaacsim.ros2.bridge.ROS2Publisher"),
                    ("Str", "omni.graph.nodes.ConstantToken"),
                ],
                [
                    ("Pub.inputs:topicName", topic),
                    ("Pub.inputs:messagePackage", "std_msgs"),
                    ("Pub.inputs:messageName", "String"),
                    ("Pub.inputs:messageSubfolder", "msg"),
                    ("Str.inputs:value", "{}"),
                ],
                [
                    ("Context.outputs:context", "Pub.inputs:context"),
                    ("Tick.outputs:tick", "Pub.inputs:execIn"),
                ],
            )
            for _ in range(2):
                self.app.update()
            source = og.Controller.attribute(graph + "/Str.inputs:value")
            data = og.Controller.attribute(graph + "/Pub.inputs:data")
            og.Controller.connect(source, data)
        except Exception as exc:
            self.errors.append(f"{key} string publisher: {exc}")
            return False
        self.string_pubs[key] = source
        return True

    def publish_string(self, key, text):
        source = self.string_pubs.get(key)
        if source is None:
            return False
        try:
            import omni.graph.core as og

            og.Controller.set(source, str(text))
            return True
        except Exception:
            return False

    def load_state(self, name):
        handle = self.twists.get("load_" + name)
        if handle is None:
            return None, 0.0
        values = handle.read()
        return handle.received, float(values[0])

    def input_twist(self, key):
        handle = self.twists.get(key)
        if handle is None:
            return None, np.zeros(6)
        values = handle.read()
        return handle.received, values

    def add_humanoid(self, name, body_path):
        if not self.enabled:
            return False
        import omni.graph.core as og
        import usdrt

        graph = f"/World/ROS2/{name}"
        try:
            self._graph(
                graph,
                [
                    ("Tick", "omni.graph.action.OnPlaybackTick"),
                    ("Time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ReadPose", "isaacsim.core.nodes.IsaacReadWorldPose"),
                    ("Odom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("PubOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    ("TfOdom", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ],
                [
                    ("Odom.inputs:chassisPrim", [usdrt.Sdf.Path(body_path)]),
                    ("ReadPose.inputs:prim", [usdrt.Sdf.Path(body_path)]),
                    ("PubOdom.inputs:topicName", f"{name}/odom"),
                    ("PubOdom.inputs:odomFrameId", "odom"),
                    ("PubOdom.inputs:chassisFrameId", f"{name}/base_link"),
                    ("PubOdom.inputs:publishRawVelocities", False),
                    ("TfOdom.inputs:topicName", "tf"),
                    ("TfOdom.inputs:parentFrameId", "odom"),
                    ("TfOdom.inputs:childFrameId", f"{name}/base_link"),
                    ("SubTwist.inputs:topicName", f"{name}/cmd_vel"),
                ],
                [
                    ("Tick.outputs:tick", "Odom.inputs:execIn"),
                    ("Odom.outputs:execOut", "PubOdom.inputs:execIn"),
                    ("ReadPose.outputs:translation", "PubOdom.inputs:position"),
                    ("ReadPose.outputs:orientation", "PubOdom.inputs:orientation"),
                    ("Odom.outputs:linearVelocity", "PubOdom.inputs:linearVelocity"),
                    ("Odom.outputs:angularVelocity", "PubOdom.inputs:angularVelocity"),
                    ("Time.outputs:simulationTime", "PubOdom.inputs:timeStamp"),
                    ("Odom.outputs:execOut", "TfOdom.inputs:execIn"),
                    ("ReadPose.outputs:translation", "TfOdom.inputs:translation"),
                    ("ReadPose.outputs:orientation", "TfOdom.inputs:rotation"),
                    ("Time.outputs:simulationTime", "TfOdom.inputs:timeStamp"),
                    ("Tick.outputs:tick", "SubTwist.inputs:execIn"),
                ],
            )
        except Exception as exc:
            self.errors.append(f"{name} graph: {exc}")
            return False

        self.twists[name] = TwistHandle(
            linear=og.Controller.attribute(graph + "/SubTwist.outputs:linearVelocity"),
            angular=og.Controller.attribute(graph + "/SubTwist.outputs:angularVelocity"),
        )
        return True

    def add_forklift(self, name, prim_path):
        """Kinematic dynamic obstacle: odom + tf for the fleet planner."""
        if not self.enabled:
            return False
        import omni.graph.core as og
        import usdrt

        graph = f"/World/ROS2/{name}"
        try:
            self._graph(
                graph,
                [
                    ("Tick", "omni.graph.action.OnPlaybackTick"),
                    ("Time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ReadPose", "isaacsim.core.nodes.IsaacReadWorldPose"),
                    ("Odom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("PubOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    ("TfOdom", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ],
                [
                    ("Odom.inputs:chassisPrim", [usdrt.Sdf.Path(prim_path)]),
                    ("ReadPose.inputs:prim", [usdrt.Sdf.Path(prim_path)]),
                    ("PubOdom.inputs:topicName", f"{name}/odom"),
                    ("PubOdom.inputs:odomFrameId", "odom"),
                    ("PubOdom.inputs:chassisFrameId", f"{name}/base_link"),
                    ("PubOdom.inputs:publishRawVelocities", False),
                    ("TfOdom.inputs:topicName", "tf"),
                    ("TfOdom.inputs:parentFrameId", "odom"),
                    ("TfOdom.inputs:childFrameId", f"{name}/base_link"),
                ],
                [
                    ("Tick.outputs:tick", "Odom.inputs:execIn"),
                    ("Odom.outputs:execOut", "PubOdom.inputs:execIn"),
                    ("ReadPose.outputs:translation", "PubOdom.inputs:position"),
                    ("ReadPose.outputs:orientation", "PubOdom.inputs:orientation"),
                    ("Odom.outputs:linearVelocity", "PubOdom.inputs:linearVelocity"),
                    ("Odom.outputs:angularVelocity", "PubOdom.inputs:angularVelocity"),
                    ("Time.outputs:simulationTime", "PubOdom.inputs:timeStamp"),
                    ("Odom.outputs:execOut", "TfOdom.inputs:execIn"),
                    ("ReadPose.outputs:translation", "TfOdom.inputs:translation"),
                    ("ReadPose.outputs:orientation", "TfOdom.inputs:rotation"),
                    ("Time.outputs:simulationTime", "TfOdom.inputs:timeStamp"),
                ],
            )
        except Exception as exc:
            self.errors.append(f"{name} graph: {exc}")
            return False
        return True

    def add_joint_states(self, name, robot_path, topic):
        """Publish joint states for an articulation that is driven locally."""
        if not self.enabled:
            return False
        import omni.graph.core as og
        import usdrt

        graph = f"/World/ROS2/{name}"
        try:
            self._graph(
                graph,
                [
                    ("Tick", "omni.graph.action.OnPlaybackTick"),
                    ("Time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PubJoints", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ],
                [
                    ("PubJoints.inputs:topicName", topic),
                    ("PubJoints.inputs:targetPrim",
                     [usdrt.Sdf.Path(robot_path)]),
                ],
                [
                    ("Tick.outputs:tick", "PubJoints.inputs:execIn"),
                    ("Time.outputs:simulationTime", "PubJoints.inputs:timeStamp"),
                ],
            )
        except Exception as exc:
            self.errors.append(f"{name} joint graph: {exc}")
            return False
        return True

    def add_franka(self, robot_path, joint_states_topic="franka/joint_states",
                   command_topic="franka/joint_command"):
        if not self.enabled:
            return False
        import omni.graph.core as og
        import usdrt

        graph = "/World/ROS2/franka"
        try:
            self._graph(
                graph,
                [
                    ("Tick", "omni.graph.action.OnPlaybackTick"),
                    ("Time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PubJoints", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubJoints", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ],
                [
                    ("PubJoints.inputs:topicName", joint_states_topic),
                    ("PubJoints.inputs:targetPrim",
                     [usdrt.Sdf.Path(robot_path)]),
                    ("SubJoints.inputs:topicName", command_topic),
                ],
                [
                    ("Tick.outputs:tick", "PubJoints.inputs:execIn"),
                    ("Time.outputs:simulationTime", "PubJoints.inputs:timeStamp"),
                    ("Tick.outputs:tick", "SubJoints.inputs:execIn"),
                ],
            )
        except Exception as exc:
            self.errors.append(f"franka graph: {exc}")
            return False

        self.joint_commands["franka"] = JointCommandHandle(
            names=og.Controller.attribute(graph + "/SubJoints.outputs:jointNames"),
            positions=og.Controller.attribute(
                graph + "/SubJoints.outputs:positionCommand"
            ),
        )
        return True

    # -- polling ----------------------------------------------------------
    def twist(self, name):
        handle = self.twists.get(name)
        return handle.read() if handle else np.zeros(6)

    def franka_command(self):
        handle = self.joint_commands.get("franka")
        return handle.read() if handle else None

    def summary(self):
        return {
            "enabled": self.enabled,
            "graphs": len(self.graphs),
            "errors": list(self.errors),
        }
