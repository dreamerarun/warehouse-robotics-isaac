"""Occupancy grid + A* planner used by the fleet manager.

The map is generated from the same rectangles that the simulator builds
(config/warehouse.yaml).  No SLAM, no map_server, no Nav2 - the demo is
deterministic and runs comfortably on a laptop.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _point_segment_distance(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class GridMap:
    def __init__(self, cfg: dict):
        bounds = cfg["world"]["bounds"]
        fleet = cfg["fleet"]
        self.res = float(fleet.get("map_resolution", 0.10))
        self.inflate = float(fleet.get("inflation_radius", 0.42))
        self.x_min = float(bounds["x_min"])
        self.y_min = float(bounds["y_min"])
        self.x_max = float(bounds["x_max"])
        self.y_max = float(bounds["y_max"])
        self.nx = int(math.ceil((self.x_max - self.x_min) / self.res)) + 1
        self.ny = int(math.ceil((self.y_max - self.y_min) / self.res)) + 1
        self.blocked = np.zeros((self.nx, self.ny), dtype=bool)

        # Perimeter safety margin.
        xs = self.x_min + np.arange(self.nx) * self.res
        ys = self.y_min + np.arange(self.ny) * self.res
        edge_x = (xs < self.x_min + self.inflate) | (xs > self.x_max - self.inflate)
        edge_y = (ys < self.y_min + self.inflate) | (ys > self.y_max - self.inflate)
        self.blocked[edge_x, :] = True
        self.blocked[:, edge_y] = True

        for rack in cfg["scene"].get("racks", []):
            self.mark_rect(rack["x"], rack["y"], rack.get("yaw", 0.0),
                           rack["size"][0], rack["size"][1])
        for box in cfg["scene"].get("static_obstacles", []):
            self.mark_rect(box["x"], box["y"], box.get("yaw", 0.0),
                           box["size"][0], box["size"][1])
        for pallet in cfg["scene"].get("pallets", []):
            self.mark_rect(pallet["x"], pallet["y"], pallet.get("yaw", 0.0),
                           pallet["size"][0], pallet["size"][1])
        # Forklift aisle: a planner-only reservation (no physical wall).  The
        # forklift has right of way; AMRs use the open north/south corridors.
        forklift = cfg["scene"].get("forklift", {})
        if forklift.get("enabled", True) and forklift.get("block_amrs", True):
            x = float(forklift.get("lane_x", -3.8))
            y0 = float(forklift.get("y_min", -4.8))
            y1 = float(forklift.get("y_max", -0.9))
            width = float(forklift.get("planner_width", 2.4))
            self.mark_rect(x, (y0 + y1) * 0.5, 0.0, width, y1 - y0)

    # -- coordinates -----------------------------------------------------
    def world_to_cell(self, x, y):
        return (
            int(round((x - self.x_min) / self.res)),
            int(round((y - self.y_min) / self.res)),
        )

    def cell_to_world(self, cx, cy):
        return (self.x_min + cx * self.res, self.y_min + cy * self.res)

    def in_bounds(self, cx, cy):
        return 0 <= cx < self.nx and 0 <= cy < self.ny

    def is_free(self, cx, cy):
        return self.in_bounds(cx, cy) and not self.blocked[cx, cy]

    def is_free_xy(self, x, y):
        return self.is_free(*self.world_to_cell(x, y))

    # -- map authoring ---------------------------------------------------
    def mark_rect(self, cx, cy, yaw, sx, sy):
        c, s = math.cos(yaw), math.sin(yaw)
        hx = sx / 2.0 + self.inflate
        hy = sy / 2.0 + self.inflate
        r = math.hypot(hx, hy)
        i0 = max(0, int((cx - r - self.x_min) / self.res))
        i1 = min(self.nx - 1, int((cx + r - self.x_min) / self.res) + 1)
        j0 = max(0, int((cy - r - self.y_min) / self.res))
        j1 = min(self.ny - 1, int((cy + r - self.y_min) / self.res) + 1)
        for i in range(i0, i1 + 1):
            wx = self.x_min + i * self.res
            for j in range(j0, j1 + 1):
                wy = self.y_min + j * self.res
                dx, dy = wx - cx, wy - cy
                lx = dx * c + dy * s
                ly = -dx * s + dy * c
                if abs(lx) <= hx and abs(ly) <= hy:
                    self.blocked[i, j] = True

    # -- queries ---------------------------------------------------------
    def nearest_free(self, cx, cy, max_radius=60):
        """Closest free cell by true Euclidean distance.

        The planner calls this when a robot has drifted inside the inflation
        margin.  Returning the first free cell found in scan order can pick a
        cell on the far side of a rack and make the robot drive *through* the
        corner to reach it; the nearest cell is always the safe escape.
        """
        if self.is_free(cx, cy):
            return cx, cy
        best = None
        best_d = float("inf")
        for radius in range(1, max_radius):
            # A cell at Chebyshev radius `radius` can never be closer than
            # `radius * res`, so stop as soon as that bound exceeds the best.
            if radius * self.res > best_d:
                break
            for di in range(-radius, radius + 1):
                for dj in (-radius, radius):
                    if self.is_free(cx + di, cy + dj):
                        d = math.hypot(di, dj)
                        if d < best_d:
                            best_d = d
                            best = (cx + di, cy + dj)
            for dj in range(-radius + 1, radius):
                for di in (-radius, radius):
                    if self.is_free(cx + di, cy + dj):
                        d = math.hypot(di, dj)
                        if d < best_d:
                            best_d = d
                            best = (cx + di, cy + dj)
        return best

    def line_clear(self, a, b, step=0.05):
        ax, ay = a
        bx, by = b
        dist = math.hypot(bx - ax, by - ay)
        n = max(1, int(dist / step))
        for k in range(n + 1):
            t = k / n
            if not self.is_free_xy(ax + (bx - ax) * t, ay + (by - ay) * t):
                return False
        return True

    # -- planning --------------------------------------------------------
    def astar(self, start_xy, goal_xy, soft_points=(), soft_cost=7.0):
        start = self.nearest_free(*self.world_to_cell(*start_xy))
        goal = self.nearest_free(*self.world_to_cell(*goal_xy))
        if start is None or goal is None:
            return []
        if start == goal:
            return [tuple(start_xy), tuple(goal_xy)]

        # Soft penalty around other robots so simultaneous plans fan out.
        penalty = np.zeros((self.nx, self.ny), dtype=np.float32)
        soft_radius = 1.25
        for px, py in soft_points:
            cx, cy = self.world_to_cell(px, py)
            r = int(soft_radius / self.res)
            for i in range(max(0, cx - r), min(self.nx, cx + r + 1)):
                for j in range(max(0, cy - r), min(self.ny, cy + r + 1)):
                    d = math.hypot((i - cx) * self.res, (j - cy) * self.res)
                    if d < soft_radius:
                        penalty[i, j] = max(
                            penalty[i, j], soft_cost * (1.0 - d / soft_radius)
                        )

        def heuristic(a, b):
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return self.res * (max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy))

        open_heap = [(heuristic(start, goal), 0.0, start)]
        parents = {start: None}
        g_score = {start: 0.0}
        neighbours = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
        ]
        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current == goal:
                break
            if g > g_score.get(current, float("inf")) + 1e-9:
                continue
            for dx, dy, cost in neighbours:
                ni, nj = current[0] + dx, current[1] + dy
                if not self.is_free(ni, nj):
                    continue
                # Do not cut corners through blocked diagonals.
                if dx and dy:
                    if not self.is_free(current[0] + dx, current[1]):
                        continue
                    if not self.is_free(current[0], current[1] + dy):
                        continue
                ng = g + cost * self.res + float(penalty[ni, nj])
                if ng < g_score.get((ni, nj), float("inf")) - 1e-9:
                    g_score[(ni, nj)] = ng
                    parents[(ni, nj)] = current
                    heapq.heappush(
                        open_heap,
                        (ng + heuristic((ni, nj), goal), ng, (ni, nj)),
                    )
        if goal not in parents:
            return []

        cells = []
        node = goal
        while node is not None:
            cells.append(node)
            node = parents[node]
        cells.reverse()
        points = [self.cell_to_world(*c) for c in cells]
        if points:
            points[0] = tuple(start_xy)
            points[-1] = tuple(goal_xy)
        return self.smooth(points, soft_points)

    def smooth(self, points, avoid_points=(), clearance=1.25):
        """Greedy line-of-sight shortcutting; keeps paths on the free grid.

        ``avoid_points`` are dynamic obstacles (other robots).  Shortcutting
        must respect them as well as the static grid, otherwise A* detours
        around a robot are immediately collapsed back into a straight line
        through it.
        """
        if len(points) <= 2:
            return points
        out = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not (
                self.line_clear(points[i], points[j])
                and self._segment_clear(points[i], points[j], avoid_points, clearance)
            ):
                j -= 1
            out.append(points[j])
            i = j
        return out

    @staticmethod
    def _segment_clear(a, b, avoid_points, clearance):
        """True if a segment does not move closer to a dynamic obstacle."""
        if not avoid_points:
            return True
        for px, py in avoid_points:
            d = _point_segment_distance(px, py, a, b)
            if d >= clearance:
                continue
            # A robot that starts inside the clearance bubble must still be
            # able to drive away from it.  Only reject segments that get
            # closer to the obstacle than the segment start already is.
            da = math.hypot(px - a[0], py - a[1])
            if d < da - 1e-6:
                return False
        return True

    @staticmethod
    def resample(points, step=0.12):
        if len(points) < 2:
            return list(points)
        out = [points[0]]
        for ax, ay in points[1:]:
            bx, by = out[-1]
            dist = math.hypot(ax - bx, ay - by)
            if dist < step:
                continue
            n = max(1, int(dist / step))
            for k in range(1, n + 1):
                t = k / n
                out.append((bx + (ax - bx) * t, by + (ay - by) * t))
        if out[-1] != points[-1]:
            out.append(points[-1])
        return out
