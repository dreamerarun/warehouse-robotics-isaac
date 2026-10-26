#!/usr/bin/env python3
"""
Warehouse AMR sim, Isaac Sim 5.1 standalone (Carter + racks + pallet stacks + walking workers).

Nav: occupancy grid (inflated racks/walls/stacks + moving humans) -> A* -> line-of-sight
smoothing -> pure pursuit -> wheel velocities via the VERIFIED WheeledRobot.apply_wheel_actions.
Safety: SLOW near a human, STOP very near a human, replan every REPLAN_S, WAIT if no path.

RUN:   HEADLESS=1 $ISAAC/python.sh scripts/warehouse_nav_sim.py     (logic test)
       HEADLESS=0 $ISAAC/python.sh scripts/warehouse_nav_sim.py     (GUI)

TEST (all required for PASS):
  [1] floor holds Carter (z stays within 0.3 m of settled height)      -> not FELL
  [2] calibration: forward drive moves >0.05 m, spin turns >0.05 rad
  [3] all GOALS reached within GOAL_TOL (XY) before timeouts
  [4] min distance Carter-to-human never below MIN_SAFE
  [5] zero static collisions (Carter footprint entering rack/stack/wall rects)
TODO for later phases:
  [ ] swap capsule workers for animated People USD assets (check /Isaac/People in assets root)
  [ ] LiDAR + costmap replacing the known-obstacle grid
  [ ] second robot + task allocator
  [ ] OmniGraph Quintic/Stanley test as a separate script
"""
import os, sys, math, heapq, random, traceback
import numpy as np
from isaacsim import SimulationApp   # must be imported and instantiated before other Isaac imports

HEADLESS = os.environ.get("HEADLESS", "0") == "1"
simulation_app = SimulationApp({          # VERIFY: copy config from test_carter.py
    "headless": HEADLESS, "width": 1280, "height": 720, "renderer": "iray",
    "limit_sampling_buffer_size": True, "depth_value_enabled": False,
})

import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdLux, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path

# ------------------------------- config -------------------------------
HX, HY = 13.0, 8.0                       # half extents of warehouse floor (m)
RACK_L, RACK_D, RACK_H = 2.6, 0.9, 3.0
ROW_Y = [-5.5, -2.0, 2.0, 5.5]
ROW_X = [-8.5, -5.5, -2.5, 2.5, 5.5, 8.5]
SHELF_Z = [0.2, 0.95, 1.7, 2.4]
FLOOR_STACKS = [(-6.5, -1.0), (6.5, 1.0)]
HUMANS = [  # (waypoints, speed m/s, phase m)
    ([(-8, 0.0), (8, 0.0)], 0.6, 4.0),      # center aisle walker
    ([(-8, 3.75), (8, 3.75)], 0.7, 0.0),    # upper aisle walker
    ([(0, -3.6), (0, 3.6)], 0.6, 0.0),      # crosses the cross-aisle Carter uses
]
GOALS = [(-5.0, -3.75), (7.0, -3.75), (7.0, 3.75), (-7.0, 3.75), (0.0, 0.0)]
GOAL_TOL = 0.35
RES, INFL = 0.1, 0.55                    # grid cell (m), static inflation (Carter half-width + margin)
HUMAN_R, HUMAN_KEEPOUT = 0.22, 1.0
STOP_DIST, SLOW_DIST, MIN_SAFE = 0.9, 1.8, 0.5
ROBOT_R = 0.33
V_MAX, W_MAX, K_W, LOOKAHEAD = 0.8, 1.2, 1.6, 0.7
WHEEL_R, WHEEL_BASE, MAXW = 0.14, 0.4132, 12.0    # VERIFY vs official test
REPLAN_S, GOAL_TIMEOUT, TOTAL_TIMEOUT = 1.0, 150.0, 400.0
RENDER_EVERY = 2
ROBOT_PRIM = "/World/Carter"
CARTER_USD = "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"

NX, NY = int(2 * HX / RES), int(2 * HY / RES)
XS = -HX + (np.arange(NX) + 0.5) * RES
YS = -HY + (np.arange(NY) + 0.5) * RES

CARD = [(0.72, 0.55, 0.36), (0.65, 0.48, 0.30), (0.80, 0.65, 0.45), (0.58, 0.44, 0.28)]
BLUE, ORANGE, WOOD = (0.10, 0.25, 0.65), (0.90, 0.45, 0.10), (0.55, 0.40, 0.22)

# ------------------------------- globals -------------------------------
my_world = None
carter = None
humans = []
sim_t, step_i, DT = 0.0, 0, 1.0 / 60.0
FWD_SIGN, TURN_SIGN = 1.0, 1.0
STATIC_OCC = None
RAW_RECTS = []


def log(*a):
    print(*a, flush=True)


def finish(code, lines):
    for l in lines:
        log(l)
    simulation_app.close()
    sys.exit(code)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------- scene building -------------------------------
def add_box(stage, path, center, size, color, collide=False, visible=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if not visible:
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    return cube


def build_rack(stage, name, cx, cy, rng):
    base = f"/World/Warehouse/Racks/{name}"
    UsdGeom.Xform.Define(stage, base)
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        add_box(stage, f"{base}/up{i}", (cx + sx * (RACK_L / 2 - 0.04), cy + sy * (RACK_D / 2 - 0.04), RACK_H / 2),
                (0.08, 0.08, RACK_H), BLUE)
    for k, z in enumerate(SHELF_Z):
        add_box(stage, f"{base}/shelf{k}", (cx, cy, z), (RACK_L, RACK_D, 0.05), ORANGE)
        for b in range(3):
            if rng.random() < 0.75:
                bw, bh = rng.uniform(0.6, 0.75), rng.uniform(0.35, 0.55)
                add_box(stage, f"{base}/box{k}_{b}", (cx - RACK_L / 3 + b * RACK_L / 3, cy, z + 0.025 + bh / 2),
                        (bw, RACK_D - 0.2, bh), rng.choice(CARD))
    add_box(stage, f"{base}/collider", (cx, cy, RACK_H / 2), (RACK_L, RACK_D, RACK_H), (0, 0, 0),
            collide=True, visible=False)
    return (cx - RACK_L / 2, cx + RACK_L / 2, cy - RACK_D / 2, cy + RACK_D / 2)


def build_floor_stack(stage, name, cx, cy, rng):
    base = f"/World/Warehouse/Stacks/{name}"
    UsdGeom.Xform.Define(stage, base)
    add_box(stage, f"{base}/pallet", (cx, cy, 0.075), (1.2, 1.0, 0.15), WOOD)
    for lvl in range(3):
        for i in range(2):
            for j in range(2):
                add_box(stage, f"{base}/c{lvl}_{i}_{j}", (cx - 0.3 + i * 0.6, cy - 0.25 + j * 0.5, 0.35 + lvl * 0.4),
                        (0.55, 0.45, 0.4), rng.choice(CARD))
    add_box(stage, f"{base}/collider", (cx, cy, 0.15 + 0.6), (1.2, 1.0, 1.35), (0, 0, 0), collide=True, visible=False)
    return (cx - 0.6, cx + 0.6, cy - 0.5, cy + 0.5)


def polyline_point(pts, d):
    for a, b in zip(pts[:-1], pts[1:]):
        l = float(np.linalg.norm(b - a))
        if d <= l:
            return a + (b - a) * (d / l)
        d -= l
    return pts[-1]


class Human:
    def __init__(self, stage, idx, pts, speed, phase):
        base = f"/World/Humans/Worker_{idx}"
        root = UsdGeom.Xform.Define(stage, base)
        self.op = UsdGeom.Xformable(root.GetPrim()).AddTranslateOp()
        body = UsdGeom.Capsule.Define(stage, base + "/body")
        body.CreateRadiusAttr(HUMAN_R); body.CreateHeightAttr(1.0); body.CreateAxisAttr("Z")
        UsdGeom.Xformable(body.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.72))
        body.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.45, 0.0)])          # hi-vis vest
        head = UsdGeom.Sphere.Define(stage, base + "/head")
        head.CreateRadiusAttr(0.13)
        UsdGeom.Xformable(head.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.54))
        head.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.80, 0.10)])         # hard hat
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
        self.pts = [np.array(p, float) for p in pts]
        self.len = sum(float(np.linalg.norm(b - a)) for a, b in zip(self.pts[:-1], self.pts[1:]))
        self.speed, self.phase = speed, phase
        self.cur = self.pts[0].copy()

    def update(self, t):
        s = (self.speed * t + self.phase) % (2 * self.len)
        d = s if s <= self.len else 2 * self.len - s
        self.cur = polyline_point(self.pts, d)
        self.op.Set(Gf.Vec3d(float(self.cur[0]), float(self.cur[1]), 0.0))


def build_warehouse(stage):
    rng = random.Random(7)
    UsdGeom.Xform.Define(stage, "/World/Warehouse")
    add_box(stage, "/World/Warehouse/floor", (0, 0, -0.05), (2 * HX + 1.0, 2 * HY + 1.0, 0.1), (0.55, 0.55, 0.56), collide=True)
    rects = []
    for name, c, s in [("wN", (0, HY + 0.15, 1.5), (2 * HX + 0.6, 0.3, 3.0)), ("wS", (0, -HY - 0.15, 1.5), (2 * HX + 0.6, 0.3, 3.0)),
                       ("wE", (HX + 0.15, 0, 1.5), (0.3, 2 * HY, 3.0)), ("wW", (-HX - 0.15, 0, 1.5), (0.3, 2 * HY, 3.0))]:
        add_box(stage, f"/World/Warehouse/{name}", c, s, (0.80, 0.80, 0.82), collide=True)
        rects.append((c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2))
    n = 0
    for ry in ROW_Y:
        for rx in ROW_X:
            rects.append(build_rack(stage, f"rack_{n}", rx, ry, rng)); n += 1
    for i, (sx, sy) in enumerate(FLOOR_STACKS):
        rects.append(build_floor_stack(stage, f"stack_{i}", sx, sy, rng))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun"); sun.CreateIntensityAttr(3000.0)
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome"); dome.CreateIntensityAttr(700.0)
    UsdGeom.Xform.Define(stage, "/World/Humans")
    hs = [Human(stage, i, p, s, ph) for i, (p, s, ph) in enumerate(HUMANS)]
    log(f"WAREHOUSE_BUILT: racks={n} stacks={len(FLOOR_STACKS)} humans={len(hs)}")
    return rects, hs


# ------------------------------- planning -------------------------------
def rasterize(rects, infl):
    occ = np.zeros((NX, NY), bool)
    for (x0, x1, y0, y1) in rects:
        mx = (XS >= x0 - infl) & (XS <= x1 + infl)
        my = (YS >= y0 - infl) & (YS <= y1 + infl)
        occ |= mx[:, None] & my[None, :]
    return occ


def to_cell(p):
    return (int(np.clip((p[0] + HX) / RES, 0, NX - 1)), int(np.clip((p[1] + HY) / RES, 0, NY - 1)))


def to_world(c):
    return np.array([XS[c[0]], YS[c[1]]])


def astar(occ, s, g):
    B = occ.tolist()
    if B[g[0]][g[1]]:
        return None
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    gs, came, closed, heap = {s: 0.0}, {}, set(), [(0.0, s)]
    while heap:
        _, c = heapq.heappop(heap)
        if c == g:
            out = [c]
            while c in came:
                c = came[c]; out.append(c)
            return out[::-1]
        if c in closed:
            continue
        closed.add(c)
        for dx, dy, cost in moves:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < NX and 0 <= n[1] < NY) or B[n[0]][n[1]]:
                continue
            if dx and dy and (B[c[0] + dx][c[1]] or B[c[1] + dy][c[1]]):
                continue
            ng = gs[c] + cost
            if ng < gs.get(n, 1e18):
                gs[n] = ng; came[n] = c
                ddx, ddy = abs(g[0] - n[0]), abs(g[1] - n[1])
                h = 1.2 * (max(ddx, ddy) + 0.414 * min(ddx, ddy))
                heapq.heappush(heap, (ng + h, n))
    return None


def los(occ, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 2
    t = np.linspace(0.0, 1.0, n)
    xi = np.round(a[0] + (b[0] - a[0]) * t).astype(int)
    yi = np.round(a[1] + (b[1] - a[1]) * t).astype(int)
    return not occ[xi, yi].any()


def smooth(occ, cells):
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not los(occ, cells[i], cells[j]):
            j -= 1
        out.append(cells[j]); i = j
    return out


def densify(pts, step=0.1):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def plan(xy, goal, human_xy):
    occ = STATIC_OCC.copy()
    for h in human_xy:
        occ |= ((XS[:, None] - h[0]) ** 2 + (YS[None, :] - h[1]) ** 2) <= HUMAN_KEEPOUT ** 2
    s, g = to_cell(xy), to_cell(goal)
    r = int(0.3 / RES)
    occ[max(0, s[0] - r):s[0] + r + 1, max(0, s[1] - r):s[1] + r + 1] = False
    cells = astar(occ, s, g)
    if cells is None:
        return None
    pts = [to_world(c) for c in smooth(occ, cells)]
    pts[0] = np.array(xy, float)
    pts[-1] = np.array(goal, float)
    return densify(pts)


# ------------------------------- control / sim helpers -------------------------------
def get_pose():
    p, q = carter.get_world_pose()
    w, x, y, z = [float(v) for v in q]
    return float(p[0]), float(p[1]), float(p[2]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def sim_step(cmd=None):
    global sim_t, step_i
    if cmd is not None:
        carter.apply_wheel_actions(ArticulationAction(joint_velocities=np.array(cmd, dtype=float)))
    for h in humans:
        h.update(sim_t)
    my_world.step(render=(step_i % RENDER_EVERY == 0))
    sim_t += DT
    step_i += 1


def wheel_cmd(v, w):
    D = TURN_SIGN * w * WHEEL_BASE / WHEEL_R
    m = FWD_SIGN * v / WHEEL_R
    return np.clip(np.array([m - D / 2, m + D / 2]), -MAXW, MAXW)


def pursue(pose, path):
    x, y, _, yaw = pose
    k = int(np.argmin(np.hypot(path[:, 0] - x, path[:, 1] - y)))
    tx, ty = path[min(k + int(LOOKAHEAD / 0.1), len(path) - 1)]
    err = wrap(math.atan2(ty - y, tx - x) - yaw)
    v = 0.0 if abs(err) > 0.7 else V_MAX * max(0.15, math.cos(err))
    dg = math.hypot(path[-1, 0] - x, path[-1, 1] - y)
    v = min(v, max(0.15, 0.6 * dg))
    return v, float(np.clip(K_W * err, -W_MAX, W_MAX))


def in_static(x, y):
    return any(x0 - ROBOT_R <= x <= x1 + ROBOT_R and y0 - ROBOT_R <= y <= y1 + ROBOT_R for (x0, x1, y0, y1) in RAW_RECTS)


# ------------------------------- main -------------------------------
def main():
    global my_world, carter, humans, DT, STATIC_OCC, RAW_RECTS, FWD_SIGN, TURN_SIGN

    # ---------------- INIT BLOCK (replace with verified sequence from test_carter.py) ----------------
    my_world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    RAW_RECTS, humans = build_warehouse(stage)          # static scene added BEFORE physics init (only addition)
    my_world.initialize_simulation_context()            # VERIFIED critical step
    my_world.play()
    assets_root = get_assets_root_path()
    carter = WheeledRobot(prim_path=ROBOT_PRIM, name="carter",
                          wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
                          create_robot=True, usd_path=assets_root + CARTER_USD,
                          position=np.array([0.0, 0.0, 0.0]))
    my_world.scene.add(carter)
    carter.initialize()
    # ---------------- END INIT BLOCK ----------------
    DT = my_world.get_physics_dt()

    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([0.0, -16.0, 18.0]), target=np.array([0.0, 0.0, 0.0]))
    except Exception as e:
        log(f"WARN camera not set: {e}")

    STATIC_OCC = rasterize(RAW_RECTS, INFL)
    for _ in range(90):
        sim_step(np.zeros(2))
    x0, y0, z0, yaw0 = get_pose()
    START_Z = z0
    log(f"START_POSITION: {[round(x0, 3), round(y0, 3), round(z0, 3)]}  GOALS: {GOALS}")

    # ---- calibration: forward drive moves >0.05 m, spin turns >0.05 rad ----
    for _ in range(90):
        sim_step([3.0, 3.0])
    x1, y1, z1, _ = get_pose()
    d = np.array([x1 - x0, y1 - y0])
    if z1 < START_Z - 0.3:
        finish(1, ["NAVIGATION_STATUS: FELL_THROUGH_FLOOR", "TEST_RESULT: FAIL"])
    if np.linalg.norm(d) < 0.05:
        finish(1, [f"CALIBRATION moved only {np.linalg.norm(d):.3f} m", "NAVIGATION_STATUS: NO_MOTION", "TEST_RESULT: FAIL"])
    FWD_SIGN = 1.0 if float(d @ np.array([math.cos(yaw0), math.sin(yaw0)])) >= 0 else -1.0
    for _ in range(30):
        sim_step(np.zeros(2))
    ya = get_pose()[3]
    for _ in range(60):
        sim_step([-2.0, 2.0])
    dyaw = wrap(get_pose()[3] - ya)
    if abs(dyaw) < 0.05:
        finish(1, ["NAVIGATION_STATUS: NO_ROTATION", "TEST_RESULT: FAIL"])
    TURN_SIGN = 1.0 if dyaw > 0 else -1.0
    for _ in range(30):
        sim_step(np.zeros(2))
    log(f"CALIBRATION: FWD_SIGN={FWD_SIGN} TURN_SIGN={TURN_SIGN} (forward moved {np.linalg.norm(d):.2f} m, spin {dyaw:.2f} rad)")

    # ---- mission ----
    goal_i, path, last_plan, goal_t0 = 0, None, -99.0, sim_t
    wait_t, min_h, static_hits, in_hit, status, next_log = 0.0, 1e9, 0, False, "RUNNING", sim_t
    failure = None
    while goal_i < len(GOALS):
        x, y, z, yaw = get_pose()
        if z < START_Z - 0.3:
            failure = "FELL_THROUGH_FLOOR"; break
        if sim_t > TOTAL_TIMEOUT or sim_t - goal_t0 > GOAL_TIMEOUT:
            failure = "TIMEOUT"; break
        goal = GOALS[goal_i]
        hpos = [h.cur for h in humans]
        hd = min(math.hypot(x - p[0], y - p[1]) for p in hpos)
        min_h = min(min_h, hd)
        hit = in_static(x, y)
        static_hits += int(hit and not in_hit)
        in_hit = hit
        if math.hypot(goal[0] - x, goal[1] - y) <= GOAL_TOL:
            log(f"GOAL_REACHED {goal_i + 1}/{len(GOALS)} {goal} at t={sim_t:.1f}s")
            goal_i += 1; path = None; goal_t0 = sim_t
            for _ in range(30):
                sim_step(np.zeros(2))
            continue
        if path is None or sim_t - last_plan >= REPLAN_S:
            newp = plan((x, y), goal, hpos)
            last_plan = sim_t
            path, status = (newp, "NAVIGATING") if newp is not None else (None, "WAITING_NO_PATH")
        if path is None:
            cmd = np.zeros(2); wait_t += DT
        else:
            v, w = pursue((x, y, z, yaw), path)
            if hd < STOP_DIST:
                v, w, status = 0.0, 0.0, "HUMAN_STOP"
            elif hd < SLOW_DIST:
                v, status = v * 0.4, "HUMAN_SLOW"
            cmd = wheel_cmd(v, w)
        sim_step(cmd)
        if sim_t >= next_log:
            next_log = sim_t + 1.0
            log(f"t={sim_t:6.1f}s goal={goal_i + 1} CURRENT_POSITION=({x:.2f},{y:.2f}) "
                f"DISTANCE_TO_GOAL={math.hypot(goal[0] - x, goal[1] - y):.2f} human={hd:.2f} {status}")

    x, y, z, _ = get_pose()
    sim_step(np.zeros(2))
    n_ok = goal_i
    ok = (failure is None and n_ok == len(GOALS) and min_h >= MIN_SAFE and static_hits == 0)
    reason = failure or ("OK" if ok else ("HUMAN_TOO_CLOSE" if min_h < MIN_SAFE else "STATIC_COLLISION"))
    finish(0 if ok else 1, [
        f"CURRENT_POSITION: {[round(x, 3), round(y, 3), round(z, 3)]}",
        f"GOALS_REACHED: {n_ok}/{len(GOALS)}", f"REACHED_GOAL: {n_ok == len(GOALS)}",
        f"MIN_HUMAN_DISTANCE: {min_h:.2f}", f"STATIC_COLLISIONS: {static_hits}", f"WAIT_TIME_S: {wait_t:.1f}",
        f"SIM_TIME_S: {sim_t:.1f}", f"NAVIGATION_STATUS: {reason}", f"TEST_RESULT: {'PASS' if ok else 'FAIL'}"])


try:
    main()
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    finish(2, ["NAVIGATION_STATUS: ERROR", "TEST_RESULT: FAIL"])