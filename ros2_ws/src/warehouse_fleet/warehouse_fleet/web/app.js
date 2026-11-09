(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const canvas = $("mapCanvas");
  const ctx = canvas.getContext("2d");

  const app = {
    state: null,
    map: null,
    mapVersion: -1,
    selected: null,
    follow: false,
    showMapping: true,
    teleopRobot: null,
    teleopOn: false,
    lastMapFetch: 0,
    loadingState: false,
    events: [],
    mode: "AUTO",
  };

  const view = { zoom: 1, panX: 0, panY: 0 };
  const drag = { active: false, moved: false, x: 0, y: 0, panX: 0, panY: 0 };
  const keys = { forward: false, back: false, left: false, right: false };
  let lastDrive = 0;
  let driveBusy = false;

  const STATE_COLORS = {
    IDLE: "#6e7d94",
    PARKING: "#b28bff",
    TO_PICKUP: "#57b4ff",
    TO_DROPOFF: "#57b4ff",
    PICKING: "#3ee08a",
    DROPPING: "#3ee08a",
    TELEOP: "#c19bff",
  };

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));
  }

  function num(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function fmt(value, digits = 1) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "–";
  }

  function timeText(stamp) {
    if (!stamp) return "--:--:--";
    const d = new Date(num(stamp) * 1000);
    return d.toLocaleTimeString([], { hour12: false });
  }

  function toast(message, bad = false) {
    const el = $("toast");
    el.textContent = message;
    el.style.borderColor = bad ? "rgba(255,93,104,.6)" : "rgba(132,158,190,.4)";
    el.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => el.classList.remove("show"), 2600);
  }

  async function postJSON(path, body) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }

  async function loadState() {
    if (app.loadingState) return;
    app.loadingState = true;
    try {
      const response = await fetch("/api/state", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      app.state = await response.json();
      const version = app.state?.ros?.map_version ?? 0;
      if (version !== app.mapVersion && Date.now() - app.lastMapFetch > 1200) {
        app.lastMapFetch = Date.now();
        loadMap();
      }
      render();
    } catch (err) {
      app.state = null;
      renderOffline();
    } finally {
      app.loadingState = false;
    }
  }

  async function loadMap() {
    try {
      const response = await fetch("/api/map", { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      if (!payload.data) return;
      const raw = atob(payload.data);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      payload.cells = new Int8Array(bytes.buffer);
      app.map = payload;
      app.mapVersion = payload.version;
    } catch (err) {
      // Map telemetry can arrive a few frames after the fleet status.
    }
  }

  function worldBounds() {
    return app.state?.scene?.world?.bounds || {
      x_min: -10, x_max: 10, y_min: -7, y_max: 7,
    };
  }

  function canvasSize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(rect.width * dpr));
    const h = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return { width: rect.width, height: rect.height, dpr };
  }

  function transformFor(bounds, width, height) {
    const cx = (bounds.x_min + bounds.x_max) / 2;
    const cy = (bounds.y_min + bounds.y_max) / 2;
    const base = Math.min(
      width / Math.max(1, bounds.x_max - bounds.x_min),
      height / Math.max(1, bounds.y_max - bounds.y_min),
    ) * 0.91;
    const scale = base * view.zoom;
    let focusX = cx;
    let focusY = cy;
    if (app.follow) {
      const robot = (app.state?.fleet?.amrs || []).find((a) => a.name === app.selected);
      if (robot?.pose) {
        focusX = robot.pose[0];
        focusY = robot.pose[1];
      }
    }
    return {
      scale,
      x: (wx) => width / 2 + (wx - focusX) * scale + view.panX,
      y: (wy) => height / 2 - (wy - focusY) * scale + view.panY,
      inv: (sx, sy) => [
        focusX + (sx - width / 2 - view.panX) / scale,
        focusY - (sy - height / 2 - view.panY) / scale,
      ],
    };
  }

  function drawMap() {
    if (!ctx) return;
    const { width, height, dpr } = canvasSize();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const bounds = worldBounds();
    const tr = transformFor(bounds, width, height);
    const scene = app.state?.scene?.scene || {};

    // Background.
    const bg = ctx.createLinearGradient(0, 0, 0, height);
    bg.addColorStop(0, "#0b121c");
    bg.addColorStop(1, "#080d15");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    // 1 m grid.
    ctx.strokeStyle = "rgba(95, 122, 157, 0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = Math.ceil(bounds.x_min); x <= bounds.x_max; x++) {
      ctx.moveTo(tr.x(x), 0);
      ctx.lineTo(tr.x(x), height);
    }
    for (let y = Math.ceil(bounds.y_min); y <= bounds.y_max; y++) {
      ctx.moveTo(0, tr.y(y));
      ctx.lineTo(width, tr.y(y));
    }
    ctx.stroke();

    // Occupancy grid from /fleet/map.
    if (app.showMapping && app.map?.cells && app.map.meta?.width) {
      const meta = app.map.meta;
      const cells = app.map.cells;
      const res = num(meta.resolution, 0.1);
      const ox = num(meta.origin?.[0], bounds.x_min);
      const oy = num(meta.origin?.[1], bounds.y_min);
      const cell = Math.max(1, res * tr.scale + 0.35);
      ctx.fillStyle = "rgba(76, 122, 176, 0.20)";
      for (let iy = 0; iy < meta.height; iy++) {
        for (let ix = 0; ix < meta.width; ix++) {
          if (cells[ix + iy * meta.width] > 50) {
            ctx.fillRect(
              tr.x(ox + ix * res),
              tr.y(oy + (iy + 1) * res),
              cell,
              cell,
            );
          }
        }
      }
    }

    // Scene objects.
    drawRects(scene.racks || [], tr, "rgba(41, 59, 82, 0.88)", "rgba(115, 150, 192, 0.34)");
    drawRects(scene.pallets || [], tr, "rgba(100, 74, 44, 0.78)", "rgba(201, 152, 91, 0.32)");
    drawRects(scene.static_obstacles || [], tr, "rgba(29, 44, 62, 0.92)", "rgba(87, 120, 160, 0.25)");
    for (const conv of scene.conveyors || []) {
      drawRect(conv, tr, "rgba(26, 91, 121, 0.78)", "rgba(92, 205, 235, 0.55)");
      const x0 = tr.x(num(conv.x) - num(conv.size?.[0], 2) / 2);
      const x1 = tr.x(num(conv.x) + num(conv.size?.[0], 2) / 2);
      ctx.strokeStyle = "rgba(112, 226, 255, 0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 5]);
      ctx.beginPath();
      ctx.moveTo(x0, tr.y(num(conv.y)));
      ctx.lineTo(x1, tr.y(num(conv.y)));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // UR10 cell.
    const industrial = scene.industrial || {};
    if (industrial.enabled) {
      if (industrial.base) drawDisc(industrial.base.x, industrial.base.y, tr, 0.32, "#ffb85a", "rgba(255,184,90,.25)");
      if (industrial.place) drawDisc(industrial.place.x, industrial.place.y, tr, 0.42, "#ff8c42", "rgba(255,140,66,.20)");
    }

    // Stations.
    for (const st of scene.stations || []) {
      const color = st.color ? `rgb(${st.color.map((v) => Math.round(v * 255)).join(",")})` : "#57b4ff";
      drawDisc(st.x, st.y, tr, 0.33, color, "rgba(255,255,255,.10)");
      ctx.fillStyle = "rgba(218, 231, 248, 0.72)";
      ctx.font = "600 9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(st.name, tr.x(num(st.x)), tr.y(num(st.y)) + 17);
    }

    // Dynamic obstacles reported by the fleet (humanoid, forklift).
    for (const obs of app.state?.fleet?.dynamic_obstacles || []) {
      if (!obs.pose) continue;
      ctx.fillStyle = obs.name === "forklift" ? "#ffbf5a" : "#ad8cff";
      ctx.beginPath();
      ctx.arc(tr.x(obs.pose[0]), tr.y(obs.pose[1]), 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "rgba(218, 231, 248, 0.7)";
      ctx.font = "8px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(obs.name, tr.x(obs.pose[0]) + 7, tr.y(obs.pose[1]) - 5);
    }

    // Humanoid pose as a separate, labelled marker.
    const h1 = app.state?.humanoid;
    if (h1?.pose) {
      ctx.fillStyle = "rgba(173, 140, 255, 0.9)";
      ctx.beginPath();
      ctx.arc(tr.x(h1.pose[0]), tr.y(h1.pose[1]), 5.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Planned paths.
    for (const amr of app.state?.fleet?.amrs || []) {
      if (!amr.path || amr.path.length < 2) continue;
      ctx.strokeStyle = amr.name === app.selected
        ? "rgba(87, 180, 255, 0.78)" : "rgba(87, 180, 255, 0.28)";
      ctx.lineWidth = amr.name === app.selected ? 2 : 1;
      ctx.beginPath();
      amr.path.forEach((point, i) => {
        const x = tr.x(point[0]);
        const y = tr.y(point[1]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // AMRs.
    for (const amr of app.state?.fleet?.amrs || []) {
      if (!amr.pose) continue;
      const sx = tr.x(amr.pose[0]);
      const sy = tr.y(amr.pose[1]);
      const color = STATE_COLORS[amr.state] || "#e8eef8";
      if (app.selected === amr.name) {
        ctx.strokeStyle = "rgba(255,255,255,.9)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx, sy, 12, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(sx, sy, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(8,13,21,.9)";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(sx + Math.cos(-num(amr.yaw)) * 14, sy + Math.sin(-num(amr.yaw)) * 14);
      ctx.stroke();
      ctx.fillStyle = "rgba(232, 238, 248, 0.92)";
      ctx.font = "700 9px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(amr.name, sx + 10, sy - 8);
      ctx.fillStyle = "rgba(141, 154, 176, 0.9)";
      ctx.font = "8px Inter, sans-serif";
      ctx.fillText(amr.state, sx + 10, sy + 2);
    }

    // Selected follow halo.
    if (app.follow && app.selected) {
      ctx.strokeStyle = "rgba(87, 180, 255, 0.4)";
      ctx.lineWidth = 1;
      ctx.strokeRect(6, 6, width - 12, height - 12);
    }

    const mapMeta = app.map?.meta || app.state?.map || {};
    $("mapInfo").textContent = mapMeta.width
      ? `${mapMeta.width}×${mapMeta.height} cells · ${fmt(mapMeta.resolution, 2)} m · ${mapMeta.blocked_cells ?? "–"} blocked`
      : "waiting for /fleet/map …";
    $("mapStats").textContent =
      `x ${fmt(bounds.x_min, 1)}…${fmt(bounds.x_max, 1)} m · y ${fmt(bounds.y_min, 1)}…${fmt(bounds.y_max, 1)} m · zoom ${fmt(view.zoom, 2)}×`;
  }

  function drawRect(item, tr, fill, stroke) {
    const x = num(item.x);
    const y = num(item.y);
    const sx = num(item.size?.[0], 0.2);
    const sy = num(item.size?.[1], 0.2);
    const yaw = num(item.yaw, 0);
    const w = sx * tr.scale;
    const h = sy * tr.scale;
    ctx.save();
    ctx.translate(tr.x(x), tr.y(y));
    ctx.rotate(-yaw);
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.fillRect(-w / 2, -h / 2, w, h);
    ctx.strokeRect(-w / 2, -h / 2, w, h);
    ctx.restore();
  }

  function drawRects(items, tr, fill, stroke) {
    for (const item of items) drawRect(item, tr, fill, stroke);
  }

  function drawDisc(x, y, tr, radius, fill, stroke) {
    ctx.beginPath();
    ctx.arc(tr.x(num(x)), tr.y(num(y)), Math.max(3, radius * tr.scale), 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  }

  function renderOffline() {
    $("linkPill").textContent = "ROS 2 offline";
    $("linkPill").className = "pill offline";
    $("modePill").textContent = "NO DATA";
    $("modePill").className = "pill";
    $("robotCards").innerHTML = '<div class="empty">dashboard lost the ROS 2 state stream. Start the simulation and autonomy stack.</div>';
    $("taskRows").innerHTML = "";
    drawMap();
  }

  function render() {
    if (!app.state) return renderOffline();
    const fleet = app.state.fleet || {};
    app.mode = fleet.mode || "AUTO";
    if (!app.selected && fleet.amrs?.length) app.selected = fleet.amrs[0].name;
    if (!app.teleopRobot && fleet.amrs?.length) app.teleopRobot = fleet.amrs[0].name;

    const link = $("linkPill");
    link.textContent = app.state.online ? "ROS 2 connected" : "ROS 2 stale";
    link.className = `pill ${app.state.online ? "online" : "offline"}`;
    const modePill = $("modePill");
    modePill.textContent = app.mode;
    modePill.className = `pill mode-${app.mode.toLowerCase()}`;
    for (const btn of document.querySelectorAll(".mode-btn")) {
      btn.classList.toggle("active", btn.dataset.mode === app.mode);
    }

    renderRobots(fleet);
    renderTasks(fleet);
    renderProcess(fleet);
    renderTeleop(fleet, fleet.teleop || []);
    renderEvents(app.state.events || []);
    populateStations(app.state.scene?.scene?.stations || []);
    populateTeleopRobots(fleet.amrs || []);
    updateCounts(fleet);
    updateStationLegend(app.state.scene?.scene?.stations || []);
    drawMap();
  }

  function updateCounts(fleet) {
    const active = (fleet.amrs || []).filter((a) => a.task).length;
    $("activeCount").textContent = `${active} active`;
    const pending = (fleet.tasks || []).filter((t) => ["PENDING", "ASSIGNED", "DELIVERING"].includes(t.status)).length;
    $("taskCount").textContent = `${pending} queued`;
  }

  function renderRobots(fleet) {
    const cards = [];
    for (const amr of fleet.amrs || []) {
      const state = esc(amr.state || "IDLE");
      const task = amr.task ? `${esc(amr.task)} · ${esc(amr.cargo || "")}` : "no task";
      const pose = amr.pose ? `${fmt(amr.pose[0], 1)}, ${fmt(amr.pose[1], 1)}` : "–";
      const loaded = amr.loaded ? "loaded" : "empty";
      const teleop = (fleet.teleop || []).includes(amr.name);
      cards.push(`
        <article class="robot-card s-${state}" data-robot="${esc(amr.name)}">
          <div class="name"><span>${esc(amr.name)}</span><span class="chip ${teleop ? "violet" : state === "IDLE" ? "" : "blue"}">${teleop ? "TELEOP" : state}</span></div>
          <div class="role">${esc(amr.role || "warehouse AMR")}</div>
          <div class="metric"><span>Task</span><b>${task}</b></div>
          <div class="metric"><span>Pose</span><b>${pose}</b></div>
          <div class="metric"><span>Payload</span><b>${loaded}</b></div>
          <button class="drive" data-drive="${esc(amr.name)}">${teleop ? "Stop teleop" : "Drive this robot"}</button>
        </article>`);
    }
    const h1 = app.state?.humanoid;
    if (h1 && Object.keys(h1).length) {
      const carrying = h1.carrying ? "carrying tote" : "empty";
      const pose = h1.pose ? `${fmt(h1.pose[0], 1)}, ${fmt(h1.pose[1], 1)}` : "–";
      cards.push(`
        <article class="robot-card s-${esc((h1.state || "idle").toUpperCase())}">
          <div class="name"><span>humanoid</span><span class="chip violet">H1</span></div>
          <div class="role">tote transfer · pickup to packing</div>
          <div class="metric"><span>State</span><b>${esc(h1.state || "–")}</b></div>
          <div class="metric"><span>Load</span><b>${carrying}</b></div>
          <div class="metric"><span>Delivered</span><b>${h1.loads ?? "–"}</b></div>
          <div class="metric"><span>Pose</span><b>${pose}</b></div>
        </article>`);
    }
    const ur10 = app.state?.ur10 || app.state?.process?.ur10;
    if (ur10 && Object.keys(ur10).length) {
      cards.push(`
        <article class="robot-card s-PICKING">
          <div class="name"><span>ur10</span><span class="chip amber">CELL</span></div>
          <div class="role">palletising · staging to output pallet</div>
          <div class="metric"><span>State</span><b>${esc(ur10.paused ? "paused" : ur10.state || "–")}</b></div>
          <div class="metric"><span>Stack</span><b>${ur10.stack ?? "–"}/${ur10.stack_size ?? "–"}</b></div>
          <div class="metric"><span>Shipped pallets</span><b>${ur10.pallets_shipped ?? "–"}</b></div>
          <div class="metric"><span>Staging stock</span><b>${ur10.staging_stock ?? "–"}</b></div>
        </article>`);
    }
    const franka = app.state?.franka;
    if (franka && Object.keys(franka).length) {
      cards.push(`
        <article class="robot-card s-PICKING">
          <div class="name"><span>franka</span><span class="chip green">QA</span></div>
          <div class="role">cube inspection / packing cell</div>
          <div class="metric"><span>State</span><b>${esc(franka.state || "–")}</b></div>
          <div class="metric"><span>Step</span><b>${franka.step ?? "–"}/${franka.steps ?? "–"}</b></div>
          <div class="metric"><span>Action</span><b>${esc(franka.label || "–")}</b></div>
        </article>`);
    }
    const fork = (fleet.dynamic_obstacles || []).find((o) => o.name === "forklift");
    if (fork) {
      cards.push(`
        <article class="robot-card s-PARKING">
          <div class="name"><span>forklift</span><span class="chip amber">SHUTTLE</span></div>
          <div class="role">cross-aisle obstacle · pallet / truck shuttle</div>
          <div class="metric"><span>Pose</span><b>${fmt(fork.pose[0], 1)}, ${fmt(fork.pose[1], 1)}</b></div>
          <div class="metric"><span>Mode</span><b>dynamic aisle</b></div>
        </article>`);
    }
    $("robotCards").innerHTML = cards.join("") || '<div class="empty">no robots reported</div>';
  }

  function renderTasks(fleet) {
    const statuses = { PENDING: "", ASSIGNED: "blue", DELIVERING: "violet", DONE: "green" };
    const tasks = (fleet.tasks || [])
      .filter((t) => t.status !== "DONE")
      .sort((a, b) => {
        const rank = { DELIVERING: 0, ASSIGNED: 1, PENDING: 2 };
        return (rank[a.status] ?? 3) - (rank[b.status] ?? 3) || num(a.priority) - num(b.priority);
      })
      .slice(0, 9);
    $("taskRows").innerHTML = tasks.map((t) => `
      <tr>
        <td>${esc(t.id)}</td>
        <td><span class="route">${esc(t.pickup)} → ${esc(t.dropoff)}</span><br><span class="muted">${esc(t.description || t.cargo || "")}</span></td>
        <td><span class="chip ${statuses[t.status] || ""}">${esc(t.status)}</span></td>
        <td>${esc(t.assigned || "—")}</td>
        <td>${t.assigned ? `<button class="cancel" data-cancel="${esc(t.assigned)}" title="Cancel and requeue">×</button>` : ""}</td>
      </tr>`).join("") || '<tr><td colspan="5" class="empty">queue empty — supplier / shipping conveyor will create work</td></tr>';
  }

  function pick(obj, key, fallback = "–") {
    return obj && obj[key] !== undefined && obj[key] !== null ? obj[key] : fallback;
  }

  function renderProcess(fleet) {
    const inv = fleet.inventory || {};
    const cap = fleet.capacity || {};
    const flow = fleet.flow || {};
    $("inv-inbound").textContent = `${pick(inv, "inbound")}${cap.inbound ? "/" + cap.inbound : ""}`;
    $("inv-staging").textContent = `${pick(inv, "staging")}${cap.staging ? "/" + cap.staging : ""}`;
    $("inv-outbound").textContent = `${pick(inv, "outbound")}${cap.outbound ? "/" + cap.outbound : ""}`;
    $("inv-empty").textContent = pick(inv, "empty");
    $("inv-arm_dock").textContent = pick(inv, "arm_dock");
    $("receivedCount").textContent = pick(flow, "received");
    $("shippedCount").textContent = pick(flow, "shipped");
    $("completedCount").textContent = fleet.completed ?? "–";
    $("flowState").textContent = fleet.managed ? "managed flow" : "fixed cycle";

    const ur10 = app.state?.ur10 || app.state?.process?.ur10 || {};
    const stackSize = num(ur10.stack_size, 3);
    const stack = num(ur10.stack, 0);
    $("palletCount").textContent = `${stack}/${stackSize}`;
    $("palletSlots").innerHTML = Array.from({ length: stackSize }, (_, i) =>
      `<span class="pallet-slot ${i < stack ? "filled" : ""}"></span>`).join("");
    const auto = ur10.auto_feed ? "visual auto-feed" : "AMR-fed";
    $("ur10Line").textContent = ur10.state
      ? `UR10: ${ur10.paused ? "PAUSED" : String(ur10.state).toUpperCase()} · stack ${stack}/${stackSize} · cycles ${pick(ur10, "cycles")} · staging stock ${pick(ur10, "staging_stock")} · ${auto}`
      : "UR10: waiting for /ur10/status";
    const belts = app.state?.process?.conveyors || {};
    $("beltLine").textContent = belts.running !== undefined
      ? `Conveyors: ${belts.running ? "RUNNING" : "STOPPED"} · ${pick(belts, "loads")} animated loads · ${fmt(belts.speed, 2)} m/s`
      : "Conveyors: status published by the simulation";
  }

  function renderTeleop(fleet, teleopList) {
    const active = teleopList.includes(app.teleopRobot);
    app.teleopOn = active && app.mode === "AUTO";
    $("teleopPill").textContent = active ? (app.mode === "AUTO" ? "armed" : "blocked") : "off";
    $("teleopPill").className = `mini-stat ${active ? "teleop-armed" : ""}`;
    $("teleopToggle").textContent = active ? "Stop teleop" : "Enable teleop";
    $("teleopToggle").classList.toggle("primary", !active);
    const select = $("teleopRobot");
    if (select.value !== app.teleopRobot) select.value = app.teleopRobot || "";
    if (!active) {
      keys.forward = keys.back = keys.left = keys.right = false;
    }
  }

  function renderEvents(events) {
    const list = [...events].slice(-30).reverse();
    $("eventCount").textContent = `${events.length} events`;
    $("eventList").innerHTML = list.map((ev) => {
      const name = ev.event || "event";
      const color = /complete|picked|shipped|received/.test(name) ? "green"
        : /assigned|teleop|mode/.test(name) ? "blue"
        : /requeue|refused|error/.test(name) ? "red"
        : /created|inbound|outbound/.test(name) ? "amber" : "";
      const detail = ev.id || ev.robot || ev.station || ev.mode || ev.reason || ev.cargo || "";
      return `<li><span class="t">${timeText(ev.received)}</span><span class="e ${color}" title="${esc(JSON.stringify(ev))}">${esc(name)} ${detail ? "· " + esc(detail) : ""}</span><span class="r">${esc(ev.robot || ev.team || "")}</span></li>`;
    }).join("") || '<li class="empty">no task events yet</li>';
  }

  function populateStations(stations) {
    const pickup = $("taskPickup");
    const dropoff = $("taskDropoff");
    if (!stations.length || pickup.dataset.ready === "1") return;
    const options = stations.map((s) => `<option value="${esc(s.name)}">${esc(s.name)}</option>`).join("");
    pickup.innerHTML = options;
    dropoff.innerHTML = options;
    if (stations.some((s) => s.name === "inbound")) pickup.value = "inbound";
    if (stations.some((s) => s.name === "staging")) dropoff.value = "staging";
    pickup.dataset.ready = "1";
    dropoff.dataset.ready = "1";
  }

  function populateTeleopRobots(amrs) {
    const select = $("teleopRobot");
    if (!amrs.length || select.dataset.ready === "1") return;
    select.innerHTML = amrs.map((a) => `<option value="${esc(a.name)}">${esc(a.name)} · ${esc(a.role || "")}</option>`).join("");
    select.value = app.teleopRobot || amrs[0].name;
    select.dataset.ready = "1";
    const original = select.onchange;
    select.onchange = () => {
      app.teleopRobot = select.value;
      if (app.teleopOn) {
        // Move teleoperation to the newly selected robot.
        setTeleop(app.selected || select.value, false).finally(() => {
          app.selected = select.value;
          setTeleop(select.value, true);
        });
      } else {
        app.selected = select.value;
      }
      if (original) original();
    };
  }

  function updateStationLegend(stations) {
    if (!stations.length) return;
    $("stationLegend").innerHTML = stations.map((s) => {
      const color = s.color ? `rgb(${s.color.map((v) => Math.round(v * 255)).join(",")})` : "#57b4ff";
      return `<span class="legend-item"><span class="legend-dot" style="background:${color}"></span>${esc(s.name)}</span>`;
    }).join("");
  }

  async function setTeleop(robot, enabled) {
    try {
      if (enabled && app.teleopOn && app.teleopRobot && app.teleopRobot !== robot) {
        await postJSON("/api/control", { action: "set_teleop", robot: app.teleopRobot, enabled: false });
      }
      await postJSON("/api/control", { action: "set_teleop", robot, enabled });
      app.teleopRobot = robot;
      app.selected = robot;
      app.teleopOn = enabled;
      if (!enabled) {
        keys.forward = keys.back = keys.left = keys.right = false;
        await postJSON("/api/control", { action: "teleop", robot, linear: 0, angular: 0 });
      }
      toast(enabled ? `Teleop armed for ${robot}` : `Teleop released for ${robot}`);
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function driveTick() {
    if (!app.teleopOn || !app.teleopRobot) return;
    if (driveBusy) return;
    const linearSlider = num($("teleopLinear").value, 0.3);
    const angularSlider = num($("teleopAngular").value, 0.6);
    let linear = 0;
    let angular = 0;
    if (keys.forward) linear += linearSlider;
    if (keys.back) linear -= linearSlider * 0.65;
    if (keys.left) angular += angularSlider;
    if (keys.right) angular -= angularSlider;
    if (!linear && !angular && performance.now() - lastDrive < 180) return;
    lastDrive = performance.now();
    driveBusy = true;
    try {
      await postJSON("/api/control", {
        action: "teleop",
        robot: app.teleopRobot,
        linear,
        angular,
      });
    } catch (err) {
      // Do not spam toasts for a transient missed drive packet.
    } finally {
      driveBusy = false;
    }
  }

  async function cancelTask(robot) {
    try {
      await postJSON("/api/control", { action: "cancel_task", robot });
      toast(`${robot} task cancelled and requeued`);
    } catch (err) {
      toast(err.message, true);
    }
  }

  function nearestRobot(sx, sy) {
    const amrs = app.state?.fleet?.amrs || [];
    const { width, height } = canvasSize();
    const tr = transformFor(worldBounds(), width, height);
    let best = null;
    let bestDist = 26;
    for (const amr of amrs) {
      if (!amr.pose) continue;
      const dx = tr.x(amr.pose[0]) - sx;
      const dy = tr.y(amr.pose[1]) - sy;
      const d = Math.hypot(dx, dy);
      if (d < bestDist) { best = amr.name; bestDist = d; }
    }
    return best;
  }

  // ------------------------------------------------------------------
  // Events
  // ------------------------------------------------------------------
  $("taskForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const result = await postJSON("/api/task", {
        pickup: $("taskPickup").value,
        dropoff: $("taskDropoff").value,
        priority: num($("taskPriority").value, 20),
      });
      toast(`Work order ${result.task.pickup} → ${result.task.dropoff} dispatched`);
    } catch (err) {
      toast(err.message, true);
    }
  });

  for (const button of document.querySelectorAll(".mode-btn")) {
    button.addEventListener("click", async () => {
      const mode = button.dataset.mode;
      if (mode === "ESTOP" && !confirm("Emergency stop all autonomous motion?")) return;
      try {
        await postJSON("/api/control", { action: "set_mode", mode });
        if (mode !== "AUTO") {
          app.teleopOn = false;
          keys.forward = keys.back = keys.left = keys.right = false;
        }
        toast(`Fleet mode ${mode}`);
      } catch (err) {
        toast(err.message, true);
      }
    });
  }

  $("teleopToggle").addEventListener("click", () => {
    const robot = $("teleopRobot").value || app.teleopRobot;
    const active = (app.state?.fleet?.teleop || []).includes(robot);
    setTeleop(robot, !active);
  });

  for (const button of document.querySelectorAll("[data-process]")) {
    button.addEventListener("click", async () => {
      try {
        await postJSON("/api/control", {
          action: "process",
          command: button.dataset.process,
        });
        toast(`UR10 command: ${button.dataset.process}`);
      } catch (err) {
        toast(err.message, true);
      }
    });
  }

  $("robotCards").addEventListener("click", (event) => {
    const drive = event.target.closest("[data-drive]");
    if (drive) {
      const robot = drive.dataset.drive;
      const active = (app.state?.fleet?.teleop || []).includes(robot);
      app.teleopRobot = robot;
      app.selected = robot;
      $("teleopRobot").value = robot;
      setTeleop(robot, !active);
      return;
    }
    const card = event.target.closest("[data-robot]");
    if (card) {
      app.selected = card.dataset.robot;
      app.teleopRobot = card.dataset.robot;
      const select = $("teleopRobot");
      if (select) select.value = card.dataset.robot;
      drawMap();
    }
  });

  $("taskRows").addEventListener("click", (event) => {
    const button = event.target.closest("[data-cancel]");
    if (button) cancelTask(button.dataset.cancel);
  });

  $("mapReset").addEventListener("click", () => {
    view.zoom = 1;
    view.panX = 0;
    view.panY = 0;
    app.follow = false;
    $("mapFollow").textContent = "Follow: off";
    drawMap();
  });

  $("mapFollow").addEventListener("click", () => {
    if (!app.selected) app.selected = app.state?.fleet?.amrs?.[0]?.name || null;
    app.follow = !app.follow;
    $("mapFollow").textContent = `Follow: ${app.follow ? app.selected || "on" : "off"}`;
    drawMap();
  });

  $("mapMapping").addEventListener("click", () => {
    app.showMapping = !app.showMapping;
    $("mapMapping").textContent = `Mapping overlay: ${app.showMapping ? "on" : "off"}`;
    drawMap();
  });

  for (const slider of [$("teleopLinear"), $("teleopAngular")]) {
    slider.addEventListener("input", () => {
      $("linearOut").textContent = `${fmt($("teleopLinear").value, 2)} m/s`;
      $("angularOut").textContent = `${fmt($("teleopAngular").value, 2)} rad/s`;
    });
  }

  // D-pad pointer handling.
  for (const button of document.querySelectorAll(".dpad-btn")) {
    const move = button.dataset.move;
    const press = (event) => {
      event.preventDefault();
      button.classList.add("pressed");
      if (move === "forward") keys.forward = true;
      if (move === "back") keys.back = true;
      if (move === "left") keys.left = true;
      if (move === "right") keys.right = true;
    };
    const release = (event) => {
      event.preventDefault();
      button.classList.remove("pressed");
      if (move === "forward") keys.forward = false;
      if (move === "back") keys.back = false;
      if (move === "left") keys.left = false;
      if (move === "right") keys.right = false;
    };
    button.addEventListener("pointerdown", press);
    button.addEventListener("pointerup", release);
    button.addEventListener("pointercancel", release);
    button.addEventListener("pointerleave", release);
  }

  // Keyboard teleop. Ignore controls when typing into a form field.
  window.addEventListener("keydown", (event) => {
    if (!app.teleopOn) return;
    const target = event.target;
    if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
    const key = event.key.toLowerCase();
    if (["w", "arrowup"].includes(key)) { keys.forward = true; event.preventDefault(); }
    if (["s", "arrowdown"].includes(key)) { keys.back = true; event.preventDefault(); }
    if (["a", "arrowleft"].includes(key)) { keys.left = true; event.preventDefault(); }
    if (["d", "arrowright"].includes(key)) { keys.right = true; event.preventDefault(); }
  });
  window.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (["w", "arrowup"].includes(key)) keys.forward = false;
    if (["s", "arrowdown"].includes(key)) keys.back = false;
    if (["a", "arrowleft"].includes(key)) keys.left = false;
    if (["d", "arrowright"].includes(key)) keys.right = false;
  });

  // Map interaction.
  canvas.addEventListener("pointerdown", (event) => {
    const rect = canvas.getBoundingClientRect();
    drag.active = true;
    drag.moved = false;
    drag.x = event.clientX - rect.left;
    drag.y = event.clientY - rect.top;
    drag.panX = view.panX;
    drag.panY = view.panY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag.active) return;
    const rect = canvas.getBoundingClientRect();
    const dx = event.clientX - rect.left - drag.x;
    const dy = event.clientY - rect.top - drag.y;
    if (Math.hypot(dx, dy) > 3) drag.moved = true;
    if (!drag.moved) return;
    view.panX = drag.panX + dx;
    view.panY = drag.panY + dy;
    drawMap();
  });
  canvas.addEventListener("pointerup", (event) => {
    const rect = canvas.getBoundingClientRect();
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;
    drag.active = false;
    if (!drag.moved) {
      const robot = nearestRobot(sx, sy);
      if (robot) {
        app.selected = robot;
        app.teleopRobot = robot;
        const select = $("teleopRobot");
        if (select) select.value = robot;
        toast(`${robot} selected for map follow / teleop`);
        drawMap();
      }
    }
  });
  canvas.addEventListener("pointercancel", () => { drag.active = false; });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 0.89;
    view.zoom = Math.max(0.45, Math.min(4.0, view.zoom * factor));
    drawMap();
  }, { passive: false });

  window.addEventListener("resize", drawMap);

  // Clock + polling.
  setInterval(() => {
    $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false });
  }, 500);
  setInterval(loadState, 650);
  setInterval(driveTick, 100);
  loadState();
})();
