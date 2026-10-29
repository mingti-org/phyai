import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import URDFLoader from "urdf-loader";
import { before, joints, alignedError, bounds } from "./data.js";

const $ = (id) => document.getElementById(id);
const colors = {
  state: "#62b7f4",
  command: "#ffb454",
  final: "#6bd5b3",
  prediction: "#e3d289",
  error: "#ff7b84",
};
const app = {
  time: 0,
  duration: 0,
  playing: false,
  speed: 1,
  window: 10,
  arm: "left",
  joint: -1,
  step: 0,
  events: [],
  requests: [],
  timeline: [],
  selected: null,
  detail: [],
  start: 0,
  end: 10,
  generation: 0,
};
let sceneView = null,
  loadTimer = null;
async function api(path) {
  const response = await fetch(path);
  if (!response.ok) throw Error(`${response.status}: ${await response.text()}`);
  return response.json();
}
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}
function series(rows, side, index) {
  return rows.flatMap((row) => {
    const q = joints(row, side);
    return q && Number.isFinite(q[index]) ? [[row.t, q[index]]] : [];
  });
}
function subset(kind) {
  return app.events
    .filter((row) => row.kind === kind)
    .sort((a, b) => a.t - b.t);
}
function sampleRows() {
  return subset("feedback");
}
function commands() {
  return subset("command").filter((row) => row.status === "success");
}
function finalRows(side) {
  return subset("ros_observed")
    .filter(
      (row) =>
        row.topic === `/tj/control/joint_cmd_${side === "left" ? "A" : "B"}`,
    )
    .map((row) => ({
      ...row,
      values: { [`joint_position_${side}`]: row.values.positions },
    }));
}
function plot(
  canvas,
  lines,
  {
    title = "",
    unit = "",
    range = [app.start, app.end],
    cursor = app.time,
    yRange = null,
    shade = null,
  } = {},
) {
  const width = Math.max(240, canvas.clientWidth),
    height = Number(
      canvas.dataset.plotHeight ?? canvas.getAttribute("height") ?? 105,
    ),
    ratio = Math.min(devicePixelRatio, 2);
  canvas.dataset.plotHeight = height;
  canvas.style.height = `${height}px`;
  if (
    canvas.width !== Math.round(width * ratio) ||
    canvas.height !== Math.round(height * ratio)
  ) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const x0 = 56,
    y0 = 23,
    w = width - 70,
    h = height - 43;
  const yr = yRange ?? bounds(lines.map((l) => l.points));
  const x = (t) =>
      x0 + ((t - range[0]) / Math.max(0.00001, range[1] - range[0])) * w,
    y = (v) => y0 + h - ((v - yr[0]) / (yr[1] - yr[0])) * h;
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#b9c8d3";
  ctx.fillText(`${title} ${unit}`, x0, 13);
  for (let i = 0; i < 3; i++) {
    const value = yr[0] + ((yr[1] - yr[0]) * i) / 2;
    ctx.fillStyle = "#9baebc";
    ctx.fillText(value.toFixed(unit === "ms" ? 0 : 2), 4, y(value) + 4);
    ctx.strokeStyle = "#30404b";
    ctx.beginPath();
    ctx.moveTo(x0, y(value));
    ctx.lineTo(x0 + w, y(value));
    ctx.stroke();
  }
  if (shade !== null) {
    ctx.fillStyle = "#ffffff0d";
    ctx.fillRect(x(shade), y0, Math.max(0, x(range[1]) - x(shade)), h);
    ctx.fillStyle = "#9baebc";
    ctx.fillText("Not executed", Math.min(x(shade) + 6, width - 85), y0 + 13);
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, w, h);
  ctx.clip();
  for (const line of lines) {
    ctx.strokeStyle = line.color;
    ctx.lineWidth = line.width ?? 1.5;
    ctx.setLineDash(line.dash ?? []);
    ctx.beginPath();
    let previous = null;
    for (const [t, value] of line.points) {
      if (!Number.isFinite(value)) {
        previous = null;
        continue;
      }
      if (!previous || t - previous > (line.maxGap ?? 0.15))
        ctx.moveTo(x(t), y(value));
      else ctx.lineTo(x(t), y(value));
      previous = t;
    }
    ctx.stroke();
    if (line.dots) {
      ctx.setLineDash([]);
      for (const [t, v] of line.points) {
        ctx.beginPath();
        ctx.arc(x(t), y(v), 2.5, 0, Math.PI * 2);
        ctx.fillStyle = line.color;
        ctx.fill();
      }
    }
  }
  ctx.setLineDash([]);
  ctx.strokeStyle = "#e5edf3aa";
  ctx.beginPath();
  ctx.moveTo(x(cursor), y0);
  ctx.lineTo(x(cursor), y0 + h);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = "#9baebc";
  ctx.fillText(range[0].toFixed(1), x0, height - 4);
  ctx.fillText(range[1].toFixed(1), width - 37, height - 4);
  canvas.onpointerdown = (e) => {
    const rect = canvas.getBoundingClientRect();
    setTime(
      range[0] +
        Math.max(0, Math.min(1, (e.clientX - rect.left - x0) / w)) *
          (range[1] - range[0]),
    );
  };
}
function buildJointPlots() {
  const container = $("joint-plots");
  container.replaceChildren();
  for (let i = 0; i < 7; i++) {
    const canvas = document.createElement("canvas");
    canvas.className = "joint-plot";
    canvas.height = 105;
    canvas.dataset.joint = i;
    canvas.setAttribute(
      "aria-label",
      `${app.arm} joint ${i + 1} feedback and targets`,
    );
    container.append(canvas);
  }
}
function drawPlots() {
  const fb = sampleRows(),
    cmd = commands(),
    final = finalRows(app.arm),
    selected = app.detail.find(
      (row) => row.kind === "response" && row.values?.actions,
    ),
    chunk = app.detail.find(
      (row) => row.kind === "chunk" && row.phase === "started",
    );
  const action = selected?.values?.actions,
    offset = app.arm === "left" ? 0 : 8;
  const indices = app.joint < 0 ? [0, 1, 2, 3, 4, 5, 6] : [app.joint];
  for (const canvas of $("joint-plots").children) {
    const i = Number(canvas.dataset.joint);
    canvas.hidden = !indices.includes(i);
    if (canvas.hidden) continue;
    const prediction =
      action && chunk
        ? action
            .slice(0, chunk.execution_horizon)
            .map((q, k) => [chunk.t + k / chunk.action_hz, q[offset + i]])
        : [];
    plot(
      canvas,
      [
        { points: series(fb, app.arm, i), color: colors.state },
        { points: series(cmd, app.arm, i), color: colors.command },
        { points: series(final, app.arm, i), color: colors.final },
        {
          points: prediction,
          color: colors.prediction,
          dash: [4, 4],
          maxGap: 0.2,
        },
      ],
      { title: `${app.arm.toUpperCase()} J${i + 1}`, unit: "rad" },
    );
  }
  const errorTarget = final.length ? final : cmd;
  plot(
    $("error"),
    indices.map((i, n) => ({
      points: alignedError(fb, errorTarget, app.arm, i),
      color: [
        "#ffb454",
        "#62b7f4",
        "#6bd5b3",
        "#ff7b84",
        "#d4c56f",
        "#c1a5ed",
        "#c4d5df",
      ][n],
    })),
    {
      title: `${final.length ? "Mux" : "Client"} target - feedback / ${app.arm}`,
      unit: "rad",
    },
  );
  for (const side of ["left", "right"]) {
    const limits = app.manifest.metadata?.client?.robot?.gripper_limits_rad ?? [
      0, 1.6,
    ];
    const f = fb.flatMap((row) => {
      const raw = row.values?.[`gripper_feedback_${side}`];
      return raw
        ? [[row.t, (raw[0] - limits[0]) / (limits[1] - limits[0])]]
        : [];
    });
    const c = cmd.flatMap((row) =>
      row.values?.[`gripper_${side}`]
        ? [[row.t, row.values[`gripper_${side}`][0]]]
        : [],
    );
    plot(
      $(`grip-${side}`),
      [
        { points: f, color: colors.state },
        { points: c, color: colors.command },
      ],
      { title: `${side.toUpperCase()} GRIPPER`, unit: "normalized" },
    );
  }
  const sourceNames = [
    ...new Set(app.requests.flatMap((row) => Object.keys(row.ages_ms ?? {}))),
  ];
  const selectedSource = $("age-source").value;
  const sources = sourceNames.filter(
    (key) => selectedSource === "all" || key === selectedSource,
  );
  const ageLines = sources.map((key, i) => ({
    points: app.requests
      .filter((r) => r.ages_ms?.[key] !== undefined)
      .map((r) => {
        const stage = $("age-stage").value;
        const sample = r.samples?.[key];
        const timestamp =
          stage === "read"
            ? r.read_end_ns
            : stage === "check"
              ? r.freshness_checked_at_ns
              : r.t_ns;
        return [
          r.t,
          stage === "request"
            ? r.ages_ms[key]
            : sample && timestamp
              ? (timestamp - sample.received_at_ns) / 1e6
              : NaN,
        ];
      }),
    color: [
      "#62b7f4",
      "#ffb454",
      "#6bd5b3",
      "#c1a5ed",
      "#d4c56f",
      "#b2c4d1",
      "#ff7b84",
    ][i % 7],
    dots: true,
    maxGap: Infinity,
  }));
  const legend = $("age-legend");
  legend.replaceChildren();
  sources.forEach((key, i) => {
    const span = document.createElement("span");
    span.style.color = ageLines[i].color;
    span.textContent = key;
    legend.append(span);
  });
  const limit = app.requests[0]?.max_age_ms ?? 500;
  ageLines.push({
    points: [
      [app.start, limit],
      [app.end, limit],
    ],
    color: colors.error,
    dash: [5, 4],
    maxGap: Infinity,
  });
  plot($("age"), ageLines, {
    title: `Callback age ${$("age-stage").selectedOptions[0].textContent} / max-age threshold`,
    unit: "ms",
    yRange: [
      0,
      ageLines.reduce(
        (maximum, line) =>
          line.points.reduce(
            (m, p) => (Number.isFinite(p[1]) ? Math.max(m, p[1] * 1.05) : m),
            maximum,
          ),
        limit * 1.15,
      ),
    ],
  });
  const executedAge = cmd.map((row) => [
    row.t,
    Math.max(
      0,
      ...Object.entries(row.observation_ages_ms ?? {})
        .filter(([k]) => selectedSource === "all" || k === selectedSource)
        .map(([, v]) => v),
    ),
  ]);
  plot(
    $("age-execution"),
    [
      { points: executedAge, color: colors.command },
      {
        points: [
          [app.start, limit],
          [app.end, limit],
        ],
        color: colors.error,
        dash: [4, 4],
        maxGap: Infinity,
      },
    ],
    {
      title: "Observation age at send (not a second freshness gate)",
      unit: "ms",
    },
  );
  const period =
    1000 / (app.manifest.metadata?.client?.control?.control_hz ?? 200);
  plot(
    $("interval"),
    [
      {
        points: cmd.slice(1).map((r, i) => [r.t, (r.t - cmd[i].t) * 1000]),
        color: colors.state,
      },
      {
        points: [
          [app.start, period],
          [app.end, period],
        ],
        color: colors.command,
        dash: [4, 4],
        maxGap: Infinity,
      },
    ],
    {
      title:
        "Command interval / configured period; chunk refill pauses remain visible",
      unit: "ms",
    },
  );
  plot(
    $("latency"),
    [
      {
        points: app.timeline
          .filter((r) => r.kind === "response")
          .map((r) => [r.t, r.duration_ms ?? 0]),
        color: colors.command,
        dots: true,
        maxGap: Infinity,
      },
      {
        points: cmd.map((r) => [r.t, r.duration_ms ?? 0]),
        color: colors.state,
      },
    ],
    { title: "RPC duration (orange) / send duration (blue)", unit: "ms" },
  );
  if (action) {
    const focus = app.joint < 0 ? 0 : app.joint;
    plot(
      $("chunk"),
      [
        {
          points: action.map((q, k) => [k, q[offset + focus]]),
          color: colors.command,
          maxGap: 2,
        },
        {
          points: action.map((_, k) => [
            k,
            app.detail.find((r) => r.kind === "request")?.values?.state?.[
              offset + focus
            ],
          ]),
          color: colors.state,
          dash: [4, 4],
          maxGap: 2,
        },
      ],
      {
        title: `${app.arm} J${focus + 1} / prediction and observation anchor`,
        unit: "rad",
        range: [0, action.length - 1],
        cursor: app.step,
        shade:
          chunk?.execution_horizon ??
          app.manifest.metadata?.client?.schedule?.execution_horizon ??
          25,
      },
    );
    $("chunk").onpointerdown = (e) => {
      const r = $("chunk").getBoundingClientRect();
      app.step = Math.max(
        0,
        Math.min(
          action.length - 1,
          Math.round(
            ((e.clientX - r.left - 56) / (r.width - 70)) * (action.length - 1),
          ),
        ),
      );
      $("action-step").value = app.step;
      draw();
    };
  }
}
function drawEvents() {
  const canvas = $("events"),
    ctx = canvas.getContext("2d"),
    width = canvas.clientWidth;
  canvas.width = width * devicePixelRatio;
  canvas.height = 48 * devicePixelRatio;
  canvas.style.height = "48px";
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const x = (t) => (t / app.duration) * width;
  ctx.fillStyle = "#18232c";
  ctx.fillRect(0, 0, width, 48);
  const modes = app.timeline.filter(
    (row) => row.kind === "mode" && row.topic === "/tj/control/input_mode",
  );
  modes.forEach((row, index) => {
    ctx.fillStyle = row.value === 3 ? "#62b7f422" : "#ff7b8433";
    ctx.fillRect(
      x(row.t),
      26,
      x(modes[index + 1]?.t ?? app.duration) - x(row.t),
      20,
    );
    ctx.fillStyle = "#c8d5de";
    ctx.font = "10px sans-serif";
    if (x(modes[index + 1]?.t ?? app.duration) - x(row.t) > 45)
      ctx.fillText(
        row.value === 3
          ? "VLA"
          : row.value === 1
            ? "TELEOP"
            : `Mode ${row.value}`,
        x(row.t) + 4,
        40,
      );
  });
  for (const row of app.timeline) {
    ctx.fillStyle =
      row.kind === "request"
        ? colors.state
        : row.kind === "response"
          ? colors.command
          : row.kind === "mode"
            ? colors.error
            : colors.final;
    ctx.fillRect(x(row.t), row.kind === "mode" ? 24 : 8, 2, 16);
  }
  ctx.strokeStyle = "#fff";
  ctx.beginPath();
  ctx.moveTo(x(app.time), 0);
  ctx.lineTo(x(app.time), 48);
  ctx.stroke();
  canvas.onclick = (e) =>
    setTime(
      ((e.clientX - canvas.getBoundingClientRect().left) / width) *
        app.duration,
    );
}
function draw() {
  $("clock").textContent =
    `${app.time.toFixed(2)} / ${app.duration.toFixed(2)} s`;
  $("timeline").value = app.time;
  $("action-label").textContent = String(app.step);
  drawEvents();
  drawPlots();
  updatePose();
}
async function loadRange() {
  const generation = ++app.generation;
  const half = app.window / 2;
  app.start = Math.max(
    0,
    Math.min(app.time - half, Math.max(0, app.duration - app.window)),
  );
  app.end = Math.min(app.duration, app.start + app.window);
  const rows = [];
  let after = 0;
  do {
    const data = await api(
      `/api/events?start=${Math.max(0, app.start - 0.2)}&end=${app.end}&limit=20000&after=${after}&kinds=feedback,command,ros_observed,mode`,
    );
    if (generation !== app.generation) return;
    rows.push(...data.events);
    after = data.next_after;
  } while (after);
  app.events = rows.sort((a, b) => a.t - b.t);
  draw();
}
async function selectRequest(id) {
  app.selected = Number(id);
  $("request").value = id;
  const data = await api(`/api/request?id=${encodeURIComponent(id)}`);
  if (app.selected !== Number(id)) return;
  app.detail = data.events;
  const request = data.events.find((row) => row.kind === "request");
  const response = data.events.find((row) => row.kind === "response");
  for (const camera of ["head_left", "left_wrist", "right_wrist"]) {
    const path = request?.images?.[camera];
    if (path) $(camera).src = `/data/${path}`;
    else $(camera).removeAttribute("src");
  }
  $("input-label").textContent =
    `Request ${id} / ${request?.t.toFixed(3)} s${request?.warmup ? " / warmup, discarded" : ""}`;
  const horizon = response?.values?.actions?.length ?? 50;
  $("action-step").max = horizon - 1;
  app.step = Math.min(app.step, horizon - 1);
  $("chunk-note").textContent =
    `${horizon} predicted targets. Shaded targets are not executed. Selected request state remains the reference anchor.`;
  draw();
}
function setTime(time) {
  app.time = Math.max(0, Math.min(app.duration, time));
  const request = before(app.requests, app.time);
  if (request && request.request_id !== app.selected)
    selectRequest(request.request_id).catch((e) => notice(e.message, true));
  if (
    (app.time < app.start + 0.3 || app.time > app.end - 0.3) &&
    loadTimer === null
  ) {
    loadTimer = setTimeout(() => {
      loadTimer = null;
      loadRange().catch((e) => notice(e.message, true));
    }, 80);
  }
  draw();
}
async function initScene() {
  if (!app.manifest.robot_available) {
    $("scene-status").textContent =
      "No robot assets in this trace. Supply --robot-assets to demo or robot_assets to recording.";
    return;
  }
  const container = $("scene"),
    renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  container.append(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#202c35");
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 30);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x405060, 2.6));
  const light = new THREE.DirectionalLight(0xffffff, 2.5);
  light.position.set(2, -3, 5);
  scene.add(light);
  const grid = new THREE.GridHelper(8, 40, 0x526773, 0x33424d);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
  const manager = new THREE.LoadingManager();
  const ready = new Promise((resolve, reject) => {
    manager.onLoad = resolve;
    manager.onError = (url) => reject(Error(`Robot asset unavailable: ${url}`));
  });
  const loader = new URDFLoader(manager);
  loader.packages = { marvin_description: "/robot/marvin_description" };
  const robot = await loader.loadAsync("/robot/robot.urdf");
  await ready;
  const ghost = robot.clone(true);
  for (const [object, color, opacity] of [
    [robot, colors.state, 1],
    [ghost, colors.command, 0.4],
  ]) {
    object.traverse((o) => {
      if (o.isMesh)
        o.material = new THREE.MeshStandardMaterial({
          color,
          transparent: opacity < 1,
          opacity,
          depthWrite: opacity === 1,
          roughness: 0.8,
        });
    });
    for (const joint of Object.values(object.joints)) joint.ignoreLimits = true;
    scene.add(object);
  }
  const reset = () => {
    const box = new THREE.Box3().setFromObject(robot),
      center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(2.8, -2.7, 1.8));
    controls.update();
    render();
  };
  function render() {
    renderer.render(scene, camera);
  }
  new ResizeObserver(() => {
    renderer.setSize(container.clientWidth, container.clientHeight);
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    render();
  }).observe(container);
  controls.addEventListener("change", render);
  $("reset").onclick = reset;
  sceneView = { robot, ghost, render };
  reset();
  $("scene-status").textContent =
    "Offline URDF pose / raw angles; limits highlighted";
  const ray = new THREE.Raycaster();
  renderer.domElement.addEventListener("click", (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(
      new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        1 - ((event.clientY - rect.top) / rect.height) * 2,
      ),
      camera,
    );
    let node = ray.intersectObject(robot, true)[0]?.object;
    while (node && !/^Joint[1-7]_[LR]$/.test(node.name)) node = node.parent;
    if (node) {
      const match = /Joint([1-7])_([LR])/.exec(node.name);
      app.arm = match[2] === "L" ? "left" : "right";
      app.joint = Number(match[1]) - 1;
      $("joint").value = app.joint;
      buildJointPlots();
      draw();
    }
  });
  updatePose();
}
function updatePose() {
  if (!sceneView) return;
  const feedback = sampleRows(),
    cmd = before(commands(), app.time);
  const request = app.detail.find((row) => row.kind === "request"),
    actions = app.detail.find((row) => row.kind === "response")?.values
      ?.actions;
  const mode = $("pose-mode").value;
  const q = {},
    target = {};
  let missing = false;
  const currentMode = before(
    app.timeline.filter(
      (r) => r.kind === "mode" && r.topic === "/tj/control/input_mode",
    ),
    app.time,
  );
  $("scene-status").textContent =
    `${mode === "action" ? "Request observation vs selected prediction" : "Feedback vs selected target"} / control: ${currentMode?.value === 3 ? "VLA" : currentMode?.value === 1 ? "TELEOP" : (currentMode?.value ?? "unknown")}`;
  for (const side of ["left", "right"]) {
    const fb = before(
      feedback.filter((row) => joints(row, side)),
      app.time,
    );
    const actual =
      mode === "action"
        ? joints(request, side)
        : app.time - (fb?.t ?? -Infinity) < 0.5
          ? joints(fb, side)
          : null;
    let goal =
      app.time - (cmd?.t ?? -Infinity) < 0.15 ? joints(cmd, side) : null;
    if (mode === "action")
      goal = actions?.[app.step]?.slice(
        side === "left" ? 0 : 8,
        side === "left" ? 7 : 15,
      );
    if (mode === "final") {
      const last = before(finalRows(side), app.time);
      goal =
        app.time - (last?.t ?? -Infinity) < 0.15 ? joints(last, side) : null;
    }
    if (!goal) missing = true;
    for (let i = 0; i < 7; i++) {
      const name = `Joint${i + 1}_${side === "left" ? "L" : "R"}`;
      if (actual) q[name] = actual[i];
      if (goal) target[name] = goal[i];
    }
  }
  sceneView.robot.visible = Object.keys(q).length > 0;
  sceneView.ghost.visible = !missing;
  sceneView.robot.setJointValues(q);
  sceneView.ghost.setJointValues(target);
  sceneView.robot.updateMatrixWorld(true);
  sceneView.ghost.updateMatrixWorld(true);
  const over = Object.entries(target)
    .filter(([name, value]) => {
      const joint = sceneView.ghost.joints[name];
      return (
        joint &&
        joint.jointType === "revolute" &&
        (value < joint.limit.lower || value > joint.limit.upper)
      );
    })
    .map(([name]) => name);
  $("limits").textContent = over.length
    ? `Outside URDF limits: ${over.join(", ")}`
    : missing
      ? "Target unavailable in this interval"
      : "";
  sceneView.render();
}
async function main() {
  app.manifest = await api("/api/manifest");
  app.duration = Math.max(
    0.01,
    app.manifest.duration ?? app.manifest.bounds?.[1] ?? 0,
  );
  let after = 0;
  do {
    const data = await api(`/api/timeline?after=${after}`);
    app.timeline.push(...data.events);
    after = data.next_after;
  } while (after);
  app.timeline.sort((a, b) => a.t - b.t);
  app.requests = app.timeline.filter((row) => row.kind === "request");
  $("timeline").max = app.duration;
  $("record-status").textContent =
    `${app.manifest.complete ? "Complete trace" : "Incomplete / recording"} · ${app.manifest.event_count} events`;
  $("integrity").textContent =
    `Dropped events: ${app.manifest.dropped_events ?? 0} · missing images: ${app.manifest.dropped_images ?? 0} · ${app.manifest.errors?.join("; ") || "No recorded writer errors"}`;
  if (app.manifest.metadata?.synthetic)
    notice("SYNTHETIC DEMO — generated signals; no robot was connected.");
  else if (!app.manifest.complete)
    notice(
      "This trace is incomplete. Gaps must not be interpreted as continuous telemetry.",
      true,
    );
  for (const source of [
    ...new Set(app.requests.flatMap((r) => Object.keys(r.ages_ms ?? {}))),
  ])
    $("age-source").add(new Option(source, source));
  $("age-source").onchange = draw;
  $("age-stage").onchange = draw;
  for (let i = 0; i < 7; i++) $("joint").add(new Option(`Joint ${i + 1}`, i));
  for (const row of app.requests)
    $("request").add(
      new Option(
        `#${row.request_id} · ${row.t.toFixed(2)}s${row.warmup ? " warmup" : ""}`,
        row.request_id,
      ),
    );
  buildJointPlots();
  $("timeline").oninput = (e) => setTime(Number(e.target.value));
  $("request").onchange = (e) => {
    selectRequest(e.target.value);
    const row = app.requests.find(
      (r) => r.request_id === Number(e.target.value),
    );
    if (row) {
      app.time = row.t;
      loadRange();
    }
  };
  $("action-step").oninput = (e) => {
    app.step = Number(e.target.value);
    $("pose-mode").value = "action";
    draw();
  };
  $("joint").onchange = (e) => {
    app.joint = Number(e.target.value);
    draw();
  };
  $("pose-mode").onchange = draw;
  $("window").onchange = (e) => {
    app.window = Number(e.target.value);
    loadRange();
  };
  $("speed").onchange = (e) => (app.speed = Number(e.target.value));
  document.querySelectorAll("[data-arm]").forEach(
    (button) =>
      (button.onclick = () => {
        app.arm = button.dataset.arm;
        document
          .querySelectorAll("[data-arm]")
          .forEach((b) => b.classList.toggle("selected", b === button));
        buildJointPlots();
        draw();
      }),
  );
  const step = (direction) => {
    const i = app.requests.findIndex((r) => r.request_id === app.selected);
    const row =
      app.requests[
        Math.max(0, Math.min(app.requests.length - 1, i + direction))
      ];
    if (row) setTime(row.t);
  };
  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  let last = null;
  function animate(now) {
    if (!app.playing) return;
    if (last !== null) setTime(app.time + ((now - last) / 1000) * app.speed);
    last = now;
    if (app.time >= app.duration) {
      app.playing = false;
      $("play").textContent = "Play";
      last = null;
      return;
    }
    requestAnimationFrame(animate);
  }
  $("play").onclick = () => {
    app.playing = !app.playing;
    $("play").textContent = app.playing ? "Pause" : "Play";
    last = null;
    if (app.playing) requestAnimationFrame(animate);
  };
  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
    if (e.code === "Space") {
      e.preventDefault();
      $("play").click();
    }
    if (e.code === "ArrowRight") step(1);
    if (e.code === "ArrowLeft") step(-1);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && app.playing) $("play").click();
  });
  await loadRange();
  if (app.requests.length) await selectRequest(app.requests[0].request_id);
  initScene().catch((e) => {
    $("scene-status").textContent = e.message;
  });
  window.addEventListener("resize", draw);
  window.traceInspector = { state: app, setTime };
  draw();
}
main().catch((error) => notice(error.message, true));
