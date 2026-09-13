// Two tabs. Live: start a session, follow its server-sent events, then replay.
// Results: fetch /api/results once and draw it. No libraries; charts are small SVG.

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const show = (v, digits = 0) => (v === null || v === undefined ? "—" : Number(v).toFixed(digits));
const COLOUR = { light: "#34D399", medium: "#F59E0B", heavy: "#E11D48" };
// An uploaded video has no hand-given class, so its point is drawn neutral rather
// than borrowing one of the three dataset colours.
const UNLABELLED = "#94A3B8";

// -- tabs -----------------------------------------------------------------------
// Results are fetched every time the tab is opened, not once: they are measured from the
// saved detections, so playing a clip or removing one changes them mid-session. The
// server answers from its own cache while nothing on disk has changed, so this is cheap.
document.querySelectorAll(".tabs button").forEach((tab) => (tab.onclick = () => {
  document.querySelectorAll(".tabs button").forEach((t) => t.setAttribute("aria-selected", t === tab));
  $("live").hidden = tab.dataset.tab !== "live";
  $("results").hidden = tab.dataset.tab !== "results";
  if (tab.dataset.tab === "results") loadResults();
}));

// -- 1. choose a video ------------------------------------------------------------
// The sidebar shows the videos *this app* holds: the ones uploaded, and any dataset
// clip that already has a saved detection. It starts empty and Clear all returns it to
// empty. The dataset itself is not browsable here -- the app is driven by what you drop
// into it, and archive/ stays untouched whatever the app does.
let clips = [];
let uploads = [];
const loadClips = () => fetch("/api/clips").then((r) => r.json()).then((list) => { clips = list; draw(); });
const loadUploads = () => fetch("/api/uploads").then((r) => r.json()).then((list) => { uploads = list; draw(); });
const reload = () => { loadClips(); loadUploads(); };
reload();

function draw() { drawVideos(); }

/** The videos in this app: every upload, plus the dataset clips that have a detection. */
function appVideos() {
  return [
    // An analysed upload shows the CNN's class; one analysed before the class was saved
    // falls back to "analysed" until it is played again.
    ...uploads.map((u) => ({ kind: "upload", key: u.id, label: u.name,
                             tag: !u.cached ? "not analysed" : u.traffic_class || "analysed",
                             tone: !u.cached ? "medium" : u.traffic_class || "light" })),
    ...clips.filter((c) => c.cached).map((c) => ({ kind: "clip", key: c.name, label: c.file,
                                                   tag: c.traffic_class, tone: c.traffic_class })),
  ];
}

function drawVideos() {
  const videos = appVideos();
  const active = document.querySelector("#videos button.active")?.dataset.key;
  $("video-count").textContent = videos.length;
  $("videos").innerHTML = videos.map((v) =>
    `<li><button data-key="${esc(v.key)}" data-kind="${v.kind}" class="${v.key === active ? "active" : ""}">
       <span class="file">${esc(v.label)}</span><span class="pill ${v.tone}">${esc(v.tag)}</span></button>
     <button class="forget" data-remove="${esc(v.key)}" data-kind="${v.kind}"
       title="Remove this video and its results from the app">Remove</button></li>`).join("")
    || `<li class="muted">No videos yet. Drop one above to analyse it.</li>`;
  $("clear-cache").disabled = videos.length === 0;
  $("clear-cache").textContent = `Clear all videos (${videos.length})`;
}

// Removing a video takes its detection and its results with it. A dataset clip keeps
// its file in archive/ and simply leaves the app; an upload is deleted outright.
async function removeCache(url, question, clip) {
  if (!confirm(question)) return;
  const response = await fetch(url, { method: "DELETE" });
  const body = await response.json();
  if (!response.ok) { setNote(body.detail); return; }
  if (session && (clip === undefined || session.name === clip)) closeSession();
  setNote(`Removed ${body.removed} video${body.removed === 1 ? "" : "s"} from the app.`);
  reload();
}
$("clear-cache").onclick = () => removeCache("/api/cache",
  "Remove every video from this app?\n\n" +
  "The list empties and the Results page goes blank. Uploaded videos are deleted; " +
  "dataset clips stay in archive/ and can be added again.\n\n" +
  "The annotated videos, figures and report on disk are not touched.");
// The app's own list: play a video again, or take it out of the app.
$("videos").onclick = (e) => {
  const remove = e.target.closest("button[data-remove]");
  if (remove) {
    const { remove: key, kind } = remove.dataset;
    return kind === "upload"
      ? removeCache(`/api/uploads/${encodeURIComponent(key)}`,
          "Remove this uploaded video? Its video file, its detection and its results are deleted.")
      : removeCache(`/api/cache/${encodeURIComponent(key)}`,
          `Remove ${key} from the app? Its results go with it; the dataset video stays in archive/.`, key);
  }
  const button = e.target.closest("button[data-key]");
  if (!button) return;
  $("videos").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === button));
  const form = new FormData();
  form.append(button.dataset.kind === "upload" ? "upload" : "clip", button.dataset.key);
  start(form, button.querySelector(".file").textContent);
};
$("file").onchange = (e) => upload(e.target.files[0]);
$("drop").ondragover = (e) => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = (e) => { e.preventDefault(); $("drop").classList.remove("over"); upload(e.dataTransfer.files[0]); };

function upload(file) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  start(form, file.name);
}

// -- 2. watch it analysed live ------------------------------------------------------
let session = null, source = null, frames = [], stats = [], replayTimer = null;

async function start(form, label) {
  stopReplay();
  if (source) source.close();
  setStatus(`starting ${label}…`);
  $("note").hidden = true;
  const response = await fetch("/api/live", { method: "POST", body: form });
  const body = await response.json();
  if (!response.ok) { setNote(body.detail); setStatus("idle"); return; }

  session = body; frames = []; stats = [];
  $("fps").value = Math.min(60, Math.max(1, Math.round(session.fps)));   // start at real time
  $("fps-value").textContent = $("fps").value;
  resetView();
  if (session.uploaded) setNote("Speed, density and LOS use the I-5 camera calibration. On another camera trust only the counts, vehicle types and tracking.");
  $("stop").hidden = false;
  showWorking("Loading the video…", 0, 1);
  // The server analyses the whole video first; the page only shows progress, then plays it.
  source = new EventSource(`/api/live/${session.id}/events`);
  source.addEventListener("progress", (e) => { const p = JSON.parse(e.data); showWorking(p.step, p.done, p.total); });
  source.addEventListener("frame", (e) => frames.push(JSON.parse(e.data).n));
  source.addEventListener("stats", (e) => stats.push(JSON.parse(e.data)));
  source.addEventListener("end", (e) => {
    source.close();
    $("working").hidden = true;
    finished(JSON.parse(e.data).state);
    // A fresh detection was saved (the video is now in the app), or an upload's class was.
    if (!session.cached_tracks || session.uploaded) reload();
    if (frames.length) { preload(); $("timeline").max = frames.length - 1; seek(0); play(0); }
  });
  source.addEventListener("error", (e) => {
    if (!e.data) return;                       // a connection hiccup: EventSource retries by itself
    source.close();
    $("working").hidden = true;
    setNote(JSON.parse(e.data).message);
    finished("failed");
  });
}

function showWorking(step, done, total) {
  $("working").hidden = false;
  $("working-step").textContent = step;
  $("working-bar").style.width = `${total ? (100 * done) / total : 0}%`;
  $("working-count").textContent = total > 1 ? `${done} / ${total}` : "";
  setStatus("analysing…");
}

$("stop").onclick = () => session && fetch(`/api/live/${session.id}/stop`, { method: "POST" });

function finished(state) {
  $("stop").hidden = true;
  $("replay").disabled = frames.length === 0;
  $("timeline").disabled = frames.length === 0;
  setStatus(state === "done" ? `0.0 s / ${(frames.length / (session.fps || 10)).toFixed(1)} s` : state);
}

// The server dropped this session, so its frames are gone: back to the empty screen.
function closeSession() {
  stopReplay();
  if (source) source.close();
  session = null; frames = []; stats = [];
  $("frame").hidden = true; $("empty").hidden = false; $("working").hidden = true; $("stop").hidden = true;
  $("replay").disabled = true; $("timeline").disabled = true;
  document.querySelectorAll("#clips button.active").forEach((b) => b.classList.remove("active"));
  setStatus("idle");
}

function resetView() {
  $("empty").hidden = true; $("frame").hidden = false;
  $("replay").disabled = true; $("timeline").disabled = true;
  ["cnn", "los", "speed", "density", "counted", "inview"].forEach((id) => ($(id).textContent = "—"));
  $("cnn-conf").textContent = ""; $("insight").textContent = "Detecting and tracking…";
  $("cnn-panel").className = "glass tl"; $("los-panel").className = "glass tr";
  $("spark").innerHTML = ""; $("lanes").innerHTML = ""; $("types").innerHTML = ""; $("changes").textContent = "";
  $("vehicles").innerHTML = `<tr><td colspan="4" class="muted">Nothing yet.</td></tr>`;
}

function showFrame(n) { $("frame").src = `/api/live/${session.id}/frames/${n}.jpg`; }

// The insight line "Slowing: 60 km/h, 62% of the limit · LOS C · CNN reads medium" as a strip:
// state pill, speed, a bar for the share of the limit, then LOS and CNN chips.
function drawInsight(s) {
  const m = /^(Free-flowing|Slowing|Stop-and-go): (\d+) km\/h, (\d+)% of the limit/.exec(s.insight || "");
  if (!m) { $("insight").className = "card insight"; $("insight").textContent = s.insight || ""; return; }
  const [, state, kmh, percent] = m;
  const tone = { "Free-flowing": "flow", "Slowing": "slow", "Stop-and-go": "jam" }[state];
  const c = s.congestion;
  $("insight").className = `card insight ${tone}`;
  $("insight").innerHTML = `
    <span class="state">${state}</span>
    <span class="kmh"><b>${kmh}</b> km/h</span>
    <span class="limit"><i style="width:${Math.min(100, percent)}%"></i></span>
    <span class="of">${percent}% of the limit</span>
    ${s.level_of_service ? `<span class="chip">LOS <b>${esc(s.level_of_service)}</b></span>` : ""}
    ${c ? `<span class="chip">CNN <b>${esc(c.label)}</b></span>` : ""}`;
}

function drawStats(s) {
  const c = s.congestion;
  $("cnn").textContent = c ? c.label : "warming up";
  $("cnn-conf").textContent = c ? `${Math.round(c.confidence * 100)}%` : "";
  $("cnn-panel").className = `glass tl ${c ? "cnn-" + c.label : ""}`;
  $("los").textContent = s.level_of_service || "—";
  $("los-panel").className = `glass tr ${s.level_of_service ? "los-" + s.level_of_service : ""}`;
  drawInsight(s);
  $("speed").textContent = show(s.speed_median_kmh);
  $("speed-sub").textContent = s.speed_p85_kmh != null ? `km/h · 85th percentile ${show(s.speed_p85_kmh)}` : "km/h";
  $("density").textContent = show(s.density_pc_mi_ln, 1);
  $("counted").textContent = show(s.vehicles_counted);
  $("tracks").textContent = s.tracks_seen != null ? `${s.tracks_seen} tracks seen` : "";
  $("inview").textContent = show(s.in_view);
  $("lanes").innerHTML = bars(Object.entries(s.lane_occupancy || {}).map(([l, v]) => [`Lane ${l}`, v, `${Math.round(v * 100)}%`]), 1);
  $("changes").textContent = s.lane_changes != null ? `${s.lane_changes} lane changes` : "";
  const types = Object.entries(s.by_class || {});
  $("types").innerHTML = bars(types.map(([k, v]) => [k, v, v]), Math.max(1, ...types.map((t) => t[1])));
  $("vehicles").innerHTML = (s.vehicles || []).map((v) =>
    `<tr><td>#${v.id}</td><td>${esc(v.class)}</td><td class="num">${show(v.lane)}</td><td class="num">${show(v.speed_kmh)}</td></tr>`).join("")
    || `<tr><td colspan="4" class="muted">No vehicles in this frame.</td></tr>`;
  drawSpark(stats.filter((x) => x.n <= s.n));
}

function bars(rows, max) {
  return rows.map(([label, value, text]) =>
    `<div class="bar"><span>${esc(label)}</span><i style="width:${(100 * value) / max}%"></i><em>${text}</em></div>`).join("")
    || `<p class="muted">Nothing yet.</p>`;
}

function drawSpark(series) {
  const points = series.filter((s) => s.speed_median_kmh != null);
  if (points.length < 2) { $("spark").innerHTML = `<text x="8" y="50">measuring…</text>`; return; }
  const top = Math.max(120, ...points.map((p) => p.speed_median_kmh));
  const tMax = Math.max(...points.map((p) => p.t));
  const xy = points.map((p) => `${(10 + (280 * p.t) / tMax).toFixed(1)},${(80 - (68 * p.speed_median_kmh) / top).toFixed(1)}`);
  $("spark").innerHTML = `<polyline points="${xy.join(" ")}" fill="none" stroke="#3730A3" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
    <text x="10" y="11">${Math.round(top)} km/h</text><text x="262" y="89">${tMax}s</text>`;
}

function setStatus(text) { $("status").textContent = text; }
function setNote(text) { $("note").textContent = text; $("note").hidden = false; }

// -- 3. replay ---------------------------------------------------------------------
function seek(index) {
  const n = frames[index];
  showFrame(n);
  const latest = stats.filter((s) => s.n <= n).pop();
  if (latest) drawStats(latest);
  $("timeline").value = index;
  // video time, not frame numbers: frame index ÷ the video's own fps
  const fps = session.fps || 10;
  setStatus(`${((index + 1) / fps).toFixed(1)} s / ${(frames.length / fps).toFixed(1)} s`);
}
function stopReplay() { clearInterval(replayTimer); replayTimer = null; $("replay").textContent = "Replay"; }
// Playback speed is the FPS slider: 10 is real time for these clips, lower is slow motion.
const playbackFps = () => Number($("fps").value);
function play(from) {
  clearInterval(replayTimer);
  let index = from;
  $("replay").textContent = "Pause";
  replayTimer = setInterval(() => {
    seek(index++);
    if (index >= frames.length) stopReplay();
  }, 1000 / playbackFps());
}
$("fps").oninput = () => {
  $("fps-value").textContent = playbackFps();
  if (replayTimer) play(Number($("timeline").value));   // keep playing from here at the new speed
};
// Fetch every frame up front so a high FPS is not held back by image downloads.
function preload() { frames.forEach((n) => { new Image().src = `/api/live/${session.id}/frames/${n}.jpg`; }); }
$("replay").onclick = () => {
  if (replayTimer) return stopReplay();
  play(Number($("timeline").value) >= frames.length - 1 ? 0 : Number($("timeline").value));
};
$("timeline").oninput = (e) => { stopReplay(); seek(Number(e.target.value)); };

// -- results -----------------------------------------------------------------------
async function loadResults() {
  const response = await fetch("/api/results");
  const body = await response.json();
  if (!response.ok) { drawBlank(body.detail); return; }
  const f = body.fundamental, m = body.classification;
  const tile = (label, value, sub) => `<div class="tile"><span>${label}</span><b>${value}</b><small>${sub}</small></div>`;
  $("results-body").innerHTML = `
    <div class="tiles">
      ${tile("Videos analysed", body.clips - body.uploads,
             `of 254 dataset clips${body.uploads ? ` · + ${body.uploads} upload${body.uploads === 1 ? "" : "s"}` : ""}`)}
      ${tile("Free-flow speed", f ? f.free_flow_speed_kmh.toFixed(0) : "—", `km/h · posted ${body.limit_kmh.toFixed(0)}`)}
      ${tile("Capacity", f ? f.capacity_veh_per_h_per_ln.toFixed(0) : "—", "veh/h/lane")}
      ${tile("Critical density", f ? f.critical_density.toFixed(0) : "—", "pc/mi/lane")}
      ${tile("Congestion model", m ? `${(m.accuracy * 100).toFixed(1)}%` : "—", "correct on recordings it never saw")}
    </div>
    <div class="results-grid">
      <div class="card"><h2>Speed vs density · ${body.clips} clip${body.clips === 1 ? "" : "s"}</h2>${scatter(body.points, f)}
        <div class="legend">${Object.entries(COLOUR).map(([k, v]) => `<span><i style="background:${v}"></i>${k}</span>`).join("")}${body.points.some((p) => !p.traffic_class) ? `<span><i style="background:${UNLABELLED}"></i>uploaded</span>` : ""}</div></div>
      ${body.by_hour.length ? `<div class="card"><h2>Median speed by hour</h2>${hourBars(body.by_hour, body.limit_kmh)}</div>` : ""}
      <div class="card"><h2>Level of service mix</h2>${bars(Object.entries(body.los).map(([k, v]) => [`LOS ${k}`, v, v]), Math.max(1, ...Object.values(body.los)))}</div>
      <div class="card"><h2>How accurate is the congestion classifier</h2>${classification(m)}</div>
    </div>
    <div class="card" style="margin-top:16px"><h2>Recommendations</h2>
      ${body.recommendations.length
        ? `<ol class="recs">${body.recommendations.map((r) => `<li><b>${esc(r.title)}</b> ${esc(r.text)}</li>`).join("")}</ol>`
        : `<p class="muted">Every recommendation quotes the measured capacity and critical density, and
           ${body.clips} clip${body.clips === 1 ? "" : "s"} ${body.clips === 1 ? "is" : "are"} too few to fit
           the speed–density curve they come from. Play more clips, or run <code>python run.py detect</code>
           for the whole dataset.</p>`}</div>`;
}

// Nothing has been analysed, so there is no chart to draw. The server's sentence is a
// correct error and a poor first screen: it names a command before it says what this
// page is for. So say what is missing, then offer the one action that fills it. Which
// line leads depends on whether the app already holds a video -- an empty app needs a
// video, an app whose detections were removed needs one played again.
function drawBlank(detail) {
  const empty = appVideos().length === 0;
  $("results-body").innerHTML = `
    <div class="card blank">
      <h2>Results</h2>
      <p class="lead">No results to show</p>
      <p>${empty
        ? "Add a video and its results appear here: speed, density, vehicle counts and the congestion reading, measured from the video itself."
        : esc(detail)}</p>
      <p class="muted">A few videos also fit the speed–density curve — free-flow speed, capacity,
        critical density — and the recommendations that follow from it.</p>
      <p class="blank-do">
        <button class="primary" id="go-live">${empty ? "Add a video" : "Go to Live"}</button>
        ${empty ? "<small>or run <code>python run.py detect</code> for the whole dataset</small>" : ""}
      </p>
    </div>`;
  // Switching tabs through the tab button itself keeps aria-selected and the panels in step.
  $("go-live").onclick = () => document.querySelector('.tabs button[data-tab="live"]').click();
}

function scatter(points, fit) {
  const W = 520, H = 300, P = 36;
  const dMax = Math.max(10, ...points.map((p) => p.density)) * 1.05, sMax = 130;
  const x = (d) => P + ((W - 2 * P) * d) / dMax, y = (s) => H - P - ((H - 2 * P) * s) / sMax;
  const dots = points.map((p) => `<circle cx="${x(p.density).toFixed(1)}" cy="${y(p.speed).toFixed(1)}" r="4" fill="${COLOUR[p.traffic_class] || UNLABELLED}" opacity=".85"><title>${p.clip}: ${p.speed.toFixed(0)} km/h, ${p.density.toFixed(1)} pc/mi/ln</title></circle>`).join("");
  let line = "";
  if (fit) {
    const end = Math.min(fit.jam_density, dMax);
    line = `<line x1="${x(0)}" y1="${y(fit.free_flow_speed_kmh)}" x2="${x(end)}" y2="${y(fit.free_flow_speed_kmh * (1 - end / fit.jam_density))}" stroke="#3730A3" stroke-width="2" stroke-dasharray="6 4"/>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Speed against density">
    <line x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}" stroke="#CBC7C0"/><line x1="${P}" y1="${P}" x2="${P}" y2="${H - P}" stroke="#CBC7C0"/>
    ${line}${dots}<text x="${W - P - 110}" y="${H - 10}">density pc/mi/ln</text><text x="4" y="${P - 12}">km/h</text></svg>`;
}

function hourBars(rows, limit) {
  const max = Math.max(limit, ...rows.map((r) => r.median_speed_kmh));
  return `<div class="bars">${bars(rows.map((r) => [`${String(r.hour).padStart(2, "0")}:00`, r.median_speed_kmh, r.median_speed_kmh.toFixed(0)]), max)}</div>`;
}

function classification(m) {
  if (!m) return `<p class="muted">models/eval_finetune.json not found.</p>`;
  const head = m.classes.map((c) => `<th>${c}</th>`).join("");
  const body = m.confusion.map((row, i) => `<tr><th>${m.classes[i]}</th>${row.map((v, j) => `<td class="${i === j ? "hit" : ""}">${v}</td>`).join("")}</tr>`).join("");
  return `<p><b>${(m.accuracy * 100).toFixed(1)}%</b> accuracy · <b>${m.macro_f1.toFixed(3)}</b> macro-F1 on all 254 clips, with whole recordings held out.
    Always answering "light" would score ${(m.majority_accuracy * 100).toFixed(0)}%.</p>
    <h2>Recall per class</h2><div class="bars">${bars(m.classes.map((c) => [c, m.recall[c], `${Math.round(m.recall[c] * 100)}%`]), 1)}</div>
    <table class="confusion" style="margin-top:12px"><thead><tr><th>true ↓ · predicted →</th>${head}</tr></thead><tbody>${body}</tbody></table>`;
}
