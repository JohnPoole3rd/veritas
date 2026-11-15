// The Lab: an urban heat & canopy field notebook for any city (owner: D).
// Talks only to the routes in web/API.md. Class labels are model estimates; NDVI and LST are measurements.
import { $, $$, api, esc, fmt, pct, niceDate, toast, pollJob, drawScatter, heatColor, clamp, term } from "./main.js";

const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";
const S = {
  map: null, meta: null, box: null, boxName: null, when: "", region: null, climate: null, mode: "rgb",
  opacity: .8, match: null, matchLabel: "", selected: null, drawing: false, busy: false, classes: {}, estimateOk: false,
};
const STORE = "pe-compare-v1";

export function initLab(meta) {
  S.meta = meta;
  S.classes = Object.fromEntries((meta?.classes || []).map((c) => [c.name, c]));
  initWhen(); initPresets(); initTabs(); initGeocode(); initAsk(); renderCompare();
  $("#mode-pill").textContent = meta ? meta.mode : "OFFLINE";
  $("#mode-pill").classList.toggle("real", meta?.mode === "REAL");
  if (meta?.disclaimer) $("#disclaimer").textContent = `${meta.disclaimer} Land cover is a model estimate; NDVI and surface temperature are satellite measurements.`;
  if (meta?.geo_available) { $("#est-info").textContent = meta.geo_available; }
  // build the map lazily when the lab is close to view
  const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { io.disconnect(); initMap(); } }, { rootMargin: "600px" });
  io.observe($("#lab-app"));
}

// ------------------------------------------------------------------ map
function initMap() {
  if (!window.maplibregl) { $("#lab-map").innerHTML = `<div class="empty" style="margin:80px 20px;color:#ccc">The map library failed to load.</div>`; return; }
  const ua = (S.meta?.presets || [])[0];
  const map = (S.map = new maplibregl.Map({
    container: "lab-map", center: ua ? center(ua.bbox) : [-83.07, 40.03], zoom: 11.2, attributionControl: { compact: true },
    style: {
      version: 8,
      sources: {
        esri: { type: "raster", tileSize: 256, maxzoom: 19, attribution: "Imagery © Esri, Maxar, Earthstar Geographics", tiles: [`${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`] },
        labels: { type: "raster", tileSize: 256, maxzoom: 19, tiles: [`${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`] },
      },
      layers: [
        { id: "bg", type: "background", paint: { "background-color": "#0a120d" } },
        { id: "esri", type: "raster", source: "esri", paint: { "raster-saturation": -.15, "raster-brightness-max": .9 } },
        { id: "labels", type: "raster", source: "labels", paint: { "raster-opacity": .8 } },
      ],
    },
  }));
  map.addControl(new maplibregl.NavigationControl({ showCompass: true, visualizePitch: true }), "bottom-right");
  map.on("error", () => {});
  map.on("load", () => {
    const empty = { type: "FeatureCollection", features: [] };
    map.addSource("box", { type: "geojson", data: empty });
    map.addSource("tiles", { type: "geojson", data: empty, promoteId: "i" });
    map.addSource("sel", { type: "geojson", data: empty });
    map.addLayer({ id: "box-fill", type: "fill", source: "box", paint: { "fill-color": "#d9ec7c", "fill-opacity": .08 } }, "labels");
    map.addLayer({ id: "box-line", type: "line", source: "box", paint: { "line-color": "#d9ec7c", "line-width": 2, "line-dasharray": [2, 1.5] } });
    map.addLayer({ id: "tiles-fill", type: "fill", source: "tiles", paint: { "fill-color": "#000", "fill-opacity": 0 } }, "labels");
    map.addLayer({ id: "tiles-line", type: "line", source: "tiles", paint: { "line-color": "#ffffff", "line-opacity": 0, "line-width": .5 } }, "labels");
    map.addLayer({ id: "sel-line", type: "line", source: "sel", paint: { "line-color": "#ffd36b", "line-width": 3 } });
    map.on("click", "tiles-fill", (e) => { if (!S.drawing && e.features?.[0]) selectTile(e.features[0].properties.i, false); });
    map.on("mouseenter", "tiles-fill", () => { if (!S.drawing) map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", "tiles-fill", () => { if (!S.drawing) map.getCanvas().style.cursor = ""; });
    initDraw();
    initLayerSwitch();
    // open with Upper Arlington ready to go
    if (ua) { setBox(ua.bbox, ua.name, { fly: false }); if (ua.cached) setTimeout(run, 400); }
  });
}
const center = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];

function setBox(bbox, name, { fly = true } = {}) {
  S.box = bbox.map((v) => +(+v).toFixed(5)); S.boxName = name || null;
  const [w, s, e, n] = S.box;
  S.map?.getSource("box")?.setData({ type: "Feature", geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } });
  if (fly && S.map) S.map.fitBounds([[w, s], [e, n]], { padding: { top: 70, bottom: 90, left: 40, right: 200 }, duration: 1600, essential: true });
  estimate();
}

async function estimate() {
  const info = $("#est-info"), btn = $("#run-btn");
  if (!S.box) return;
  try {
    const r = await api("/api/region/estimate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ bbox: S.box }) });
    S.estimateOk = r.ok;
    info.classList.toggle("bad", !r.ok);
    const cached = presetFor(S.box)?.cached;
    info.innerHTML = r.ok
      ? `${esc(S.boxName || "Custom area")} · <b>${r.n_tiles} squares</b> · ${r.width_km.toFixed(1)} × ${r.height_km.toFixed(1)} km · ${cached ? "<b>saved, opens instantly</b>" : `about ${Math.max(10, Math.round(r.eta_s || 30))} s`}`
      : `<b>${esc(r.message || "Too large")}</b>`;
    btn.disabled = !r.ok || S.busy;
  } catch (e) { info.textContent = e.message; btn.disabled = true; }
}
const presetFor = (b) => (S.meta?.presets || []).find((p) => p.bbox.every((v, i) => Math.abs(v - b[i]) < 1e-4));

// ------------------------------------------------------------------ drawing a box (mouse + touch)
function initDraw() {
  const map = S.map, btn = $("#draw-btn"), hint = $("#map-hint"), canvas = map.getCanvasContainer();
  let start = null;
  const set = (on) => {
    S.drawing = on; btn.classList.toggle("on", on); hint.classList.toggle("off", !on);
    canvas.style.cursor = on ? "crosshair" : "";
    on ? map.dragPan.disable() : map.dragPan.enable();
  };
  btn.addEventListener("click", () => set(!S.drawing));
  addEventListener("keydown", (e) => { if (e.key === "Escape" && S.drawing) { start = null; set(false); } });
  const ll = (e) => { const r = canvas.getBoundingClientRect(); return map.unproject([e.clientX - r.left, e.clientY - r.top]); };
  canvas.addEventListener("pointerdown", (e) => { if (!S.drawing) return; start = ll(e); canvas.setPointerCapture(e.pointerId); e.preventDefault(); });
  canvas.addEventListener("pointermove", (e) => {
    if (!S.drawing || !start) return;
    const p = ll(e);
    const w = Math.min(start.lng, p.lng), east = Math.max(start.lng, p.lng), s = Math.min(start.lat, p.lat), n = Math.max(start.lat, p.lat);
    map.getSource("box").setData({ type: "Feature", geometry: { type: "Polygon", coordinates: [[[w, s], [east, s], [east, n], [w, n], [w, s]]] } });
  });
  canvas.addEventListener("pointerup", (e) => {
    if (!S.drawing || !start) return;
    const p = ll(e), a = start; start = null; set(false);
    if (Math.abs(a.lng - p.lng) < 1e-3 || Math.abs(a.lat - p.lat) < 1e-3) { toast("That box is too small. Drag a larger area."); return; }
    const c = [(a.lng + p.lng) / 2, (a.lat + p.lat) / 2];
    setBox([Math.min(a.lng, p.lng), Math.min(a.lat, p.lat), Math.max(a.lng, p.lng), Math.max(a.lat, p.lat)], `Area near ${c[1].toFixed(2)}°, ${c[0].toFixed(2)}°`, { fly: false });
  });
  $("#run-btn").addEventListener("click", run);
}

// ------------------------------------------------------------------ presets, year, geocoder
function initPresets() {
  const el = $("#presets");
  const ps = S.meta?.presets || [];
  el.innerHTML = ps.map((p) => `<button class="chip" data-key="${esc(p.key)}" title="${esc(p.note || "")}">${p.cached ? "● " : ""}${esc(p.name)}</button>`).join("") || `<span class="muted">Presets load from the API.</span>`;
  $$("button", el).forEach((b) => b.addEventListener("click", () => {
    const p = ps.find((x) => x.key === b.dataset.key);
    $$("button", el).forEach((x) => x.classList.toggle("on", x === b));
    setBox(p.bbox, p.name);
    if (p.cached && !S.when) run();
  }));
}
function initWhen() {
  const sel = $("#when-sel"); const y = new Date().getUTCFullYear();
  for (let k = y; k >= 2019; k--) sel.insertAdjacentHTML("beforeend", `<option value="${k}">Summer ${k}</option>`);
  sel.addEventListener("change", () => { S.when = sel.value; estimate(); });
}
function initGeocode() {
  const input = $("#geo-q"), box = $("#geo-results"); let t = null, items = [], sel = -1;
  const close = () => { box.classList.remove("on"); sel = -1; };
  const pick = (it) => {
    close(); input.value = it.name.split(",")[0];
    setBox(fitBox(it), it.name.split(",").slice(0, 2).join(","));
  };
  input.addEventListener("input", () => {
    clearTimeout(t); const q = input.value.trim(); if (q.length < 3) { close(); return; }
    t = setTimeout(async () => {
      try { items = await api(`/api/geocode?q=${encodeURIComponent(q)}`); } catch (e) { items = []; }
      box.innerHTML = items.length ? items.map((it, i) => `<button data-i="${i}" role="option"><span>${esc(it.name.split(",")[0])}</span><small>${esc(it.name)}</small></button>`).join("") : `<button disabled><small>No places found (offline?)</small></button>`;
      box.classList.add("on");
      $$("button[data-i]", box).forEach((b) => b.addEventListener("click", () => pick(items[+b.dataset.i])));
    }, 280);
  });
  input.addEventListener("keydown", (e) => {
    const bs = $$("button[data-i]", box);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); sel = clamp(sel + (e.key === "ArrowDown" ? 1 : -1), 0, bs.length - 1); bs.forEach((b, i) => b.classList.toggle("sel", i === sel)); }
    if (e.key === "Enter" && items.length) { e.preventDefault(); pick(items[Math.max(sel, 0)]); }
    if (e.key === "Escape") close();
  });
  input.addEventListener("blur", () => setTimeout(close, 180));
}
// a study box: the place's own extent, clipped to ~11 km a side so it stays under the tile limit
function fitBox(it) {
  const half = 5.6, minHalf = 2.2; // km
  const kx = 111.32 * Math.cos(it.lat * Math.PI / 180), ky = 110.57;
  let [w, s, e, n] = it.bbox || [it.lon - .03, it.lat - .03, it.lon + .03, it.lat + .03];
  const hx = clamp((e - w) * kx / 2, minHalf, half), hy = clamp((n - s) * ky / 2, minHalf, half);
  return [it.lon - hx / kx, it.lat - hy / ky, it.lon + hx / kx, it.lat + hy / ky];
}

// ------------------------------------------------------------------ running an analysis
async function run() {
  if (!S.box || S.busy) return;
  S.busy = true; $("#run-btn").disabled = true;
  switchTab("start");
  const stages = $("#stages"); stages.innerHTML = ""; $("#run-error").innerHTML = "";
  const t0 = performance.now();
  let live = null;
  const add = (st) => {
    if (live) live.classList.remove("live");
    live = document.createElement("li"); live.className = "live";
    live.innerHTML = `<span>${esc(st.detail)}</span><span class="t">${((performance.now() - t0) / 1000).toFixed(1)}s</span>`;
    stages.append(live);
  };
  try {
    add({ detail: "Sending the study area" });
    const { job } = await api("/api/region", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ bbox: S.box, when: S.when || null }) });
    const region = await pollJob(job, add);
    S.region = region; S.climate = null; S.match = null; S.selected = null;
    loadRegion(region);
    add({ detail: "Measuring greenness and surface temperature" });
    const cj = await api(`/api/region/${region.id}/climate`, { method: "POST" });
    S.climate = await pollJob(cj.job, add);
    loadClimate();
    if (live) live.classList.remove("live");
    stages.insertAdjacentHTML("beforeend", `<li><span><b>Done</b> in ${((performance.now() - t0) / 1000).toFixed(1)} s. See the Heat tab.</span><span></span></li>`);
    saveCompare();
    setTimeout(() => switchTab("heat"), 700);
  } catch (e) {
    if (live) { live.classList.remove("live"); live.classList.add("err"); }
    $("#run-error").innerHTML = `<div class="errbox">${esc(e.message)}</div>`;
  } finally {
    S.busy = false; estimate();
  }
}

function tileFeatures() {
  const r = S.region; if (!r) return [];
  const c = S.climate || {};
  const ndvi = c.ndvi?.tile || [], lst = c.lst?.tile || [];
  const pri = new Map((c.stats?.priority || []).map((p, k) => [p.i, k + 1]));
  const ref = new Map((c.stats?.refuges || []).map((p, k) => [p.i, k + 1]));
  let mmin = 0, mmax = 1;
  if (S.match) { mmin = S.match.min; mmax = S.match.max; }
  return r.tiles.map((t) => {
    const [tl, tr, br, bl] = t.corners;
    const m = S.match ? (S.match.scores[t.i] - mmin) / ((mmax - mmin) || 1) : 0;
    return {
      type: "Feature", id: t.i,
      geometry: { type: "Polygon", coordinates: [[tl, bl, br, tr, tl]] },
      properties: { i: t.i, cls: t.cls, color: S.classes[t.cls]?.color || "#888", valid: t.valid !== false,
        ndvi: ndvi[t.i] ?? null, lst: lst[t.i] ?? null, m: S.match && S.match.scores[t.i] != null ? m : -1,
        pri: pri.get(t.i) || 0, ref: ref.get(t.i) || 0 },
    };
  });
}
function refreshTiles() { S.map?.getSource("tiles")?.setData({ type: "FeatureCollection", features: tileFeatures() }); applyMode(); }

function loadRegion(r) {
  const map = S.map; if (!map) return;
  ["rgb", "ndvi", "lst"].forEach((k) => { if (map.getLayer(`r-${k}`)) map.removeLayer(`r-${k}`); if (map.getSource(`r-${k}`)) map.removeSource(`r-${k}`); });
  map.addSource("r-rgb", { type: "image", url: r.image, coordinates: r.corners });
  map.addLayer({ id: "r-rgb", type: "raster", source: "r-rgb", paint: { "raster-opacity": 1, "raster-fade-duration": 300 } }, "tiles-fill");
  map.getSource("box").setData({ type: "FeatureCollection", features: [] });
  const lons = r.corners.map((c) => c[0]), lats = r.corners.map((c) => c[1]);
  map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], { padding: { top: 70, bottom: 90, left: 30, right: 200 }, duration: 1400 });
  const name = S.boxName || "Study area";
  $("#panel-title").textContent = name; $("#panel-title").title = name;
  $("#provenance").innerHTML = `${esc(r.scene.platform || "Sentinel-2")} · ${niceDate(r.scene.datetime)} · ${fmt(r.scene.cloud_cover, 1)}% cloud<br>${r.rows}×${r.cols} squares · ${fmt(r.area_km2, 1)} km² · ${r.mode}${r.from_cache ? " · saved pass" : ""}`;
  $$("#layer-switch button").forEach((b) => (b.disabled = !["rgb", "cls"].includes(b.dataset.layer)));
  refreshTiles(); setMode("cls");
  renderCover(); renderExport(); $("#ask-results").innerHTML = ""; $("#inspector").innerHTML = "";
}

function loadClimate() {
  const map = S.map, r = S.region, c = S.climate; if (!map || !r || !c) return;
  if (c.ndvi?.image) { map.addSource("r-ndvi", { type: "image", url: c.ndvi.image, coordinates: r.corners }); map.addLayer({ id: "r-ndvi", type: "raster", source: "r-ndvi", paint: { "raster-opacity": 0 } }, "tiles-fill"); }
  if (c.lst?.image) { map.addSource("r-lst", { type: "image", url: c.lst.image, coordinates: r.corners }); map.addLayer({ id: "r-lst", type: "raster", source: "r-lst", paint: { "raster-opacity": 0, "raster-resampling": "nearest" } }, "tiles-fill"); }
  $$("#layer-switch button").forEach((b) => {
    const k = b.dataset.layer;
    b.disabled = (k === "ndvi" && !c.ndvi?.image) || (k === "lst" && !c.lst?.image) || (k === "priority" && !(c.stats?.priority || []).length) || (k === "match" && !S.match);
  });
  refreshTiles(); renderHeat(); renderExport();
  setMode(c.lst?.image ? "lst" : "ndvi");
}

// ------------------------------------------------------------------ layers
function initLayerSwitch() {
  $$("#layer-switch button").forEach((b) => b.addEventListener("click", () => !b.disabled && setMode(b.dataset.layer)));
  $("#layer-op").addEventListener("input", (e) => { S.opacity = e.target.value / 100; applyMode(); });
}
function setMode(m) { S.mode = m; $$("#layer-switch button").forEach((b) => b.classList.toggle("on", b.dataset.layer === m)); applyMode(); }
function applyMode() {
  const map = S.map; if (!map || !map.getLayer("tiles-fill")) return;
  const m = S.mode, op = S.opacity;
  const set = (id, k, v) => map.getLayer(id) && map.setPaintProperty(id, k, v);
  set("r-ndvi", "raster-opacity", m === "ndvi" ? op : 0);
  set("r-lst", "raster-opacity", m === "lst" ? op : 0);
  set("r-rgb", "raster-opacity", 1);
  const fills = {
    rgb: ["#000", 0],
    cls: [["get", "color"], ["case", ["get", "valid"], op * .78, 0]],
    ndvi: ["#000", 0], lst: ["#000", 0],
    priority: [["case", [">", ["get", "pri"], 0], "#ff5a36", [">", ["get", "ref"], 0], "#3ddc84", "#000"], ["case", [">", ["get", "pri"], 0], op * .85, [">", ["get", "ref"], 0], op * .75, 0]],
    match: [["interpolate", ["linear"], ["get", "m"], 0, "#0b3d1e", .5, "#4f9a3a", .8, "#a6dc4a", 1, "#e4ff6b"],
      ["case", ["<", ["get", "m"], 0], 0, ["interpolate", ["linear"], ["get", "m"], 0, 0, .45, op * .25, 1, op * .92]]],
  }[m];
  set("tiles-fill", "fill-color", fills[0]); set("tiles-fill", "fill-opacity", fills[1]);
  set("tiles-line", "line-opacity", S.region ? (m === "rgb" ? .35 : .18) : 0);
  legend();
}
function rampCss(stops) { return `linear-gradient(90deg, ${stops.map(([p, c]) => `${c} ${(p * 100).toFixed(0)}%`).join(",")})`; }
function legend() {
  const el = $("#map-legend"), c = S.climate, m = S.mode;
  let h = "";
  if (m === "ndvi" && c?.ndvi) h = `Greenness (NDVI) · measured<div class="ramp" style="background:${rampCss(c.ndvi.palette || [])}"></div><div class="ends"><span>${fmt(c.ndvi.vmin, 1)} bare</span><span>${fmt(c.ndvi.vmax, 1)} dense canopy</span></div>`;
  if (m === "lst" && c?.lst) h = `Land surface temp · measured<div class="ramp" style="background:${rampCss(c.lst.palette || [])}"></div><div class="ends"><span>${fmt(c.lst.vmin, 0)} °C</span><span>${fmt(c.lst.vmax, 0)} °C</span></div><div style="margin-top:4px;opacity:.7">Landsat · ${niceDate(c.lst.scene?.datetime)}</div>`;
  if (m === "cls" && S.region) {
    const present = new Set(S.region.tiles.map((t) => t.cls));
    h = `Land cover · model estimate<div class="cls">${(S.meta?.classes || []).filter((k) => present.has(k.name)).map((k) => `<span><i style="background:${k.color}"></i>${esc(k.label)}</span>`).join("")}</div>`;
  }
  if (m === "priority") h = `<div class="cls" style="grid-template-columns:1fr"><span><i style="background:#ff5a36"></i>Plant first: hottest quarter, least green</span><span><i style="background:#3ddc84"></i>Cool refuges: green and cool</span></div>`;
  if (m === "match" && S.match) h = `Match: “${esc(S.matchLabel)}”<div class="ramp" style="background:linear-gradient(90deg,#0b3d1e,#4f9a3a,#a6dc4a,#e4ff6b)"></div><div class="ends"><span>weak</span><span>strong (cosine)</span></div>`;
  el.innerHTML = h; el.classList.toggle("on", !!h);
}

// ------------------------------------------------------------------ tabs
function initTabs() { $$(".tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab))); }
function switchTab(k) {
  $$(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === k));
  $$(".tabpane").forEach((p) => p.classList.toggle("on", p.dataset.pane === k));
  if (k === "heat") requestAnimationFrame(drawHeatScatter);
}

function kpi(k, v, unit, s, cls = "") { return `<div class="kpi ${cls}"><div class="k">${k}</div><div class="v">${v}${unit ? `<small>${unit}</small>` : ""}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`; }
const signed = (v, d = 1) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`);

function renderHeat() {
  const c = S.climate, st = c?.stats || {}, pane = $('[data-pane="heat"]');
  if (!c) { pane.innerHTML = `<div class="empty">Analyse an area to measure heat and greenness.</div>`; return; }
  const reg = st.regression, ter = st.tercile;
  const lstNote = c.lst ? `${esc((c.lst.scene?.platform || "Landsat").replace("landsat-", "Landsat "))} · ${niceDate(c.lst.scene?.datetime)} · ${c.lst.days_from_s2 === 0 ? "same day as" : `${Math.abs(c.lst.days_from_s2)} d ${c.lst.days_from_s2 < 0 ? "before" : "after"}`} Sentinel-2` : "";
  pane.innerHTML = `
    ${c.lst_error ? `<div class="errbox" style="margin:0 0 12px">${esc(c.lst_error)} Greenness is still shown.</div>` : ""}
    <p class="explain" style="margin:0 0 12px"><b>How to read this:</b> a negative first number means greener squares are cooler. On the map, pick “Plant first”: orange = hot with few trees.</p><div class="sec-title" style="margin-top:0">Key numbers <span class="tag measured">${term("measured", "measured")}</span></div>
    <div class="kpis">
      ${kpi(term("slope", "Cooling per +0.1 NDVI"), signed(reg?.slope_per_0p1_ndvi, 2), "°C", reg ? `OLS on ${reg.n} land squares · r = ${fmt(reg.r, 2)}` : "Needs at least 8 land squares with both layers", "cool")}
      ${kpi(term("tercile", "Least-green minus greenest third"), signed(ter?.hot_minus_green_c, 1), "°C", ter ? `${fmt(ter.least_green_mean, 1)} vs ${fmt(ter.greenest_mean, 1)} °C · ${ter.n_each} squares each` : "—", "hot")}
      ${kpi(term("lst", "Mean surface temp"), fmt(st.lst_mean, 1), "°C", st.lst_min != null ? `range ${fmt(st.lst_min, 1)} – ${fmt(st.lst_max, 1)} °C` : "no thermal pass")}
      ${kpi(term("ndvi", "Mean greenness"), fmt(st.ndvi_mean, 2), "NDVI", `${st.n_land ?? "—"} land · ${st.n_water ?? 0} water squares`)}
      ${st.built_minus_vegetated_c != null ? kpi("Built minus vegetated", signed(st.built_minus_vegetated_c, 1), "°C", "by the model's land cover (mixed)", "span2") : ""}
    </div>
    <div class="sec-title">Greenness vs temperature <span class="mono" style="letter-spacing:0;text-transform:none">click a dot</span></div>
    <canvas class="chart" id="lab-scatter" style="height:260px" aria-label="Scatter plot of NDVI against surface temperature"></canvas>
    <p class="explain">${lstNote}. Each dot is one 640 m land square; water squares are left out. Descriptive, not causal.</p>
    ${(st.by_group || []).length ? `<div class="sec-title">By surface type <span class="tag model">model groups</span></div>
      <div class="rows">${st.by_group.map((g) => `<div class="r"><i style="background:${{ Built: "#b4483a", Vegetated: "#3f8a45", Cropland: "#d39b35", Water: "#2c6d8c" }[g.group] || "#888"}"></i><span>${esc(g.group)} <span class="muted">· ${g.n}</span></span><span class="num">${fmt(g.ndvi_mean, 2)} NDVI</span><span class="num">${fmt(g.lst_mean, 1)} °C</span></div>`).join("")}</div>` : ""}
    <div class="sec-title">${term("priority", "Plant first")} · hot &amp; bare <span class="tag measured">ranked</span></div>
    <div class="list">${(st.priority || []).map((p, k) => tileLi(p, k, "hot")).join("") || `<div class="muted" style="font-size:13px">Not enough squares to rank.</div>`}</div>
    <p class="explain">Squares in the hottest quarter, ranked by how hot and how un-green they are (z-score of temperature minus z-score of NDVI). Start field visits here.</p>
    <div class="sec-title">${term("refuge", "Cool refuges")} · green &amp; cool</div>
    <div class="list">${(st.refuges || []).map((p, k) => tileLi(p, k, "cool")).join("")}</div>`;
  $$(".li[data-i]", pane).forEach((el) => el.addEventListener("click", () => selectTile(+el.dataset.i, true)));
  const cv = $("#lab-scatter");
  cv.addEventListener("click", (e) => {
    const pts = scatterPts(); if (!S._scale || !pts.length) return;
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 1e9;
    pts.forEach(([nx, ly, i]) => { const d = (S._scale.X(nx) - x) ** 2 + (S._scale.Y(ly) - y) ** 2; if (d < bd) { bd = d; best = i; } });
    if (best != null && bd < 400) selectTile(best, true);
  });
  drawHeatScatter();
}
function tileLi(p, k, kind) {
  const label = S.classes[p.cls]?.label || p.cls;
  return `<div class="li" data-i="${p.i}"><img src="/api/region/${S.region.id}/tile/${p.i}.png" alt=""><div><div class="t">#${k + 1} · ${esc(label)}</div><div class="d">NDVI ${fmt(p.ndvi, 2)} · ${p.center ? `${p.center[1].toFixed(4)}, ${p.center[0].toFixed(4)}` : ""}</div></div><span class="badge ${kind}">${fmt(p.lst, 1)} °C</span></div>`;
}
function scatterPts() {
  const c = S.climate; if (!c?.ndvi || !c?.lst) return [];
  const water = c.water?.tile || [];
  return c.ndvi.tile.map((n, i) => [n, c.lst.tile[i], i]).filter(([n, l, i]) => n != null && l != null && !(water[i] > .5) && S.region.tiles[i]?.valid !== false);
}
function drawHeatScatter() {
  const cv = $("#lab-scatter"); if (!cv || !cv.clientWidth) return;
  const pts = scatterPts();
  if (!pts.length) { const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, cv.width, cv.height); return; }
  S._scale = drawScatter(cv, pts, S.climate.stats?.regression, S.meta, { highlight: S.selected });
}

function renderCover() {
  const r = S.region, pane = $('[data-pane="cover"]');
  const comp = r.composition || [];
  pane.innerHTML = `
    <div class="sec-title" style="margin-top:0">Land cover <span class="tag model">${term("model", "model estimate")}</span></div>
    <div class="stack">${comp.map((c) => `<span style="flex-grow:${c.share};background:${S.classes[c.cls]?.color}" title="${esc(S.classes[c.cls]?.label)} ${pct(c.share)}"></span>`).join("")}</div>
    <div class="rows">${comp.map((c) => `<div class="r"><i style="background:${S.classes[c.cls]?.color}"></i><span>${esc(S.classes[c.cls]?.label || c.cls)}</span><span class="num">${fmt(c.km2, 1)} km²</span><span class="num">${pct(c.share)}</span></div>`).join("")}</div>
    <div class="sec-title">IPCC land-use categories <span class="tag model">roll-up</span></div>
    <div class="rows">${(r.ipcc || []).map((c) => `<div class="r"><i style="background:var(--muted)"></i><span>${esc(c.category)}</span><span class="num">${fmt(c.km2, 1)} km²</span><span class="num">${pct(c.share)}</span></div>`).join("")}</div>
    <p class="explain">Each 640 m square gets the class whose caption sits closest after the rotation. The categories follow the IPCC 2006 Guidelines (Vol. 4), as a quick inventory-style summary. It is a coarse screen: in Upper Arlington the model is right on 64% of hand-labelled squares, and mixed suburban squares collapse to “Residential”.</p>`;
}

// ------------------------------------------------------------------ ask: text search + inspector + similar
function initAsk() {
  const chips = ["tree canopy", "parking lots and big roofs", "sports fields and lawns", "river with trees", "highway interchange", "farm fields"];
  $("#ask-chips").innerHTML = chips.map((q) => `<button class="chip" type="button">${q}</button>`).join("");
  $$("#ask-chips .chip").forEach((b) => b.addEventListener("click", () => { $("#ask-q").value = b.textContent; ask(); }));
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
}
async function ask() {
  const q = $("#ask-q").value.trim();
  if (!S.region) { toast("Analyse an area first."); return; }
  if (!q) return;
  try {
    const r = await api(`/api/region/${S.region.id}/search?q=${encodeURIComponent(q)}&k=8`);
    showMatch(r, `“${q}”`);
  } catch (e) { $("#ask-results").innerHTML = `<div class="errbox">${esc(e.message)}</div>`; }
}
function showMatch(r, label) {
  S.match = r; S.matchLabel = label.replace(/[“”]/g, "");
  $('#layer-switch button[data-layer="match"]').disabled = false;
  refreshTiles(); setMode("match");
  $("#ask-results").innerHTML = `<div class="sec-title">Best matches for ${esc(label)} <span class="tag model">${term("cosine", "similarity")}</span></div>
    <div class="list">${r.top.map((t, k) => { const tile = S.region.tiles[t.i]; return `<div class="li" data-i="${t.i}"><img src="/api/region/${S.region.id}/tile/${t.i}.png" alt=""><div><div class="t">#${k + 1} · ${esc(S.classes[tile.cls]?.label || tile.cls)}</div><div class="d">${tileClimate(t.i)}</div></div><span class="badge cool">${t.score.toFixed(3)}</span></div>`; }).join("")}</div>`;
  $$("#ask-results .li").forEach((el) => el.addEventListener("click", () => selectTile(+el.dataset.i, true)));
}
function tileClimate(i) {
  const c = S.climate; if (!c) return "";
  const n = c.ndvi?.tile?.[i], l = c.lst?.tile?.[i];
  return [n != null ? `NDVI ${fmt(n, 2)}` : null, l != null ? `${fmt(l, 1)} °C` : null].filter(Boolean).join(" · ");
}
function selectTile(i, fly) {
  const r = S.region; if (!r) return;
  const t = r.tiles[i]; if (!t) return;
  S.selected = i;
  const [tl, tr, br, bl] = t.corners;
  S.map.getSource("sel").setData({ type: "Feature", geometry: { type: "LineString", coordinates: [tl, tr, br, bl, tl] } });
  if (fly) S.map.easeTo({ center: t.center, zoom: Math.max(S.map.getZoom(), 13.4), duration: 900 });
  const c = S.climate || {};
  const w = c.water?.tile?.[i];
  $("#inspector").innerHTML = `<div class="inspector">
    <img src="/api/region/${r.id}/tile/${i}.png" alt="Selected square">
    <dl>
      <dt>square</dt><dd>#${i} · row ${t.row}, col ${t.col}</dd>
      <dt>model</dt><dd>${(t.top3 || []).map((x, k) => `${k ? "" : "<b>"}${esc(S.classes[x.label]?.label || x.label)} ${x.score.toFixed(2)}${k ? "" : "</b>"}`).join(" · ")}</dd>
      <dt>NDVI</dt><dd>${fmt(c.ndvi?.tile?.[i], 2)}</dd>
      <dt>surface</dt><dd>${c.lst?.tile?.[i] != null ? `${fmt(c.lst.tile[i], 1)} °C` : "—"}</dd>
      <dt>water</dt><dd>${w != null ? pct(w) : "—"}</dd>
      <dt>centre</dt><dd class="mono">${t.center[1].toFixed(5)}, ${t.center[0].toFixed(5)}</dd>
    </dl>
    <div class="actions"><button class="btn small" id="sim-btn">Find squares like this</button><button class="btn small ghost" id="copy-ll">Copy coordinates</button></div>
  </div>`;
  $("#sim-btn").addEventListener("click", async () => {
    try { const s = await api(`/api/region/${r.id}/similar?i=${i}&k=8`); showMatch(s, `like square #${i}`); switchTab("ask"); }
    catch (e) { toast(e.message); }
  });
  $("#copy-ll").addEventListener("click", () => { navigator.clipboard?.writeText(`${t.center[1].toFixed(5)}, ${t.center[0].toFixed(5)}`); toast("Coordinates copied"); });
  if (!$('[data-pane="heat"]').classList.contains("on")) switchTab("ask");
  drawHeatScatter();
}

// ------------------------------------------------------------------ compare across cities
function loadCompare() { try { return JSON.parse(localStorage.getItem(STORE) || "[]"); } catch (e) { return []; } }
function saveCompare() {
  const r = S.region, st = S.climate?.stats || {};
  if (!r) return;
  const row = {
    id: r.id, name: S.boxName || "Study area", s2: r.scene.datetime, landsat: S.climate?.lst?.scene?.datetime || null,
    n: r.n_tiles, lst_mean: st.lst_mean, ndvi_mean: st.ndvi_mean, slope: st.regression?.slope_per_0p1_ndvi ?? null, r: st.regression?.r ?? null,
    tercile: st.tercile?.hot_minus_green_c ?? null, settle: (r.ipcc || []).find((x) => x.category === "Settlements")?.share ?? null, mode: r.mode,
  };
  const rows = loadCompare().filter((x) => x.id !== row.id); rows.unshift(row);
  try { localStorage.setItem(STORE, JSON.stringify(rows.slice(0, 30))); } catch (e) { /* storage blocked */ }
  renderCompare();
}
function renderCompare() {
  const rows = loadCompare(), el = $("#compare");
  if (!rows.length) { el.innerHTML = `<div class="empty">Analyse two or more areas to compare them.</div>`; return; }
  const sl = rows.filter((r) => r.slope != null);
  const lo = Math.min(-3, ...sl.map((r) => r.slope)), hi = Math.max(.5, ...sl.map((r) => r.slope));
  const X = (v) => 8 + (v - lo) / (hi - lo) * 284;
  el.innerHTML = `
    <div class="sec-title" style="margin-top:6px">Cooling per +0.1 NDVI, by area</div>
    <svg class="chart" viewBox="0 0 300 ${20 + sl.length * 34}" style="height:auto">
      <line class="axis" x1="${X(0)}" x2="${X(0)}" y1="6" y2="${14 + sl.length * 34}"/>
      <text x="${X(0) - 4}" y="14" text-anchor="end">0 °C</text>
      ${sl.map((r, k) => `<g><text x="8" y="${30 + k * 34}" style="fill:var(--ink)">${esc(r.name.slice(0, 30))} · ${signed(r.slope, 2)} °C</text>
        <line x1="${X(0)}" x2="${X(r.slope)}" y1="${40 + k * 34}" y2="${40 + k * 34}" stroke-width="5" stroke-linecap="round" style="stroke:${r.id === S.region?.id ? "var(--moss)" : "var(--line-2)"}"/>
        <circle cx="${X(r.slope)}" cy="${40 + k * 34}" r="4.5" fill="${heatColor(clamp(-r.slope / 3))}"/></g>`).join("")}
    </svg>
    <table class="compare-table">
      <thead><tr><th>Area</th><th class="n">°C/0.1</th><th class="n">r</th><th class="n">Δ terc.</th><th class="n">mean °C</th><th></th></tr></thead>
      <tbody>${rows.map((r) => `<tr class="${r.id === S.region?.id ? "cur" : ""}"><td>${esc(r.name)}<div class="muted" style="font-size:11px">${niceDate(r.s2)} · ${r.n} sq</div></td>
        <td class="n">${signed(r.slope, 2)}</td><td class="n">${fmt(r.r, 2)}</td><td class="n">${signed(r.tercile, 1)}</td><td class="n">${fmt(r.lst_mean, 1)}</td>
        <td><button data-del="${r.id}" aria-label="Remove ${esc(r.name)}">×</button></td></tr>`).join("")}</tbody>
    </table>
    <div class="btn-row"><button class="btn small ghost" id="cmp-csv">Download comparison CSV</button><button class="btn small ghost" id="cmp-clear">Clear</button></div>
    <p class="explain">Slopes from different dates, seasons and Landsat passes are not strictly comparable. Note the dates before drawing conclusions.</p>`;
  $$("[data-del]", el).forEach((b) => b.addEventListener("click", () => { localStorage.setItem(STORE, JSON.stringify(loadCompare().filter((x) => x.id !== b.dataset.del))); renderCompare(); }));
  $("#cmp-clear").addEventListener("click", () => { localStorage.removeItem(STORE); renderCompare(); });
  $("#cmp-csv").addEventListener("click", () => {
    const cols = ["name", "id", "s2", "landsat", "n", "lst_mean", "ndvi_mean", "slope", "r", "tercile", "settle", "mode"];
    const csv = [cols.join(",")].concat(rows.map((r) => cols.map((c) => (r[c] == null ? "" : /[",\n]/.test(String(r[c])) ? `"${String(r[c]).replace(/"/g, '""')}"` : r[c])).join(","))).join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "platonic-earth-comparison.csv"; a.click();
  });
}

// ------------------------------------------------------------------ export + methods text
function renderExport() {
  const r = S.region, c = S.climate, pane = $('[data-pane="export"]');
  if (!r) return;
  const st = c?.stats || {};
  const cite = [
    `Study area: ${S.boxName || "custom"} — bbox ${r.bbox.map((v) => v.toFixed(4)).join(", ")} (W, S, E, N)`,
    `Grid: ${r.rows} × ${r.cols} squares of ${r.tile_m} m (${fmt(r.area_km2, 1)} km²), EPSG:${r.scene.epsg || "UTM"}`,
    `Optical: ${r.scene.id} (${r.scene.platform}, ${r.scene.datetime}, ${fmt(r.scene.cloud_cover, 2)}% cloud) — ${r.scene.source || "Copernicus Sentinel-2 L2A"}`,
    c?.lst ? `Thermal: ${c.lst.scene.id} (${c.lst.scene.platform}, ${c.lst.scene.datetime}; ${c.lst.days_from_s2} d from optical) — ${c.lst.scene.source}` : `Thermal: ${c?.lst_error || "not run"}`,
    `NDVI = (B08 − B04)/(B08 + B04) from the optical pass; SCL cloud/shadow masked. LST = Landsat C2 L2 ST_B10 in °C, resampled to 32 m.`,
    st.regression ? `Result: LST = ${st.regression.intercept} ${st.regression.slope_per_0p1_ndvi < 0 ? "−" : "+"} ${Math.abs(st.regression.slope_per_0p1_ndvi * 10).toFixed(1)}·NDVI °C (OLS, n = ${st.regression.n} land squares, r = ${st.regression.r}); least-green minus greenest tercile = ${st.tercile?.hot_minus_green_c} °C.` : "",
    `Land cover (estimate): frozen DINOv2 ViT-S/14 + all-MiniLM-L6-v2 aligned by one orthogonal Procrustes rotation fit on EuroSAT; label = best-matching of 50 class captions (cosine).`,
    `Tool: Platonic Earth (Veritas AI Hackathon 2026). ${S.meta?.disclaimer || ""}`,
  ].filter(Boolean).join("\n");
  pane.innerHTML = `
    <div class="sec-title" style="margin-top:0">Per-square data</div>
    <div class="btn-row" style="margin-top:0">
      <a class="btn small" href="/api/region/${r.id}/export.geojson" download>GeoJSON</a>
      <a class="btn small" href="/api/region/${r.id}/export.csv" download>CSV</a>
      <a class="btn small ghost" href="${r.image}" download>True-colour PNG</a>
      ${c?.ndvi?.image ? `<a class="btn small ghost" href="${c.ndvi.image}" download>NDVI PNG</a>` : ""}
      ${c?.lst?.image ? `<a class="btn small ghost" href="${c.lst.image}" download>LST PNG</a>` : ""}
    </div>
    <p class="explain">Columns: tile, row, col, lon, lat, class, score, second, third, IPCC category, surface group, ndvi, water_frac, lst_c, nodata/cloud fractions, and both scene IDs. Opens directly in QGIS, ArcGIS, pandas or R.</p>
    <div class="sec-title">Methods &amp; provenance <button class="btn small ghost" id="copy-cite" style="padding:5px 10px">Copy</button></div>
    <pre class="cite" id="cite">${esc(cite)}</pre>`;
  $("#copy-cite").addEventListener("click", () => { navigator.clipboard?.writeText(cite); toast("Methods text copied"); });
}
