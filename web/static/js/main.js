// Platonic Earth — story page (owner: D). Vanilla ES modules, no build step.
// Every number shown is read from /api/meta or /static/data/story.json; the HTML holds only fallbacks.
import { initLab } from "./lab.js";

export const $ = (s, el = document) => el.querySelector(s);
export const $$ = (s, el = document) => [...el.querySelectorAll(s)];
export const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const smooth = (a, b, v) => { const t = clamp((v - a) / (b - a)); return t * t * (3 - 2 * t); };
export const pct = (v, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const fmt = (v, d = 1) => (v == null || !isFinite(v) ? "—" : Number(v).toFixed(d));
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

export async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error((body && body.error) || `Request failed (${r.status})`);
  return body;
}

export function toast(msg, ms = 3200) {
  const t = $("#toast");
  t.textContent = msg; t.classList.add("on");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("on"), ms);
}

// --------------------------------------------------------------- chrome: theme, nav, progress, reveals
function initChrome() {
  $("#theme-btn").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("pe-theme", root.dataset.theme); } catch (e) { /* private mode */ }
  });
  const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), io.unobserve(e.target))), { rootMargin: "0px 0px -8% 0px" });
  $$(".reveal").forEach((el) => io.observe(el));

  // nav: dark over dark sections, hide on fast scroll down, active link
  const darkSecs = $$(".hero, .minds, .lab-sec, footer");
  const nav = $("#nav"); let lastY = scrollY;
  const links = $$(".nav a.link");
  addEventListener("scroll", () => {
    const y = scrollY;
    const h = document.documentElement.scrollHeight - innerHeight;
    $("#progress").style.transform = `scaleX(${h > 0 ? y / h : 0})`;
    const probe = 40;
    nav.classList.toggle("on-dark", darkSecs.some((s) => { const r = s.getBoundingClientRect(); return r.top <= probe && r.bottom > probe; }));
    nav.classList.toggle("hide", y > lastY + 4 && y > innerHeight * 1.2 && !nav.matches(":hover"));
    if (y < lastY - 2) nav.classList.remove("hide");
    lastY = y;
    let cur = null;
    for (const a of links) { const s = $(a.getAttribute("href")); if (s && s.getBoundingClientRect().top < innerHeight * .4) cur = a; }
    links.forEach((a) => a.classList.toggle("active", a === cur));
  }, { passive: true });
}

// manifesto: words light up as the paragraph scrolls through
function initManifesto() {
  const p = $("#manifesto");
  const walk = (node) => {
    for (const child of [...node.childNodes]) {
      if (child.nodeType === 3) {
        const frag = document.createDocumentFragment();
        child.textContent.split(/(\s+)/).forEach((w) => {
          if (!w.trim()) { frag.append(w); return; }
          const s = document.createElement("span"); s.className = "w"; s.textContent = w; frag.append(s);
        });
        child.replaceWith(frag);
      } else if (child.nodeType === 1) walk(child);
    }
  };
  walk(p);
  const words = $$(".w", p);
  const tick = () => {
    const r = p.getBoundingClientRect();
    const t = REDUCED ? 1 : clamp((innerHeight * .85 - r.top) / (r.height + innerHeight * .35));
    const n = Math.round(t * words.length);
    words.forEach((w, i) => w.classList.toggle("lit", i < n));
  };
  addEventListener("scroll", tick, { passive: true }); tick();
}

// --------------------------------------------------------------- scroll scenes
const scenes = [];
function sceneProgress(el) {
  const r = el.getBoundingClientRect();
  const span = r.height - innerHeight;
  return span > 0 ? clamp(-r.top / span) : 0;
}
function onScene(el, fn) { scenes.push({ el, fn }); }
function runScenes() {
  for (const s of scenes) {
    const r = s.el.getBoundingClientRect();
    if (r.bottom < -innerHeight || r.top > innerHeight * 2) continue;
    s.fn(sceneProgress(s.el));
  }
}
let ticking = false;
addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(() => { ticking = false; runScenes(); }); } }, { passive: true });
addEventListener("resize", () => requestAnimationFrame(runScenes));

function initSteps() {
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    const steps = e.target.parentElement;
    $$(".step", steps).forEach((s) => s.classList.toggle("is-active", s === e.target));
    e.target.dispatchEvent(new CustomEvent("step", { bubbles: true }));
  }), { rootMargin: "-45% 0px -45% 0px" });
  $$(".step").forEach((s) => io.observe(s));
}

// --------------------------------------------------------------- hero: scroll-driven globe flight
const UA = { lon: -83.0735, lat: 40.0275 };
function initHero(story) {
  const scene = $(".hero");
  const caps = $$(".hero-cap");
  const copy = $("#hero-copy"), hint = $("#scroll-hint"), readout = $("#coord-readout");
  const ua = story && story.ua;
  if (ua) {
    $("#cap-scene").textContent = `${ua.scene.platform.replace("sentinel-", "Sentinel-").toUpperCase().replace("SENTINEL", "Sentinel")} · ${niceDate(ua.scene.datetime)} · ${(ua.scene.cloud_cover ?? 0).toFixed(1)}% cloud`;
    $("#cap-tiles").textContent = `${ua.rows * ua.cols} squares of 640 m`;
  }
  let map = null, ready = false, spin = 0, p = 0;
  const hasGL = (() => { try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch (e) { return false; } })();
  if (!hasGL || !window.maplibregl) { document.body.classList.add("no-webgl"); }
  else {
    try {
      map = new maplibregl.Map({
        container: "globe", interactive: false, attributionControl: { compact: true },
        center: [-40, 22], zoom: innerWidth < 700 ? .55 : 1.35, fadeDuration: 0, renderWorldCopies: false,
        style: {
          version: 8, projection: { type: "globe" },
          sky: { "sky-color": "#0b140f", "horizon-color": "#2c4a36", "fog-color": "#0b140f", "atmosphere-blend": ["interpolate", ["linear"], ["zoom"], 0, 1, 5, 1, 8, 0] },
          sources: { esri: { type: "raster", tileSize: 256, maxzoom: 18, attribution: "Esri World Imagery",
            tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"] } },
          layers: [{ id: "bg", type: "background", paint: { "background-color": "#0b140f" } }, { id: "esri", type: "raster", source: "esri", paint: { "raster-saturation": .12, "raster-contrast": .08 } }],
        },
      });
      map.on("error", () => {});
      map.on("load", () => {
        ready = true;
        if (ua) {
          map.addSource("ua", { type: "image", url: "/static/img/ua-truecolor.jpg", coordinates: ua.corners });
          map.addLayer({ id: "ua", type: "raster", source: "ua", paint: { "raster-opacity": 0, "raster-fade-duration": 0 } });
          map.addSource("ua-grid", { type: "geojson", data: gridLines(ua.corners, ua.rows, ua.cols) });
          map.addLayer({ id: "ua-grid", type: "line", source: "ua-grid", paint: { "line-color": "#d9ec7c", "line-width": 1, "line-opacity": 0 } });
          map.addSource("ua-edge", { type: "geojson", data: { type: "Feature", geometry: { type: "LineString", coordinates: [...ua.corners, ua.corners[0]] } } });
          map.addLayer({ id: "ua-edge", type: "line", source: "ua-edge", paint: { "line-color": "#d9ec7c", "line-width": 2, "line-opacity": 0 } });
        }
        frame();
      });
    } catch (e) { document.body.classList.add("no-webgl"); map = null; }
  }

  function camera(p) {
    const tc = ease(smooth(.1, .5, p));         // center converges first
    const tz = smooth(.12, .7, p);                // then zoom
    const spinFade = 1 - smooth(.02, .2, p);
    const startLon = -40 + spin * spinFade;
    const lon = lerp(startLon, UA.lon, tc), lat = lerp(22, UA.lat, tc);
    const zoom = lerp(innerWidth < 700 ? .55 : 1.35, 11.55, Math.pow(tz, 1.15)) + smooth(.86, 1, p) * .35;
    const pitch = smooth(.84, 1, p) * 48, bearing = smooth(.84, 1, p) * -14;
    return { center: [lon, lat], zoom, pitch, bearing };
  }

  function frame() {
    if (map && ready) {
      const c = camera(p);
      map.jumpTo(c);
      if (map.getLayer("ua")) {
        map.setPaintProperty("ua", "raster-opacity", smooth(.66, .78, p));
        map.setPaintProperty("ua-grid", "line-opacity", smooth(.74, .84, p) * .55);
        map.setPaintProperty("ua-edge", "line-opacity", smooth(.6, .7, p));
      }
      const km = (40075 * Math.cos(c.center[1] * Math.PI / 180) / Math.pow(2, c.zoom + 9)) * innerWidth;
      readout.innerHTML = `LAT <b>${Math.abs(c.center[1]).toFixed(3)}° ${c.center[1] >= 0 ? "N" : "S"}</b><br>LON <b>${Math.abs(c.center[0]).toFixed(3)}° ${c.center[0] >= 0 ? "E" : "W"}</b><br>VIEW <b>${km > 2000 ? "planet" : `${km.toFixed(km < 50 ? 1 : 0)} km`}</b>`;
    }
    copy.style.opacity = 1 - smooth(.04, .13, p);
    copy.style.visibility = p > .14 ? "hidden" : "visible";
    copy.style.transform = `scale(${1 + smooth(0, .14, p) * .06}) translateY(${-smooth(0, .14, p) * 30}px)`;
    hint.style.opacity = 1 - smooth(0, .05, p);
    caps.forEach((c) => c.classList.toggle("on", p >= +c.dataset.from && p < +c.dataset.to));
  }
  onScene(scene, (v) => { p = v; frame(); });
  // idle spin while the globe is in view at the top
  let last = performance.now();
  const loop = (t) => {
    const dt = Math.min(64, t - last); last = t;
    if (!REDUCED && p < .2 && scrollY < innerHeight * 2) { spin -= dt * .0016; frame(); }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  frame();
}

function gridLines(c, rows, cols) {
  const [tl, tr, br, bl] = c;
  const at = (u, v) => { // bilinear inside the quad
    const top = [lerp(tl[0], tr[0], u), lerp(tl[1], tr[1], u)], bot = [lerp(bl[0], br[0], u), lerp(bl[1], br[1], u)];
    return [lerp(top[0], bot[0], v), lerp(top[1], bot[1], v)];
  };
  const feats = [];
  for (let i = 1; i < cols; i++) feats.push([at(i / cols, 0), at(i / cols, 1)]);
  for (let j = 1; j < rows; j++) feats.push([at(0, j / rows), at(1, j / rows)]);
  return { type: "FeatureCollection", features: feats.map((coordinates) => ({ type: "Feature", geometry: { type: "LineString", coordinates } })) };
}

export function niceDate(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

// --------------------------------------------------------------- chapter 1: two minds
const FALLBACK_COLORS = ["#E3B23C", "#1E6B45", "#6DB36A", "#7B7F86", "#B4483A", "#B7D36B", "#B97A2B", "#E8957A", "#3E9BC7", "#1D5E8C"];
function initMinds(story, meta) {
  const emb = story && story.embedding;
  const cv = $("#minds-canvas");
  if (!emb) return;
  const ctx = cv.getContext("2d");
  const cmeta = Object.fromEntries((meta?.classes || []).map((c) => [c.name, c]));
  const colorOf = (i) => {
    const c = cmeta[emb.classes[i]]?.color || FALLBACK_COLORS[i % 10];
    return c === "#1E6B45" ? "#3fa36c" : c === "#1D5E8C" ? "#3c86c2" : c; // lift the darkest colours on night bg
  };
  const labelOf = (i) => cmeta[emb.classes[i]]?.label || emb.classes[i];
  $("#minds-legend").innerHTML = emb.classes.map((_, i) => `<span><i style="background:${colorOf(i)}"></i>${esc(labelOf(i))}</span>`).join("") +
    `<span style="color:#eef3e6"><i class="star" style="background:#fff"></i>caption</span>`;
  $$('[data-m="cos_before"]').forEach((e) => (e.textContent = emb.mean_cos_to_own_caption.before.toFixed(2)));
  $$('[data-m="cos_after"]').forEach((e) => (e.textContent = emb.mean_cos_to_own_caption.after.toFixed(2)));
  $$('[data-m="emb_n"]').forEach((e) => (e.textContent = emb.n_images));
  const cos = $("#minds-cos");
  // caption centroid per class (target positions for labels)
  const cent = emb.classes.map((_, k) => {
    const pts = emb.text.filter((_, j) => emb.text_class[j] === k);
    return [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];
  });
  // stable per-point jitter phase for the idle shimmer
  const phase = emb.before.map((_, i) => (i * 2.399) % (Math.PI * 2));
  let p = 0, W = 0, H = 0, dpr = 1;
  const resize = () => {
    dpr = Math.min(2, devicePixelRatio || 1);
    W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr;
  };
  resize(); addEventListener("resize", resize);
  function draw(time) {
    const t = ease(smooth(.3, .72, p));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const narrow = W < 900;
    const cx = narrow ? W / 2 : W * .38, cy = H * .54, R = Math.min(narrow ? W * .44 : W * .3, H * .38);
    // faint rings
    ctx.strokeStyle = "rgba(217,236,124,.07)"; ctx.lineWidth = 1;
    for (let r = 1; r <= 3; r++) { ctx.beginPath(); ctx.arc(cx, cy, R * r / 3 * 1.05, 0, Math.PI * 2); ctx.stroke(); }
    const theta = (1 - t) * 2.4; // the before cloud swings in under the rotation
    const cs = Math.cos(theta), sn = Math.sin(theta);
    for (let i = 0; i < emb.before.length; i++) {
      const b = emb.before[i], a = emb.after[i];
      const bx = b[0] * cs - b[1] * sn, by = b[0] * sn + b[1] * cs;
      const wob = REDUCED ? 0 : (1 - t) * .012 * Math.sin(time / 900 + phase[i]);
      const x = lerp(bx, a[0], t) + wob, y = lerp(by, a[1], t) - wob;
      ctx.fillStyle = colorOf(emb.image_class[i]);
      ctx.globalAlpha = .82;
      ctx.beginPath(); ctx.arc(cx + x * R, cy - y * R, 2.6, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    for (let j = 0; j < emb.text.length; j++) {
      const [x, y] = emb.text[j];
      star(ctx, cx + x * R, cy - y * R, 6.5, colorOf(emb.text_class[j]));
    }
    // class labels once aligned
    const la = smooth(.68, .85, p);
    if (la > 0) {
      ctx.font = `500 ${narrow ? 10 : 12}px "JetBrains Mono", monospace`;
      ctx.textAlign = "center";
      cent.forEach(([x, y], k) => {
        ctx.globalAlpha = la * .95;
        const px = cx + x * R, py = cy - y * R - 14;
        const label = labelOf(k);
        const w = ctx.measureText(label).width + 12;
        ctx.fillStyle = "rgba(11,20,15,.75)"; roundRect(ctx, px - w / 2, py - 11, w, 18, 6); ctx.fill();
        ctx.fillStyle = "#eef3e6"; ctx.fillText(label, px, py + 2);
      });
      ctx.globalAlpha = 1;
    }
    cos.textContent = (t < .5 ? emb.mean_cos_to_own_caption.before : emb.mean_cos_to_own_caption.after).toFixed(2);
    cos.style.color = t < .5 ? "#93a596" : "#d9ec7c";
  }
  onScene($('[data-scene="minds"]'), (v) => { p = v; });
  const loop = (time) => {
    const r = cv.getBoundingClientRect();
    if (r.bottom > 0 && r.top < innerHeight) draw(time);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
function star(ctx, x, y, r, color) {
  ctx.save(); ctx.translate(x, y); ctx.rotate(Math.PI / 4);
  ctx.fillStyle = color; ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let k = 0; k < 8; k++) { const rr = k % 2 ? r * .38 : r; const a = k * Math.PI / 4; ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr); }
  ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
}
export function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// --------------------------------------------------------------- chapter 2: evidence (bind metrics)
function bindMetrics(meta) {
  const m = meta?.metrics || {};
  const set = (k, v) => v != null && $$(`[data-m="${k}"]`).forEach((e) => {
    const small = e.querySelector("small");
    if (small) { e.firstChild.textContent = v; } else e.textContent = v;
  });
  const ret = m.retrieval || {}, cc = m.caption_classification || {}, loc = m.local || {};
  if (ret.procrustes) {
    set("search_top1", Math.round(ret.procrustes.top1 * 100));
    set("search_nq", ret.procrustes.n_captions);
    set("search_ni", ret.procrustes.n_images?.toLocaleString("en-US"));
  }
  if (cc.procrustes) set("cls_n", cc.procrustes.n_images?.toLocaleString("en-US"));
  if (ret.random_rotation) set("random_top1", Math.round(ret.random_rotation.top1 * 100));
  if (cc.procrustes) set("ours_acc", pct(cc.procrustes.accuracy));
  if (m.majority) set("maj_acc", pct(m.majority.accuracy));
  renderConfusion(meta);
  if (cc.procrustes && m.resnet18) set("gap", Math.round((m.resnet18.accuracy - cc.procrustes.accuracy) * 100));
  if (loc.procrustes) set("local_acc", Math.round(loc.procrustes.accuracy * 100));
  if (loc.oracle_majority_local) set("local_oracle", pct(loc.oracle_majority_local.accuracy));
  const river = (m.local_by_category || []).find((r) => r.local_category === "water_river");
  if (river) set("river_local", `${Math.round(river.accuracy * river.n)} of ${river.n}`);
  const rh = (m.top_confusions || []).find((r) => r.true === "River" && r.predicted === "Highway");
  if (rh) set("rh_share", pct(rh.share_of_true));
  const g = Object.fromEntries((meta?.galleries || []).map((x) => [x.key, x.n]));
  if (g.local) set("n_local", g.local);
  if (g.eurosat) set("n_eurosat", g.eurosat.toLocaleString("en-US"));
  if (meta?.limits?.max_tiles) set("max_tiles", meta.limits.max_tiles);
  if (meta?.disclaimer) { $("#foot-disclaimer").textContent = meta.disclaimer; }
  $("#foot-mode").textContent = meta ? `${meta.mode} mode${meta.device ? ` · ${meta.device}` : ""}` : "API offline";

  const barSets = {
    retrieval: [
      ["Procrustes rotation (ours)", ret.procrustes?.top1, "ours"],
      ["Random rotation", ret.random_rotation?.top1],
      ["No rotation", ret.unaligned?.top1],
      ["Chance (random pick)", ret.procrustes ? 0.1 : null],
    ],
    classify: [
      ["ResNet-18 (trained on labels)", m.resnet18?.accuracy, "strong"],
      ["Procrustes rotation (ours)", cc.procrustes?.accuracy, "ours"],
      ["Random rotation (control)", cc.random_rotation?.accuracy],
      ["Majority class", m.majority?.accuracy],
      ["No rotation (control)", cc.unaligned?.accuracy],
    ],
  };
  $$("[data-bars]").forEach((el) => {
    const rows = barSets[el.dataset.bars].filter((r) => r[1] != null);
    el.innerHTML = rows.map(([n, v, cls]) => `<div class="bar ${cls || ""}"><span class="name">${esc(n)}</span><span class="track"><span class="fill" data-w="${v}"></span></span><span class="v">${pct(v, 1)}</span></div>`).join("");
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return;
      $$(".fill", el).forEach((f, i) => setTimeout(() => (f.style.width = `${f.dataset.w * 100}%`), i * 120));
      io.disconnect();
    }), { threshold: .3 });
    io.observe(el);
  });
}

function renderConfusion(meta) {
  const m = meta?.metrics || {}, cm = m.confusion, cls = Object.fromEntries((meta?.classes || []).map((c) => [c.name, c]));
  const short = (n) => ({ AnnualCrop: "Annual crop", HerbaceousVegetation: "Herbaceous", PermanentCrop: "Perm. crop", SeaLake: "Sea/lake", Industrial: "Industrial", Residential: "Residential" }[n] || n);
  if (cm?.matrix) {
    const names = cm.classes, n = names.length;
    let h = `<div class="cm" style="grid-template-columns:84px repeat(${n}, minmax(0,1fr))"><span></span>${names.map((c) => `<span class="top">${esc(short(c))}</span>`).join("")}`;
    cm.matrix.forEach((row, i) => {
      const tot = row.reduce((a, b) => a + b, 0) || 1;
      h += `<span class="rl">${esc(short((cm.rows || names)[i]))}</span>` + row.map((v, j) => {
        const f = v / tot, diag = i === j;
        const bg = diag ? `rgba(63,138,69,${.12 + f * .88})` : f > 0 ? `rgba(194,83,44,${Math.min(1, f * 2.2)})` : "transparent";
        return `<span class="c" style="background:${bg};color:${f > .45 ? "#fff" : "var(--muted)"}" title="${esc(names[i])} → ${esc(names[j])}: ${v}">${f >= .05 ? Math.round(f * 100) : ""}</span>`;
      }).join("");
    });
    $("#confusion").innerHTML = h + `</div><p class="note">% of each true class. Green = right, orange = mixed up. River → Highway stands out.</p>`;
  }
  if (m.per_class?.length) {
    $("#perclass").innerHTML = `<div class="pc">${m.per_class.map((r) => `<div class="row"><span>${esc(short(r.class))}</span><span class="tr"><i class="a" style="width:${r.recall * 100}%"></i><i class="b" style="width:${(r.resnet18_recall ?? 0) * 100}%"></i></span><span class="v">${pct(r.recall)}</span></div>`).join("")}
      <div class="key"><span><i style="background:var(--moss-2)"></i>ours</span><span><i style="background:var(--river)"></i>ResNet-18</span></div></div>`;
  }
}

// --------------------------------------------------------------- chapter 3: search Upper Arlington
function initSearch(meta, story) {
  const form = $("#ua-form"), input = $("#ua-q"), out = $("#ua-results"), note = $("#ua-note"), pins = $("#ua-pins");
  let gallery = "local";
  const examples = meta?.examples || ["dense forest", "residential neighborhood", "park or grassland", "commercial area", "river corridor"];
  $("#ua-chips").innerHTML = examples.map((q) => `<button class="chip" type="button" data-q="${esc(q)}">${esc(q)}</button>`).join("") +
    `<button class="chip" type="button" data-q="climate change" title="A failure case: not a land cover">climate change <small>probe</small></button>`;
  $$("#ua-chips .chip").forEach((b) => b.addEventListener("click", () => { input.value = b.dataset.q; run(); }));
  $$("[data-gallery]").forEach((b) => b.addEventListener("click", () => {
    gallery = b.dataset.gallery; $$("[data-gallery]").forEach((x) => x.classList.toggle("on", x === b)); run();
  }));

  // pins for the 50 local tiles on the mosaic
  const ua = story?.ua; const tiles = meta?.local_tiles || [];
  const pos = {};
  if (ua && tiles.length) {
    const [tl, tr, , bl] = ua.corners;
    const ax = tr[0] - tl[0], ay = tr[1] - tl[1], bx = bl[0] - tl[0], by = bl[1] - tl[1], det = ax * by - ay * bx;
    const Wpx = ua.cols * 64, Hpx = ua.rows * 64;
    pins.innerHTML = tiles.map((t) => {
      const dx = t.lon - tl[0], dy = t.lat - tl[1];
      const u = (dx * by - dy * bx) / det, v = (ax * dy - ay * dx) / det;
      const x = u * Wpx, y = v * Hpx; pos[t.tile_id] = [x, y];
      return `<rect class="pin" data-id="${t.tile_id}" x="${x - 32}" y="${y - 32}" width="64" height="64" rx="4"><title>${esc(t.tile_id)} · ${esc(t.label)}</title></rect>`;
    }).join("");
  }

  const EXPECT = { "dense forest": ["Forest"], "residential neighborhood": ["Residential"], "park or grassland": ["HerbaceousVegetation", "Pasture"],
    "commercial area": ["Industrial"], "river corridor": ["River"] };
  async function run(e) {
    e && e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    out.innerHTML = Array.from({ length: 6 }, () => `<div class="tile-card" style="opacity:.4"><div style="aspect-ratio:1;background:var(--paper-3)"></div><div class="meta"><b>&nbsp;</b></div></div>`).join("");
    try {
      const r = await api(`/api/search?q=${encodeURIComponent(q)}&gallery=${gallery}&k=6`);
      const want = EXPECT[q.toLowerCase()];
      const cls = Object.fromEntries((meta?.classes || []).map((c) => [c.name, c.label]));
      out.innerHTML = r.hits.map((h, i) => `
        <figure class="tile-card ${want ? (want.includes(h.label) ? "hit" : "miss") : ""}" style="animation-delay:${i * 60}ms;margin:0" data-id="${esc(h.tile_id)}">
          <img src="${h.image}" alt="${esc(h.label)} tile ${esc(h.tile_id)}" loading="lazy">
          <figcaption class="meta"><b>${want ? (want.includes(h.label) ? "✓ " : "✗ ") : ""}${esc(cls[h.label] || h.label)}</b><span class="mono muted">${h.score.toFixed(2)}</span></figcaption>
        </figure>`).join("");
      const nOk = want ? r.hits.filter((h) => want.includes(h.label)).length : null;
      $("#ua-key").innerHTML = want
        ? `<b>Checked against the truth:</b> ${nOk} of ${r.hits.length} results really are ${want.map((w) => cls[w] || w).join(" or ")} (green ✓ = right, red ✗ = wrong, by our human labels).`
        : `Labels under each photo are the human labels. Try a preset button to see results checked against the truth.`;
      note.textContent = "The number is a similarity score: higher = closer match.";
      $$(".pin", pins).forEach((p) => p.classList.remove("hit", "top"));
      if (gallery === "local") r.hits.forEach((h, i) => { const p = $(`.pin[data-id="${h.tile_id}"]`, pins); if (p) { p.classList.add("hit"); if (i === 0) p.classList.add("top"); } });
    } catch (err) {
      out.innerHTML = `<div class="empty" style="grid-column:1/-1">${esc(err.message)}</div>`;
    }
  }
  form.addEventListener("submit", run);
  // first search when the section comes into view
  const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { run(); io.disconnect(); } }, { rootMargin: "200px" });
  io.observe($("#search"));
}

// --------------------------------------------------------------- chapter 4: heat scrolly
function rampCss(stops) { return `linear-gradient(90deg, ${stops.map(([p, c]) => `${c} ${(p * 100).toFixed(0)}%`).join(",")})`; }
async function initHeat(meta, story) {
  const ua = story?.ua; if (!ua) return;
  const layers = $$("#heat-visual .layer"), sc = $("#heat-scatter"), tag = $("#heat-tag"), legend = $("#heat-legend");
  $$('[data-m="ua_scene_date"]').forEach((e) => (e.textContent = niceDate(ua.scene.datetime)));
  const gap = ua.days_from_s2;
  $$('[data-m="ua_lst_gap"]').forEach((e) => (e.textContent = gap === 0 ? "the same day" : `${Math.abs(gap)} days ${gap < 0 ? "earlier" : "later"}`));
  // points: land tiles only (water > 0.5 left out), as in web/analysis.climate_stats
  const pts = [];
  ua.tile_ndvi.forEach((n, i) => { const t = ua.tile_lst[i], w = ua.tile_water[i]; if (n != null && t != null && !(w > .5)) pts.push([n, t, i]); });
  // headline stats: prefer the server's numbers for this exact region, fall back to local OLS
  let stats = null;
  try {
    const preset = (meta?.presets || []).find((p) => p.key === "upper-arlington");
    if (preset?.cached) {
      const job = await api("/api/region", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ bbox: preset.bbox }) });
      const reg = await pollJob(job.job);
      const cj = await api(`/api/region/${reg.id}/climate`, { method: "POST" });
      stats = (await pollJob(cj.job)).stats;
    }
  } catch (e) { stats = null; }
  if (!stats) {
    const n = pts.length, mx = pts.reduce((a, p) => a + p[0], 0) / n, my = pts.reduce((a, p) => a + p[1], 0) / n;
    let sxy = 0, sxx = 0, syy = 0; pts.forEach(([x, y]) => { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2; });
    const slope = sxy / sxx; const sorted = [...pts].sort((a, b) => a[0] - b[0]); const k = Math.floor(n / 3);
    const mean = (a) => a.reduce((s, p) => s + p[1], 0) / a.length;
    const lstv = ua.tile_lst.filter((v) => v != null), ndv = ua.tile_ndvi.filter((v) => v != null);
    stats = { regression: { slope_per_0p1_ndvi: slope * .1, intercept: my - slope * mx, r: sxy / Math.sqrt(sxx * syy), n },
      tercile: { hot_minus_green_c: mean(sorted.slice(0, k)) - mean(sorted.slice(-k)) },
      lst_min: Math.min(...lstv), lst_max: Math.max(...lstv), ndvi_mean: ndv.reduce((a, b) => a + b, 0) / ndv.length };
  }
  const H = (k, v) => $$(`[data-h="${k}"]`).forEach((e) => (e.textContent = v));
  H("ndvi_mean", fmt(stats.ndvi_mean, 2)); H("lst_min", fmt(stats.lst_min, 0)); H("lst_max", fmt(stats.lst_max, 0));
  if (stats.regression) { H("slope", `${stats.regression.slope_per_0p1_ndvi > 0 ? "+" : "−"}${fmt(Math.abs(stats.regression.slope_per_0p1_ndvi), 2)}`); H("r", fmt(stats.regression.r, 2).replace("-", "−")); H("n_fit", stats.regression.n); }
  if (stats.tercile) H("tercile", `${fmt(stats.tercile.hot_minus_green_c, 1)} °C`);

  const looks = [
    { tag: `True colour · Sentinel-2 · ${niceDate(ua.scene.datetime)}`, legend: null },
    { tag: "Greenness · NDVI · measured", legend: { title: "NDVI", stops: ua.ndvi_palette, ends: ua.ndvi_range.map((v) => v.toFixed(1)) } },
    { tag: `Surface temperature · Landsat · ${niceDate(ua.lst_scene.datetime)}`, legend: { title: "Land surface temperature, °C", stops: ua.lst_palette, ends: ua.lst_range.map((v) => `${v}°`) } },
    { tag: "Greenness vs heat · one dot per square", legend: null, scatter: true },
    { tag: "Greenness vs heat · one dot per square", legend: null, scatter: true },
  ];
  let cur = -1;
  function show(i) {
    if (i === cur) return; cur = i;
    const L = looks[i];
    layers.forEach((l) => { const k = +l.dataset.layer; l.style.opacity = i >= 3 ? (k === 2 ? .18 : 0) : (k <= i ? 1 : 0); });
    sc.style.opacity = L.scatter ? 1 : 0;
    tag.textContent = L.tag; tag.classList.toggle("off", !!L.scatter);
    if (L.legend) {
      legend.classList.add("on"); $("#heat-legend-title").textContent = L.legend.title;
      $(".ramp", legend).style.background = rampCss(L.legend.stops);
      const [a, b] = $$(".ends span", legend); a.textContent = L.legend.ends[0]; b.textContent = L.legend.ends[1];
    } else legend.classList.remove("on");
    if (L.scatter) drawScatter(sc, pts, stats.regression, meta);
  }
  $$("[data-heat]").forEach((s) => s.addEventListener("step", () => show(+s.dataset.heat)));
  show(0);
  addEventListener("resize", () => cur >= 3 && drawScatter(sc, pts, stats.regression, meta));
}

// shared with the lab: canvas scatter of NDVI (x) vs LST (y)
export function drawScatter(cv, pts, reg, meta, opts = {}) {
  const dpr = Math.min(2, devicePixelRatio || 1);
  const W = cv.clientWidth, Hh = cv.clientHeight; if (!W || !Hh) return;
  cv.width = W * dpr; cv.height = Hh * dpr;
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--ink").trim(), muted = css.getPropertyValue("--muted").trim(), line = css.getPropertyValue("--line").trim();
  const pad = { l: 46, r: 18, t: 36, b: 42 };
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(0, ...xs), x1 = Math.max(.9, ...xs);
  const y0 = Math.floor(Math.min(...ys) - .5), y1 = Math.ceil(Math.max(...ys) + .5);
  const X = (v) => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r), Y = (v) => Hh - pad.b - (v - y0) / (y1 - y0) * (Hh - pad.t - pad.b);
  ctx.font = '500 10px "JetBrains Mono", monospace'; ctx.fillStyle = muted; ctx.strokeStyle = line; ctx.lineWidth = 1;
  for (let v = 0; v <= x1 + 1e-9; v += .2) { ctx.beginPath(); ctx.moveTo(X(v), pad.t); ctx.lineTo(X(v), Hh - pad.b); ctx.stroke(); ctx.textAlign = "center"; ctx.fillText(v.toFixed(1), X(v), Hh - pad.b + 16); }
  const step = (y1 - y0) > 12 ? 4 : 2;
  for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) { ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(W - pad.r, Y(v)); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(`${v}°`, pad.l - 8, Y(v) + 3); }
  ctx.textAlign = "center"; ctx.fillText("greenness (NDVI) →", (pad.l + W - pad.r) / 2, Hh - 8);
  ctx.save(); ctx.translate(12, (pad.t + Hh - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("surface temperature °C →", 0, 0); ctx.restore();
  // dots coloured by temperature
  const tmin = Math.min(...ys), tmax = Math.max(...ys);
  pts.forEach(([x, y, i]) => {
    const t = (y - tmin) / (tmax - tmin || 1);
    ctx.fillStyle = heatColor(t); ctx.globalAlpha = opts.highlight != null && opts.highlight !== i ? .35 : .9;
    ctx.beginPath(); ctx.arc(X(x), Y(y), opts.highlight === i ? 6 : 3.6, 0, Math.PI * 2); ctx.fill();
  });
  ctx.globalAlpha = 1;
  if (reg) {
    const s = reg.slope_per_0p1_ndvi * 10, b = reg.intercept;
    ctx.strokeStyle = ink; ctx.lineWidth = 1.8; ctx.setLineDash([6, 5]);
    ctx.beginPath(); ctx.moveTo(X(Math.min(...xs)), Y(b + s * Math.min(...xs))); ctx.lineTo(X(Math.max(...xs)), Y(b + s * Math.max(...xs))); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = ink; ctx.textAlign = "left"; ctx.font = '600 12px "JetBrains Mono", monospace';
    ctx.fillText(`${reg.slope_per_0p1_ndvi > 0 ? "+" : "−"}${Math.abs(reg.slope_per_0p1_ndvi).toFixed(2)} °C per +0.1 NDVI · r = ${reg.r.toFixed(2)} · n = ${reg.n}`, pad.l, 20);
  }
  return { X, Y };
}
export function heatColor(t) {
  const stops = [[0, [29, 59, 115]], [.25, [75, 143, 179]], [.5, [227, 205, 120]], [.7, [242, 153, 74]], [.86, [209, 73, 46]], [1, [122, 20, 32]]];
  for (let k = 1; k < stops.length; k++) if (t <= stops[k][0]) {
    const [p0, c0] = stops[k - 1], [p1, c1] = stops[k], u = (t - p0) / (p1 - p0);
    return `rgb(${c0.map((c, j) => Math.round(lerp(c, c1[j], u))).join(",")})`;
  }
  return "rgb(122,20,32)";
}

export async function pollJob(id, onStage) {
  let seen = 0;
  for (;;) {
    const j = await api(`/api/jobs/${id}`);
    if (onStage) { j.stages.slice(seen).forEach(onStage); seen = j.stages.length; }
    if (j.status === "done") return j.result;
    if (j.status === "error") throw new Error(j.error || "The job failed.");
    await new Promise((r) => setTimeout(r, 450));
  }
}

// --------------------------------------------------------------- chapter 6: limits
function initLimits(meta) {
  const wrong = (meta?.local_tiles || []).filter((t) => t.correct === false);
  const cls = Object.fromEntries((meta?.classes || []).map((c) => [c.name, c.label]));
  if (wrong.length) {
    $("#wrong-count").textContent = `${wrong.length} of ${meta.local_tiles.length} wrong`;
    $("#wrong-grid").innerHTML = wrong.map((t) => `<figure><img src="${t.image}" alt="${esc(t.tile_id)}" loading="lazy"><figcaption><span>is <b>${esc(cls[t.label] || t.label)}</b></span><span class="no">said ${esc(cls[t.pred] || t.pred)}</span></figcaption></figure>`).join("");
  } else $(".wrong").style.display = "none";
  const tiles = Object.fromEntries((meta?.local_tiles || []).map((t) => [t.tile_id, t]));
  const ex = (meta?.local_examples || []).map((e) => ({ ...e, t: tiles[e.tile_id] })).filter((e) => e.t);
  const el = $("#lim-pair");
  if (ex.length) {
    el.innerHTML = `<div class="pair">${ex.map((e) => `<figure><img src="${e.t.image}" alt="${esc(e.tile_id)}"><figcaption>${esc(e.kind)} · ${esc(e.tile_id)}</figcaption></figure>`).join("")}</div>
      <div class="body">${ex.map((e) => `
        <div class="tag mono muted" style="text-transform:uppercase;letter-spacing:.12em;margin-top:6px">${esc(e.kind)} · ${esc(e.tile_id)}</div>
        <div class="verdict"><span>human: ${esc(e.t.label)}</span><span class="${e.t.correct ? "ok" : "no"}">ours: ${esc(e.t.pred)}</span>${e.t.resnet18_pred ? `<span class="${e.t.resnet18_correct ? "ok" : "no"}">ResNet-18: ${esc(e.t.resnet18_pred)}</span>` : ""}</div>
        <p style="margin-bottom:12px">${esc(e.why)}</p>`).join("")}</div>`;
  } else el.innerHTML = `<div class="body"><p>Examples load from the API.</p></div>`;
  const pr = meta?.probe;
  if (pr) {
    $("#probe-q").textContent = pr.query;
    $("#probe-hits").innerHTML = (pr.hits || []).map((h) => `<span>${esc(h.label)} · ${h.score.toFixed(2)}</span>`).join("");
  }
}

// --------------------------------------------------------------- glossary tooltips (hover, focus or tap any .term)
export const GLOSSARY = {
  baseline: ["Baseline", "The simplest possible guess, used as a bar to beat. If our model can't beat it, the model isn't adding anything."],
  majority: ["Majority-class baseline", "Always guessing the most common answer. It is right sometimes by luck, so a real model must do better."],
  unseen: ["Unseen test data", "Photos kept aside and never used while building the model. Testing on them shows how it does on new photos, not ones it memorised."],
  dinov2: ["DINOv2", "A picture AI from Meta, trained on 142 million images. It turns any photo into a list of 384 numbers that describe what is in it. We never retrained it."],
  minilm: ["MiniLM", "A small word AI. It turns any sentence into 384 numbers that capture its meaning. We never retrained it."],
  embedding: ["Embedding", "The list of numbers an AI uses to describe a photo or sentence. Think of it as a point on a map: similar things land close together."],
  procrustes: ["Procrustes rotation", "A classic math trick that finds the one rotation which best lines up two sets of points. It can turn the map but never stretch it."],
  cosine: ["Similarity score (cosine)", "How closely two points on the map point the same way. Higher = better match. It is not a percentage or a probability."],
  resnet: ["ResNet-18", "A standard image AI trained directly on the labelled photos. It is the strong, normal way to do this task, shown for comparison."],
  eurosat: ["EuroSAT", "A public dataset of 27,000 labelled Sentinel-2 satellite photos of Europe in 10 land types (forest, river, homes…)."],
  sentinel: ["Sentinel-2", "Free European satellites that photograph all land every few days. Each pixel covers 10 × 10 m."],
  landsat: ["Landsat", "NASA/USGS satellites with a heat camera. They measure how hot the ground surface is."],
  ndvi: ["NDVI (greenness)", "A number from 0 to about 0.9 that measures how many healthy leaves are in a spot, using how plants reflect light. Higher = greener."],
  lst: ["Land surface temperature", "How hot the ground, roofs and roads are, measured from space by Landsat. It is not the air temperature you feel."],
  confusion: ["Confusion matrix", "A grid showing what the model answered for each true land type. The diagonal is right; anything off the diagonal is a mix-up."],
  slope: ["°C per +0.1 greenness", "How much cooler a square gets, on average, when its greenness (NDVI) goes up by 0.1. A straight-line fit through the dots."],
  r: ["Pearson r", "How tightly the dots follow a line, from −1 to 1. −0.6 means a clear pattern: greener goes with cooler."],
  tercile: ["Least-green vs greenest third", "Split all squares into three groups by greenness, then compare the average temperature of the least-green group and the greenest group."],
  priority: ["Plant first", "Squares in the hottest quarter of the area that also have the fewest plants. These are the best places to add trees."],
  refuge: ["Cool refuges", "The greenest, coolest squares. Places to protect, and where people can cool down."],
  model: ["Model estimate", "A guess by the AI. It can be wrong, so check it on the ground."],
  measured: ["Measured", "Read directly from satellite sensors, not guessed by AI."],
  top1: ["Right first try (top-1)", "The very first result was the correct land type."],
};
function initGlossary() {
  const tip = document.createElement("div"); tip.className = "tip"; tip.setAttribute("role", "tooltip"); document.body.append(tip);
  let cur = null;
  const show = (el) => {
    const g = GLOSSARY[el.dataset.term]; if (!g) return;
    cur = el; tip.innerHTML = `<b>${esc(g[0])}</b>${esc(g[1])}`; tip.classList.add("on");
    const r = el.getBoundingClientRect(), w = Math.min(290, innerWidth - 24);
    tip.style.maxWidth = `${w}px`;
    const left = clamp(r.left + r.width / 2 - w / 2, 12, innerWidth - w - 12);
    tip.style.left = `${left}px`;
    const h = tip.offsetHeight;
    tip.style.top = `${r.top - h - 10 > 8 ? r.top - h - 10 : r.bottom + 10}px`;
  };
  const hide = () => { cur = null; tip.classList.remove("on"); };
  document.addEventListener("mouseover", (e) => { const t = e.target.closest?.(".term"); if (t) show(t); else if (cur) hide(); });
  document.addEventListener("focusin", (e) => { const t = e.target.closest?.(".term"); if (t) show(t); });
  document.addEventListener("focusout", hide);
  document.addEventListener("click", (e) => { const t = e.target.closest?.(".term"); if (t) { e.preventDefault(); cur === t ? hide() : show(t); } else hide(); });
  addEventListener("scroll", () => cur && hide(), { passive: true });
}
export const term = (k, text) => `<span class="term" data-term="${k}" tabindex="0">${text}</span>`;

// --------------------------------------------------------------- boot
async function boot() {
  initChrome(); initSteps(); initManifesto(); initGlossary();
  const [meta, story] = await Promise.all([
    api("/api/meta").catch((e) => { toast(`API unavailable: ${e.message}`); return null; }),
    fetch("/static/data/story.json").then((r) => r.json()).catch(() => null),
  ]);
  window.PE = { meta, story };
  initHero(story);
  bindMetrics(meta);
  initMinds(story, meta);
  initSearch(meta, story);
  initLimits(meta);
  initLab(meta);
  initHeat(meta, story);
  runScenes();
}
boot();
