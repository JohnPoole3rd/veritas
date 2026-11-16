// Static-site API (owner: D). Loaded ONLY in the GitHub Pages copy built by web/build_static.py.
// It answers the page's /api/... calls from pre-computed files, so every number matches the live server.
// Free-text search runs the same MiniLM text model in the browser (transformers.js), subtracts the same text
// mean (mu_y) and compares with the rotated image vectors exported by the build: the same maths as web/server.py.
(function () {
  "use strict";
  const realFetch = window.fetch.bind(window);
  window.PE_STATIC = true;
  const base = new URL(".", document.baseURI);
  const url = (p) => new URL(p, base).toString();
  const slug = (q) => q.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "q";
  const ok = (obj) => new Response(JSON.stringify(obj), { status: 200, headers: { "Content-Type": "application/json" } });
  const fail = (status, error) => new Response(JSON.stringify({ error }), { status, headers: { "Content-Type": "application/json" } });
  const getJSON = async (p) => { const r = await realFetch(url(p)); if (!r.ok) throw new Error(`missing ${p}`); return r.json(); };
  const getBin = async (p, T) => { const r = await realFetch(url(p)); if (!r.ok) throw new Error(`missing ${p}`); return new T(await r.arrayBuffer()); };
  const LIVE_ONLY = "This public copy only has the saved areas (Upper Arlington and Phoenix). New places need the live server: see “Run it yourself” on GitHub.";

  let userAsked = false; // the silent pre-load shouldn't pop messages at people who never search
  function note(msg, ms) {
    if (window.PE_QUIET || !userAsked) return;
    const t = document.getElementById("toast"); if (!t) return;
    t.textContent = msg; t.classList.add("on");
    clearTimeout(note._t); note._t = setTimeout(() => t.classList.remove("on"), ms);
  }
  // start fetching the model early (idle time), so the first typed search is fast
  addEventListener("load", () => setTimeout(() => { if (!window.PE_NO_PRELOAD) embedText("warm up", true).catch(() => {}); }, 4000));

  let cfgP = null;
  const cfg = () => (cfgP ||= getJSON("api/static.json"));
  const mem = {};
  const once = (k, f) => (mem[k] ||= f());

  // ---- float16 -> float32 rows (vectors are already L2-normalised by the build)
  function halfToFloat(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  async function matrix(p, dim) {
    const h = await getBin(p, Uint16Array);
    const out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) out[i] = halfToFloat(h[i]);
    return { data: out, n: out.length / dim, dim };
  }

  // ---- the text model, in the browser
  let extractorP = null;
  async function embedText(q, background) {
    if (!background) { userAsked = true; if (!extractorP) note("Loading the word model into your browser (one time)…", 60000); }
    const c = await cfg();
    extractorP ||= (async () => {
      const T = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1");
      const ex = await T.pipeline("feature-extraction", c.browser_model, { dtype: window.PE_DTYPE || "fp16" });
      if (userAsked) note("Word model ready. Searches now run in your browser.", 2500);
      return ex;
    })();
    const ex = await extractorP;
    const out = await ex(q, { pooling: "mean", normalize: true });
    const v = Float32Array.from(out.data);
    const mu = await once("mu", () => getBin("api/mu_y.bin", Float32Array));
    let n = 0;
    for (let i = 0; i < v.length; i++) { v[i] -= mu[i]; n += v[i] * v[i]; }
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= n;
    return v;
  }
  function scores(M, v) {
    const s = new Float32Array(M.n);
    for (let r = 0; r < M.n; r++) { let d = 0; const o = r * M.dim; for (let k = 0; k < M.dim; k++) d += M.data[o + k] * v[k]; s[r] = d; }
    return s;
  }
  const r4 = (x) => Math.round(x * 1e4) / 1e4;
  function topK(s, k, valid, exclude) {
    const idx = [];
    for (let i = 0; i < s.length; i++) if (isFinite(s[i]) && (!valid || valid[i]) && i !== exclude) idx.push(i);
    idx.sort((a, b) => s[b] - s[a] || a - b);
    return idx.slice(0, k).map((i) => ({ i, score: r4(s[i]) }));
  }
  function payload(s, k, valid, query, exclude) {
    const vs = []; for (let i = 0; i < s.length; i++) if ((!valid || valid[i]) && isFinite(s[i])) vs.push(s[i]);
    return { query, scores: Array.from(s, r4), top: topK(s, k, valid, exclude),
      min: vs.length ? r4(Math.min(...vs)) : null, max: vs.length ? r4(Math.max(...vs)) : null };
  }

  // ---- routes
  async function gallerySearch(q, gallery, k) {
    const c = await cfg();
    const G = c.galleries[gallery];
    if (!G) return fail(404, "That gallery isn't in this copy.");
    if (c.baked_queries.includes(q.toLowerCase())) {
      try { return ok(await getJSON(`api/search/${gallery}/${slug(q)}.json`)); } catch (e) { /* fall through */ }
    }
    try {
      const M = await once(`g:${gallery}`, () => matrix(`api/gallery/${gallery}.bin`, G.dim));
      const s = scores(M, await embedText(q));
      const hits = topK(s, k).map(({ i, score }) => {
        const it = G.items[i];
        const h = { tile_id: it.tile_id, label: it.label, score, image: `api/tiles/${gallery}/${it.tile_id}.${G.ext}` };
        if (gallery === "local") { h.lat = it.lat; h.lon = it.lon; }
        return h;
      });
      return ok({ query: q, gallery, hits });
    } catch (e) {
      return fail(503, `The text model couldn't load in this browser (${e.message}). Try a preset button.`);
    }
  }
  const regionByBox = async (bbox) => (await cfg()).regions.find((r) => r.bbox.every((v, i) => Math.abs(v - bbox[i]) < 1e-4));
  const jobs = {};
  async function route(path, opts) {
    const u = new URL(path, "http://x/");
    const p = u.pathname.replace(/^\/?api\//, "").replace(/^\//, "");
    const method = (opts?.method || "GET").toUpperCase();
    const body = opts?.body ? JSON.parse(opts.body) : null;
    let m;
    if (p === "meta") return ok(await getJSON("api/meta.json"));
    if (p === "search") return gallerySearch(u.searchParams.get("q") || "", u.searchParams.get("gallery") || "local", +(u.searchParams.get("k") || 6));
    if (p === "geocode") return ok([]);
    if (p === "regions/cached") return ok((await cfg()).regions);
    if (p === "region/estimate") {
      const r = await regionByBox(body.bbox);
      if (r) return ok(await getJSON(`api/region/${r.id}/estimate.json`));
      return ok({ ok: false, message: LIVE_ONLY, n_tiles: 0, width_km: 0, height_km: 0 });
    }
    if (p === "region" && method === "POST") {
      const r = await regionByBox(body.bbox);
      if (!r || body.when) return fail(503, LIVE_ONLY);
      const id = `r${Object.keys(jobs).length}`; jobs[id] = { kind: "region", rid: r.id }; return ok({ job: id });
    }
    if ((m = p.match(/^region\/([\w-]+)\/climate$/)) && method === "POST") {
      const id = `c${Object.keys(jobs).length}`; jobs[id] = { kind: "climate", rid: m[1] }; return ok({ job: id });
    }
    if ((m = p.match(/^jobs\/(\w+)$/))) {
      const j = jobs[m[1]]; if (!j) return fail(404, "No job with that id.");
      const stages = [{ stage: "cache", detail: "Opening the saved copy (this public site has no live satellite access)" }];
      const result = await getJSON(j.kind === "region" ? `api/region/${j.rid}.json` : `api/region/${j.rid}/climate.json`);
      return ok({ id: m[1], kind: j.kind, status: "done", stages, result });
    }
    if ((m = p.match(/^region\/([\w-]+)\/search$/))) {
      const rid = m[1], q = u.searchParams.get("q") || "", k = +(u.searchParams.get("k") || 8);
      const c = await cfg();
      if (c.baked_queries.includes(q.toLowerCase())) { try { return ok(await getJSON(`api/region/${rid}/search/${slug(q)}.json`)); } catch (e) { /* live */ } }
      try {
        const reg = c.regions.find((r) => r.id === rid);
        const M = await once(`a:${rid}`, () => matrix(`api/region/${rid}/aligned.bin`, c.galleries.local.dim));
        const valid = await once(`v:${rid}`, () => getJSON(`api/region/${rid}/valid.json`));
        return ok(payload(scores(M, await embedText(q)), k, valid, q));
      } catch (e) { return fail(503, `The text model couldn't load in this browser (${e.message}).`); }
    }
    if ((m = p.match(/^region\/([\w-]+)\/similar$/))) {
      const rid = m[1], i = +u.searchParams.get("i"), k = +(u.searchParams.get("k") || 8);
      const c = await cfg();
      const M = await once(`a:${rid}`, () => matrix(`api/region/${rid}/aligned.bin`, c.galleries.local.dim));
      const valid = await once(`v:${rid}`, () => getJSON(`api/region/${rid}/valid.json`));
      const v = M.data.slice(i * M.dim, (i + 1) * M.dim);
      return ok(payload(scores(M, v), k, valid, i, i));
    }
    if ((m = p.match(/^region\/([\w-]+)$/))) return ok(await getJSON(`api/region/${m[1]}.json`));
    return fail(404, LIVE_ONLY);
  }

  window.fetch = function (input, opts) {
    const s = typeof input === "string" ? input : input.url;
    if (/^\/?api\//.test(s) && !/\.(png|jpg|csv|geojson)$/.test(s.split("?")[0])) {
      return route(s, opts).catch((e) => fail(500, e.message));
    }
    if (/^\/static\//.test(s)) return realFetch(url(s.replace(/^\//, "")), opts);
    return realFetch(input, opts);
  };
})();
