import React from "react";
import { AbsoluteFill, Img, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Bar, Clip, Fade, Kicker, Pop, Tile } from "./parts";
import { C, F, FPS, N, fmt1 } from "./theme";
import marks from "./marks.json";

// when caption line i of a scene starts (seconds into the scene) — visuals follow the voice
const capAt = (key: string, i: number) => (N.timeline.find((t) => t.key === key)?.captions[i]?.from ?? 0);
const B = N.bench;

const s = (x: number) => Math.round(x * FPS);
const Big: React.FC<{ children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties }> = ({ children, size = 104, color = C.ink, style }) => (
  <div style={{ fontFamily: F.serif, fontSize: size, lineHeight: 1.04, letterSpacing: "-0.02em", color, fontWeight: 500, ...style }}>{children}</div>
);

// 1 — hook: real Landsat surface temperature of Upper Arlington, with the measured gap
export const Hook: React.FC = () => {
  const f = useCurrentFrame();
  const { durationInFrames: d } = useVideoConfig();
  const sc = interpolate(f, [0, d], [1.05, 1.16]);
  return (
    <Fade inF={1}>
      <AbsoluteFill style={{ backgroundColor: C.night }}>
        <Img src={staticFile("img/ua-lst.png")} style={{ position: "absolute", right: -40, top: -60, height: 1200, transform: `scale(${sc})`, transformOrigin: "60% 50%", imageRendering: "pixelated" }} />
        <AbsoluteFill style={{ background: "linear-gradient(90deg, rgba(11,20,15,.97) 0%, rgba(11,20,15,.88) 42%, rgba(11,20,15,0) 70%)" }} />
        <div style={{ position: "absolute", left: 110, top: 190, width: 980 }}>
          <Pop><div style={{ fontFamily: F.mono, fontSize: 28, letterSpacing: "0.14em", color: C.sprout, textTransform: "uppercase" }}>Upper Arlington, Ohio · {N.dates.lst}</div></Pop>
          <Pop delay={6}>
            <div style={{ display: "flex", gap: 70, marginTop: 40 }}>
              <div><Big size={150} color="#ff8a5c">{fmt1(N.heat.hot)}°C</Big><div style={{ fontFamily: F.sans, fontSize: 36, color: C.nightInk, marginTop: 6 }}>least-green third</div></div>
              <div><Big size={150} color="#8fc3e0">{fmt1(N.heat.green)}°C</Big><div style={{ fontFamily: F.sans, fontSize: 36, color: C.nightInk, marginTop: 6 }}>greenest third</div></div>
            </div>
          </Pop>
          <Pop delay={14}><div style={{ fontFamily: F.mono, fontSize: 24, color: C.nightMuted, marginTop: 36 }}>Ground surface temperature, measured by Landsat</div></Pop>
        </div>
      </AbsoluteFill>
    </Fade>
  );
};

// 2 — the question, over the site's real globe fly-in
export const Problem: React.FC = () => (
  <Fade>
    <Clip src="hero" from={9.5} s0={1} s1={1.12} dim={0.5} />
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", textAlign: "center" }}>
      <Pop><Big size={128} color="#fff">Where should the city plant</Big></Pop>
      <Pop delay={6}><Big size={128} color={C.sprout} style={{ fontStyle: "italic" }}>its next trees?</Big></Pop>
      <Pop delay={20}><div style={{ marginTop: 40, fontFamily: F.mono, fontSize: 30, color: C.nightInk, backgroundColor: "rgba(11,20,15,.7)", padding: "12px 26px", borderRadius: 999, letterSpacing: "0.06em" }}>For: Upper Arlington urban-forestry staff</div></Pop>
    </AbsoluteFill>
  </Fade>
);

// 3 — the idea, over the 50 real local tiles
export const Idea: React.FC = () => {
  const f = useCurrentFrame();
  const ids = Array.from({ length: N.local_n }, (_, i) => `ua_${String(i + 1).padStart(3, "0")}`);
  return (
    <Fade>
      <AbsoluteFill style={{ backgroundColor: C.paper }}>
        <div style={{ position: "absolute", right: 70, top: 90, display: "grid", gridTemplateColumns: "repeat(7, 118px)", gap: 12 }}>
          {ids.slice(0, 49).map((id, i) => (
            <div key={id} style={{ opacity: interpolate(f - i * 0.9, [0, 10], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) }}><Tile id={id} size={118} style={{ borderRadius: 8 }} /></div>
          ))}
        </div>
        <div style={{ position: "absolute", left: 110, top: 300, width: 860 }}>
          <Pop><Big size={96}>Search satellite photos of anywhere on Earth, in <i style={{ color: C.moss }}>plain English.</i></Big></Pop>
          <Pop delay={14}>
            <div style={{ display: "flex", gap: 16, marginTop: 44, flexWrap: "wrap" }}>
              {["2 AI models, not retrained", "1 rotation", "0 training"].map((t) => (
                <span key={t} style={{ fontFamily: F.mono, fontSize: 28, color: C.ink, backgroundColor: C.sprout, padding: "10px 20px", borderRadius: 999 }}>{t}</span>
              ))}
            </div>
          </Pop>
        </div>
      </AbsoluteFill>
    </Fade>
  );
};

// 4 — Search anywhere: radius, typed query, matches (real recording of the site running the 1M rotation)
export const Search: React.FC = () => {
  const c1 = capAt("search", 1), c2 = capAt("search", 2), m = marks.search;
  return (
    <Fade>
      <Sequence durationInFrames={s(c1)}><Clip src="search" from={m.radius - 0.5} fx={0.25} fy={0.35} s0={1.08} s1={1.16} /></Sequence>
      <Sequence from={s(c1)} durationInFrames={s(c2 - c1)}><Clip src="search" from={m.type - 0.2} fx={0.9} fy={0.3} s0={1.3} s1={1.36} /></Sequence>
      <Sequence from={s(c2)}><Clip src="search" from={m.results - 0.2} fx={0.3} fy={0.45} s0={1.1} s1={1.2} /></Sequence>
    </Fade>
  );
};

// 5 — heat on the same map, real recording
export const Heat: React.FC = () => {
  const c1 = capAt("heat", 1), c2 = capAt("heat", 2), c3 = capAt("heat", 3), m = marks.lab;
  return (
    <Fade>
      <Sequence durationInFrames={s(c1)}><Clip src="lab" from={m.ndvi - 0.4} fx={0.3} fy={0.45} s0={1.05} s1={1.14} /></Sequence>
      <Sequence from={s(c1)} durationInFrames={s(c2 - c1)}><Clip src="lab" from={m.heat + 0.6} fx={0.86} fy={0.42} s0={1.45} s1={1.55} /></Sequence>
      <Sequence from={s(c2)} durationInFrames={s(c3 - c2)}><Clip src="lab" from={m.priority + 0.6} fx={0.3} fy={0.5} s0={1.1} s1={1.18} /></Sequence>
      <Sequence from={s(c3)}><Clip src="lab" from={m.export + 0.4} fx={0.86} fy={0.35} s0={1.3} s1={1.38} /></Sequence>
    </Fade>
  );
};

// 6 — how it works: two encoders, one rotation, then the search result vs controls
const Box: React.FC<{ title: string; sub: string; color: string }> = ({ title, sub, color }) => (
  <div style={{ border: `3px solid ${color}`, borderRadius: 22, padding: "20px 28px", backgroundColor: C.card, width: 380 }}>
    <div style={{ fontFamily: F.sans, fontWeight: 700, fontSize: 38, color: C.ink }}>{title}</div>
    <div style={{ fontFamily: F.sans, fontSize: 26, color: C.muted, marginTop: 4 }}>{sub}</div>
  </div>
);
const Vec: React.FC<{ color: string; seed: number; delay: number }> = ({ color, seed, delay }) => {
  const f = useCurrentFrame();
  return (
    <div style={{ display: "flex", gap: 5, alignItems: "center", height: 90 }}>
      {Array.from({ length: 22 }, (_, i) => {
        const h = 20 + 60 * Math.abs(Math.sin(seed * 7.3 + i * 1.7));
        const p = interpolate(f - delay - i, [0, 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
        return <div key={i} style={{ width: 10, height: h * p, backgroundColor: color, borderRadius: 3 }} />;
      })}
    </div>
  );
};
export const How: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const c1 = capAt("how", 1), c2 = capAt("how", 2);
  const rot = spring({ frame: f - s(c1), fps, config: { damping: 18 }, durationInFrames: 40 }) * 360;
  return (
    <Fade>
      <AbsoluteFill style={{ backgroundColor: C.paper }}>
        <Kicker n="03" text="How it works" />
        <div style={{ position: "absolute", left: 110, top: 230, display: "flex", alignItems: "center", gap: 40 }}>
          <Pop delay={s(1.2)}><Tile id={N.river_tiles[0]} size={150} /></Pop>
          <Pop delay={s(1.4)}><Box title="DINOv2" sub="picture AI · Meta" color={C.moss} /></Pop>
          <Vec color={C.moss} seed={1} delay={s(1.7)} />
        </div>
        <div style={{ position: "absolute", left: 110, top: 560, display: "flex", alignItems: "center", gap: 40 }}>
          <Pop delay={s(3)}><div style={{ width: 150, fontFamily: F.serif, fontStyle: "italic", fontSize: 40, color: C.ink }}>“parking lots”</div></Pop>
          <Pop delay={s(3.2)}><Box title="MiniLM" sub="word AI" color={C.river} /></Pop>
          <Vec color={C.river} seed={2} delay={s(3.5)} />
        </div>
        {f > s(c1) && (
          <Pop delay={s(c1)} style={{ position: "absolute", left: 1300, top: 250, width: 520, textAlign: "center" }}>
            <svg width="260" height="260" viewBox="0 0 100 100" style={{ transform: `rotate(${rot}deg)` }}>
              <circle cx="50" cy="50" r="40" fill="none" stroke={C.clay} strokeWidth="6" strokeDasharray="200 52" />
              <path d="M80 18 L92 30 L76 34 Z" fill={C.clay} />
            </svg>
            <div style={{ fontFamily: F.serif, fontSize: 52, color: C.ink, marginTop: 10 }}>one rotation</div>
            <div style={{ fontFamily: F.mono, fontSize: 26, color: C.muted, marginTop: 6 }}>(orthogonal Procrustes) · both AIs frozen</div>
          </Pop>
        )}
        {f > s(c2) && (
          <Pop delay={s(c2)} style={{ position: "absolute", left: 1300, top: 660, width: 520, textAlign: "center" }}>
            <div style={{ fontFamily: F.serif, fontSize: 76, fontWeight: 600, color: C.moss }}>{B.n_fit.toLocaleString("en-US")}</div>
            <div style={{ fontFamily: F.sans, fontSize: 30, color: C.ink, marginTop: 4 }}>Sentinel-2 photos, captioned from OpenStreetMap</div>
          </Pop>
        )}
      </AbsoluteFill>
    </Fade>
  );
};

// 7 — the product test: text → places on held-out regions
export const Test: React.FC = () => {
  const f = useCurrentFrame();
  const c1 = capAt("test", 1), c2 = capAt("test", 2);
  return (
    <Fade>
      <AbsoluteFill style={{ backgroundColor: C.paper }}>
        <Kicker n="04" text="Results · search in words" />
        <div style={{ position: "absolute", left: 110, top: 190 }}>
          <Pop><Big size={66}>How many of the top {B.k} results are right?</Big></Pop>
          <Pop delay={4}><div style={{ fontFamily: F.mono, fontSize: 26, color: C.muted, marginTop: 10, maxWidth: 1700 }}>{B.pairs} searches · {B.squares.toLocaleString("en-US")} squares · {B.regions} regions held out from training · “right” = mapped in OpenStreetMap</div></Pop>
        </div>
        <div style={{ position: "absolute", left: 0, top: 410, display: "flex", flexDirection: "column", gap: 30 }}>
          <Bar label="Our rotation" sub="1M photos" value={B.ours} color={C.moss} delay={s(c1)} bold max={50} />
          <Bar label="No rotation" sub="control" value={B.none} color={C.ochre} delay={s(c1) + 8} max={50} />
          <Bar label="Chance" value={B.chance} color="#b9b2a0" delay={s(c1) + 16} max={50} />
        </div>
        {f > s(c2) && (
          <Pop delay={s(c2)} style={{ position: "absolute", right: 110, top: 760 }}>
            <div style={{ fontFamily: F.sans, fontSize: 34, color: C.ink, backgroundColor: C.sprout, padding: "14px 24px", borderRadius: 14 }}>“parking lots”: <b>{B.parking}%</b> right · chance {B.parking_chance}%</div>
          </Pop>
        )}
      </AbsoluteFill>
    </Fade>
  );
};

// 8 — the baseline: our own first rotation
export const Baseline: React.FC = () => (
  <Fade>
    <AbsoluteFill style={{ backgroundColor: C.paper }}>
      <Kicker n="04" text="Results · versus our first rotation" />
      <div style={{ position: "absolute", left: 110, top: 200 }}>
        <Pop><Big size={66}>A million photos helped only a little.</Big></Pop>
        <Pop delay={4}><div style={{ fontFamily: F.mono, fontSize: 26, color: C.muted, marginTop: 10 }}>same {B.pairs} searches, same places · top-{B.k} results that are right</div></Pop>
      </div>
      <div style={{ position: "absolute", left: 0, top: 440, display: "flex", flexDirection: "column", gap: 34 }}>
        <Bar label="1M-photo rotation" value={B.ours} color={C.moss} delay={s(1.6)} bold max={50} />
        <Bar label="First rotation" sub="EuroSAT photos" value={B.first} color={C.river} delay={s(3.2)} max={50} />
      </div>
      <Pop delay={s(3.8)} style={{ position: "absolute", right: 110, top: 760 }}>
        <div style={{ fontFamily: F.sans, fontSize: 32, color: C.ink, backgroundColor: C.sprout, padding: "14px 24px", borderRadius: 14 }}>+{B.diff} points (95% CI {B.diff_ci[0]} to {B.diff_ci[1]})</div>
      </Pop>
    </AbsoluteFill>
  </Fade>
);

// 8 — honest limits: the local tie and the river failure
export const Limits: React.FC = () => {
  const f = useCurrentFrame();
  const c1 = capAt("limits", 1);
  const part2 = f >= s(c1);
  return (
    <Fade>
      <AbsoluteFill style={{ backgroundColor: C.night }}>
        <Kicker n="05" text="Where it fails" dark />
        {!part2 ? (
          <>
            <div style={{ position: "absolute", left: 110, top: 200 }}>
              <Pop><Big size={66} color={C.nightInk}>Some searches are <span style={{ color: C.ochre }}>near chance.</span></Big></Pop>
              <Pop delay={4}><div style={{ fontFamily: F.mono, fontSize: 26, color: C.nightMuted, marginTop: 10 }}>top-{B.k} results that are right · held-out regions</div></Pop>
            </div>
            <div style={{ position: "absolute", left: 0, top: 410, display: "flex", flexDirection: "column", gap: 22 }}>
              {B.weak.flatMap((w, i) => [
                <Bar key={w.q} label={`“${w.q}”`} value={w.ours} color={C.fern} delay={s(1.2) + i * 14} bold dark max={50} />,
                <Bar key={w.q + "c"} label="chance" value={w.chance} color="#8a8576" delay={s(1.2) + i * 14 + 5} dark max={50} />,
              ])}
            </div>
          </>
        ) : (
          <Sequence from={s(c1)} layout="none">
            <div style={{ position: "absolute", left: 110, top: 200 }}>
              <Pop><Big size={66} color={C.nightInk}>Rivers: <span style={{ color: "#ff8a5c" }}>{N.local.river_right} of {N.local.river_n}</span> named right.</Big></Pop>
              <Pop delay={4}><div style={{ fontFamily: F.mono, fontSize: 26, color: C.nightMuted, marginTop: 10 }}>Upper Arlington, labelled by us · at 10 m per pixel, a thin river looks like a road.</div></Pop>
            </div>
            <div style={{ position: "absolute", left: 110, top: 420, display: "flex", gap: 46 }}>
              {N.river_wrong.slice(0, 5).map((t, i) => (
                <Pop key={t.id} delay={8 + i * 5}>
                  <Tile id={t.id} size={250} />
                  <div style={{ fontFamily: F.mono, fontSize: 26, color: C.nightInk, marginTop: 12 }}>river · <span style={{ color: "#ff8a5c" }}>said {t.pred}</span></div>
                </Pop>
              ))}
            </div>
          </Sequence>
        )}
      </AbsoluteFill>
    </Fade>
  );
};

// 9 — next step
export const Next: React.FC = () => (
  <Fade>
    <AbsoluteFill style={{ backgroundColor: C.paper }}>
      <Kicker n="06" text="Next" />
      <div style={{ position: "absolute", left: 110, top: 280, display: "flex", flexDirection: "column", gap: 44 }}>
        {[
          ["Sharper aerial photos", "find open grass, block by block"],
          ["An honest “I don’t know”", "for rivers and odd questions"],
          ["More local labels", "checked by city staff"],
        ].map(([a, b], i) => (
          <Pop key={a} delay={i * 8}>
            <Big size={78}>{a}</Big>
            <div style={{ fontFamily: F.sans, fontSize: 36, color: C.muted, marginTop: 6 }}>{b}</div>
          </Pop>
        ))}
      </div>
      <Img src={staticFile("img/ua-ndvi.png")} style={{ position: "absolute", right: 110, top: 160, height: 680, borderRadius: 24, imageRendering: "pixelated" }} />
    </AbsoluteFill>
  </Fade>
);

// 10 — end card
export const End: React.FC = () => (
  <Fade outF={1}>
    <AbsoluteFill style={{ backgroundColor: C.night, justifyContent: "center", alignItems: "center", textAlign: "center" }}>
      <Pop><Big size={140} color={C.nightInk}>Platonic <i style={{ color: C.sprout }}>Earth</i></Big></Pop>
      <Pop delay={8}><div style={{ fontFamily: F.mono, fontSize: 40, color: C.sprout, marginTop: 34 }}>{N.site_url}</div></Pop>
      <Pop delay={12}><div style={{ fontFamily: F.mono, fontSize: 30, color: C.nightMuted, marginTop: 14 }}>{N.repo_url}</div></Pop>
      <Pop delay={20}>
        <div style={{ fontFamily: F.sans, fontSize: 24, color: C.nightMuted, marginTop: 70, maxWidth: 1500, lineHeight: 1.5 }}>
          {N.disclaimer} Imagery: Copernicus Sentinel-2 (ESA), Landsat (USGS/NASA), Esri basemap. Models: DINOv2 (Meta), all-MiniLM-L6-v2. Data: © OpenStreetMap contributors, EuroSAT.
          Built with help from AI coding tools (Claude). Veritas AI High School Hackathon 2026 · AI + Climate.
        </div>
      </Pop>
    </AbsoluteFill>
  </Fade>
);

