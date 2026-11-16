import React from "react";
import { AbsoluteFill, Img, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Bar, Clip, Fade, Kicker, Pop, Tile } from "./parts";
import { C, F, FPS, N, fmt1 } from "./theme";

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
          <Pop><Big size={96}>Search satellite photos in <i style={{ color: C.moss }}>plain English.</i></Big></Pop>
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

// 4 — live search, real recording
export const Search: React.FC = () => (
  <Fade>
    <Sequence durationInFrames={s(9)}><Clip src="search" from={0.6} fx={0.3} fy={0.6} s0={1.12} s1={1.22} /></Sequence>
    <Sequence from={s(9)}><Clip src="search" from={9.3} fx={0.3} fy={0.6} s0={1.16} s1={1.24} /></Sequence>
  </Fade>
);

// 5 — the Heat Lab, real recording
export const Heat: React.FC = () => (
  <Fade>
    <Sequence durationInFrames={s(6)}><Clip src="lab" from={3} fx={0.3} fy={0.45} s0={1.05} s1={1.15} /></Sequence>
    <Sequence from={s(6)} durationInFrames={s(6)}><Clip src="lab" from={5} fx={0.86} fy={0.42} s0={1.45} s1={1.55} /></Sequence>
    <Sequence from={s(12)} durationInFrames={s(5)}><Clip src="lab" from={12} fx={0.3} fy={0.45} s0={1.1} s1={1.18} /></Sequence>
    <Sequence from={s(17)}><Clip src="lab" from={17.8} fx={0.78} fy={0.45} s0={1.25} s1={1.35} /></Sequence>
  </Fade>
);

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
  const rot = spring({ frame: f - s(6), fps, config: { damping: 18 }, durationInFrames: 40 }) * 360;
  const showBars = f >= s(12);
  const o1 = interpolate(f, [s(11.5), s(12)], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <Fade>
      <AbsoluteFill style={{ backgroundColor: C.paper }}>
        <Kicker n="03" text="How it works" />
        {!showBars && (
          <AbsoluteFill style={{ opacity: o1 }}>
            <div style={{ position: "absolute", left: 110, top: 230, display: "flex", alignItems: "center", gap: 40 }}>
              <Pop><Tile id={N.river_tiles[0]} size={150} /></Pop>
              <Pop delay={6}><Box title="DINOv2" sub="picture AI · Meta" color={C.moss} /></Pop>
              <Vec color={C.moss} seed={1} delay={14} />
            </div>
            <div style={{ position: "absolute", left: 110, top: 560, display: "flex", alignItems: "center", gap: 40 }}>
              <Pop delay={20}><div style={{ width: 150, fontFamily: F.serif, fontStyle: "italic", fontSize: 40, color: C.ink }}>“river corridor”</div></Pop>
              <Pop delay={26}><Box title="MiniLM" sub="word AI" color={C.river} /></Pop>
              <Vec color={C.river} seed={2} delay={34} />
            </div>
            {f > s(5.5) && (
              <Pop delay={s(5.5)} style={{ position: "absolute", left: 1300, top: 300, width: 520, textAlign: "center" }}>
                <svg width="260" height="260" viewBox="0 0 100 100" style={{ transform: `rotate(${rot}deg)` }}>
                  <circle cx="50" cy="50" r="40" fill="none" stroke={C.clay} strokeWidth="6" strokeDasharray="200 52" />
                  <path d="M80 18 L92 30 L76 34 Z" fill={C.clay} />
                </svg>
                <div style={{ fontFamily: F.serif, fontSize: 52, color: C.ink, marginTop: 10 }}>one rotation</div>
                <div style={{ fontFamily: F.mono, fontSize: 26, color: C.muted, marginTop: 6 }}>(Procrustes) · fit on EuroSAT</div>
              </Pop>
            )}
          </AbsoluteFill>
        )}
        {showBars && (
          <Sequence from={s(12)} layout="none">
            <div style={{ position: "absolute", left: 110, top: 200 }}>
              <Pop><Big size={66}>Right land type comes first</Big></Pop>
              <Pop delay={4}><div style={{ fontFamily: F.mono, fontSize: 26, color: C.muted, marginTop: 10 }}>{N.search.queries} text searches · {N.search.tiles.toLocaleString("en-US")} photos never used for fitting</div></Pop>
            </div>
            <div style={{ position: "absolute", left: 0, top: 430, display: "flex", flexDirection: "column", gap: 34 }}>
              <Bar label="Our rotation" value={N.search.ours} color={C.moss} delay={6} bold />
              <Bar label="Random rotation" sub="control" value={N.search.random} color={C.ochre} delay={12} />
              <Bar label="No rotation" sub="control" value={N.search.none} color="#b9b2a0" delay={18} />
            </div>
          </Sequence>
        )}
      </AbsoluteFill>
    </Fade>
  );
};

// 7 — the strong baseline, stated plainly
export const Baseline: React.FC = () => (
  <Fade>
    <AbsoluteFill style={{ backgroundColor: C.paper }}>
      <Kicker n="04" text="Results · label each photo" />
      <div style={{ position: "absolute", left: 110, top: 200 }}>
        <Pop><Big size={66}>A trained model labels better.</Big></Pop>
        <Pop delay={4}><div style={{ fontFamily: F.mono, fontSize: 26, color: C.muted, marginTop: 10 }}>{N.label.n.toLocaleString("en-US")} unseen EuroSAT photos · same task for every row</div></Pop>
      </div>
      <div style={{ position: "absolute", left: 0, top: 420, display: "flex", flexDirection: "column", gap: 30 }}>
        <Bar label="ResNet-18" sub="trained on the labels" value={N.label.resnet} color={C.river} delay={6} />
        <Bar label="Our rotation" sub="no training" value={N.label.ours} color={C.moss} delay={12} bold />
        <Bar label="Always most common" sub="baseline" value={N.label.majority} color="#b9b2a0" delay={18} />
      </div>
      <Pop delay={40} style={{ position: "absolute", right: 110, top: 220 }}>
        <div style={{ fontFamily: F.sans, fontSize: 32, color: C.ink, backgroundColor: C.sprout, padding: "14px 24px", borderRadius: 14, maxWidth: 520 }}>ResNet can’t take a question in words. Ours can.</div>
      </Pop>
    </AbsoluteFill>
  </Fade>
);

// 8 — honest limits: the local tie and the river failure
export const Limits: React.FC = () => {
  const f = useCurrentFrame();
  const part2 = f >= s(7);
  return (
    <Fade>
      <AbsoluteFill style={{ backgroundColor: C.night }}>
        <Kicker n="05" text="Where it fails" dark />
        {!part2 ? (
          <>
            <div style={{ position: "absolute", left: 110, top: 200 }}>
              <Pop><Big size={66} color={C.nightInk}>Our own town: <span style={{ color: C.ochre }}>a tie.</span></Big></Pop>
              <Pop delay={4}><div style={{ fontFamily: F.mono, fontSize: 26, color: C.nightMuted, marginTop: 10 }}>{N.local.n} Upper Arlington photos, labelled by us</div></Pop>
            </div>
            <div style={{ position: "absolute", left: 0, top: 430, display: "flex", flexDirection: "column", gap: 30 }}>
              <Bar label="Our rotation" value={N.local.ours} color={C.fern} delay={6} bold dark />
              <Bar label="ResNet-18" value={N.local.resnet} color={C.river} delay={10} dark />
              <Bar label="Always “Residential”" sub="baseline" value={N.local.oracle} color="#8a8576" delay={14} dark />
            </div>
          </>
        ) : (
          <Sequence from={s(7)} layout="none">
            <div style={{ position: "absolute", left: 110, top: 200 }}>
              <Pop><Big size={66} color={C.nightInk}>Rivers: <span style={{ color: "#ff8a5c" }}>{N.local.river_right} of {N.local.river_n}</span> named right.</Big></Pop>
              <Pop delay={4}><div style={{ fontFamily: F.mono, fontSize: 26, color: C.nightMuted, marginTop: 10 }}>At 10 m per pixel, a thin river looks like a road.</div></Pop>
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
          {N.disclaimer} Imagery: Copernicus Sentinel-2 (ESA), Landsat (USGS/NASA), Esri basemap. Models: DINOv2 (Meta), all-MiniLM-L6-v2, ResNet-18. Data: EuroSAT.
          Built with help from AI coding tools (Claude). Veritas AI High School Hackathon 2026 · AI + Climate.
        </div>
      </Pop>
    </AbsoluteFill>
  </Fade>
);

