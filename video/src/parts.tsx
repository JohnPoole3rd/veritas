import React from "react";
import { AbsoluteFill, Easing, Img, OffthreadVideo, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, F } from "./theme";

/** Fade a whole scene in and out over a few frames. */
export const Fade: React.FC<{ children: React.ReactNode; inF?: number; outF?: number }> = ({ children, inF = 8, outF = 8 }) => {
  const f = useCurrentFrame();
  const { durationInFrames: d } = useVideoConfig();
  const o = Math.min(interpolate(f, [0, inF], [0, 1], { extrapolateRight: "clamp" }), interpolate(f, [d - outF, d], [1, 0], { extrapolateLeft: "clamp" }));
  return <AbsoluteFill style={{ opacity: o }}>{children}</AbsoluteFill>;
};

/** A real screen recording (1600×900) filling the frame, with a slow push-in towards a focus point. */
export const Clip: React.FC<{ src: string; from: number; fx?: number; fy?: number; s0?: number; s1?: number; dim?: number }> = ({
  src, from, fx = 0.5, fy = 0.5, s0 = 1, s1 = 1.08, dim = 0,
}) => {
  const f = useCurrentFrame();
  const { durationInFrames: d, fps } = useVideoConfig();
  const s = interpolate(f, [0, d], [s0, s1], { easing: Easing.inOut(Easing.cubic) });
  return (
    <AbsoluteFill style={{ backgroundColor: C.night, overflow: "hidden" }}>
      <AbsoluteFill style={{ transform: `scale(${s})`, transformOrigin: `${fx * 100}% ${fy * 100}%` }}>
        <OffthreadVideo src={staticFile(`clips/${src}.mp4`)} trimBefore={Math.round(from * fps)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>
      {dim > 0 && <AbsoluteFill style={{ backgroundColor: `rgba(11,20,15,${dim})` }} />}
    </AbsoluteFill>
  );
};

/** Small uppercase chapter label, top-left. */
export const Kicker: React.FC<{ n: string; text: string; dark?: boolean }> = ({ n, text, dark }) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [4, 16], [0, 1], { extrapolateRight: "clamp" });
  return (
    <div style={{ position: "absolute", left: 96, top: 72, opacity: o, display: "flex", gap: 18, alignItems: "center", fontFamily: F.mono, fontSize: 26, letterSpacing: "0.14em", textTransform: "uppercase", color: dark ? C.sprout : C.moss, backgroundColor: dark ? "rgba(11,20,15,.72)" : "rgba(243,239,228,.9)", padding: "12px 22px", borderRadius: 999 }}>
      <span style={{ opacity: 0.7 }}>{n}</span>
      <span>{text}</span>
    </div>
  );
};

export const Pop: React.FC<{ delay?: number; children: React.ReactNode; style?: React.CSSProperties; y?: number }> = ({ delay = 0, children, style, y = 30 }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = spring({ frame: f - delay, fps, config: { damping: 200 }, durationInFrames: 22 });
  return <div style={{ opacity: p, transform: `translateY(${(1 - p) * y}px)`, ...style }}>{children}</div>;
};

/** Horizontal bar chart row. value in percent. */
export const Bar: React.FC<{ label: string; sub?: string; value: number; color: string; delay: number; bold?: boolean; dark?: boolean; suffix?: string }> = ({
  label, sub, value, color, delay, bold, dark, suffix = "%",
}) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = spring({ frame: f - delay, fps, config: { damping: 200 }, durationInFrames: 34 });
  const ink = dark ? C.nightInk : C.ink;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "520px 1fr", alignItems: "center", gap: 36, opacity: interpolate(f - delay, [0, 8], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) }}>
      <div style={{ textAlign: "right" }}>
        <div style={{ fontFamily: F.sans, fontSize: 40, fontWeight: bold ? 700 : 500, color: ink }}>{label}</div>
        {sub && <div style={{ fontFamily: F.mono, fontSize: 24, color: dark ? C.nightMuted : C.muted, marginTop: 4 }}>{sub}</div>}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
        <div style={{ height: 64, width: `${(value / 100) * 900 * p}px`, backgroundColor: color, borderRadius: 10 }} />
        <div style={{ fontFamily: F.serif, fontSize: 60, fontWeight: 600, color: ink, fontVariantNumeric: "tabular-nums" }}>
          {(value * p).toFixed(value % 1 ? 1 : 0)}
          {suffix}
        </div>
      </div>
    </div>
  );
};

export const Tile: React.FC<{ id: string; size: number; style?: React.CSSProperties }> = ({ id, size, style }) => (
  <Img src={staticFile(`tiles/${id}.png`)} style={{ width: size, height: size, borderRadius: 14, objectFit: "cover", imageRendering: "auto", ...style }} />
);

export const Captions: React.FC<{ items: { start: number; text: string; from: number; to: number }[] }> = ({ items }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = f / fps;
  const cur = items.find((c) => t >= c.start + c.from && t < c.start + c.to);
  if (!cur) return null;
  const local = t - (cur.start + cur.from);
  const o = interpolate(local, [0, 0.18], [0, 1], { extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 64, pointerEvents: "none" }}>
      <div style={{ maxWidth: 1560, opacity: o, backgroundColor: "rgba(11,20,15,.86)", color: "#fff", fontFamily: F.sans, fontSize: 44, fontWeight: 500, lineHeight: 1.3, padding: "18px 34px", borderRadius: 18, textAlign: "center", boxShadow: "0 10px 40px rgba(0,0,0,.35)" }}>
        {cur.text}
      </div>
    </AbsoluteFill>
  );
};
