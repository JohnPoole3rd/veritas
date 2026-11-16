import { continueRender, delayRender, staticFile } from "remotion";
import data from "./numbers.json";

export const N = data;
export const FPS = 30;
export const W = 1920;
export const H = 1080;

export const C = {
  paper: "#f3efe4",
  paper2: "#ebe5d4",
  card: "#faf8f1",
  ink: "#13251a",
  ink2: "#33473a",
  muted: "#677568",
  moss: "#2f6b35",
  fern: "#7aa95a",
  sprout: "#d9ec7c",
  clay: "#c2532c",
  ember: "#9f2f1f",
  ochre: "#d39b35",
  river: "#2c6d8c",
  night: "#0b140f",
  night2: "#122019",
  nightInk: "#e8efdf",
  nightMuted: "#93a596",
};
export const F = {
  serif: '"Fraunces", Georgia, serif',
  sans: '"Inter", system-ui, sans-serif',
  mono: '"JetBrains Mono", Menlo, monospace',
};

const FONTS: [string, string, Record<string, string>][] = [
  ["Fraunces", "fonts/fraunces-latin.woff2", { weight: "100 900" }],
  ["Fraunces", "fonts/fraunces-italic-latin.woff2", { weight: "100 900", style: "italic" }],
  ["Inter", "fonts/inter-latin.woff2", { weight: "100 900" }],
  ["JetBrains Mono", "fonts/jetbrains-mono-latin.woff2", { weight: "100 800" }],
];
let loaded = false;
export function loadFonts() {
  if (loaded || typeof document === "undefined") return;
  loaded = true;
  const h = delayRender("fonts");
  Promise.all(
    FONTS.map(([fam, file, desc]) => {
      const f = new FontFace(fam, `url(${staticFile(file)}) format("woff2")`, desc as FontFaceDescriptors);
      return f.load().then((x) => document.fonts.add(x));
    }),
  ).then(() => continueRender(h), () => continueRender(h));
}

export const fmt1 = (v: number) => `${Number(v.toFixed(1))}`;
export const signed = (v: number) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(2)}`;
