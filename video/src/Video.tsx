import React from "react";
import { AbsoluteFill, Audio, Sequence, Series, interpolate, staticFile } from "remotion";
import { Captions } from "./parts";
import { Baseline, End, Heat, Hook, How, Idea, Limits, Next, Problem, Search, Test } from "./Scenes";
import { FPS, N, loadFonts } from "./theme";

const SCENE: Record<string, React.FC> = { hook: Hook, problem: Problem, idea: Idea, search: Search, heat: Heat, how: How, test: Test, baseline: Baseline, limits: Limits, next: Next, end: End };

// Scene order, lengths and caption text all come from numbers.json (video/scripts/make_numbers.py).
export const PlatonicEarth: React.FC = () => {
  loadFonts();
  const caps = N.timeline.flatMap((sc) => sc.captions.map((c) => ({ ...c, start: sc.start })));
  return (
    <AbsoluteFill style={{ backgroundColor: "#0b140f" }}>
      <Series>
        {N.timeline.map((sc) => {
          const S = SCENE[sc.key];
          return (
            <Series.Sequence key={sc.key} name={sc.key} durationInFrames={Math.round(sc.dur * FPS)}>
              <S />
            </Series.Sequence>
          );
        })}
      </Series>
      <Captions items={caps} />
      {/* quiet music under everything; it dips further while someone is speaking */}
      <Audio src={staticFile("audio/music.wav")} volume={(f) => {
        const t = f / FPS, end = N.total_s;
        const speaking = N.timeline.some((sc) => sc.voices.length > 0 && t >= sc.start && t < sc.start + sc.dur);
        const fade = Math.min(interpolate(t, [0, 2], [0, 1], { extrapolateRight: "clamp" }), interpolate(t, [end - 3, end], [1, 0], { extrapolateLeft: "clamp" }));
        return (speaking ? 0.14 : 0.45) * fade;
      }} />
      {/* one voice piece per caption line, each starting on its caption's beat (video/scripts/cut_voice.py) */}
      {N.timeline.flatMap((sc) => (sc.voices as { src: string; at: number }[]).map((v, i) => (
        <Sequence key={`v-${sc.key}-${i}`} name={`voice ${sc.key} ${i}`} from={Math.round((sc.start + v.at) * FPS)} layout="none">
          <Audio src={staticFile(v.src)} />
        </Sequence>
      )))}
    </AbsoluteFill>
  );
};
