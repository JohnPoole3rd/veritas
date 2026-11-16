import { Composition } from "remotion";
import { PlatonicEarth } from "./Video";
import { FPS, H, N, W } from "./theme";

export const RemotionRoot: React.FC = () => (
  <Composition id="PlatonicEarth" component={PlatonicEarth} durationInFrames={Math.round(N.total_s * FPS)} fps={FPS} width={W} height={H} />
);
