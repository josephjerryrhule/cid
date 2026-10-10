import { Composition } from "remotion";
import cues from "../public/cues.json";
import { SyncTest } from "./SyncTest";

export const Root = () => (
  <Composition
    id="SyncTest"
    component={SyncTest}
    fps={cues.source.fps}
    width={1080}
    height={1080}
    durationInFrames={Math.ceil(cues.source.duration * cues.source.fps)}
  />
);
