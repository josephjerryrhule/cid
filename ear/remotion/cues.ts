// React entry for Remotion projects. Copy cues.ts and cues-core.ts into your project.
import { useMemo } from "react";
import { curveAt, type Cues, type Curves } from "./cues-core";

export * from "./cues-core";

export function useCues(cues: Cues, curves?: Curves) {
  return useMemo(
    () => ({
      beats: cues.beats,
      hits: cues.hits,
      sections: cues.sections,
      moments: cues.moments,
      curve: (name: string, frame: number) => (curves ? curveAt(curves, name, frame) : 0),
    }),
    [cues, curves],
  );
}
