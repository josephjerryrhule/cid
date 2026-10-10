import { AbsoluteFill, Audio, staticFile, useCurrentFrame } from "remotion";
import cuesJson from "../public/cues.json";
import curvesJson from "../public/curves.json";
import { beatPulse, momentAt, useCues, type Cues, type Curves } from "./lib/cues";

export const SyncTest = () => {
  const frame = useCurrentFrame();
  const { beats, moments, curve } = useCues(cuesJson as unknown as Cues, curvesJson as unknown as Curves);
  const pulse = beatPulse(frame, beats, { decay: 6 });
  const down = beatPulse(frame, beats, { decay: 10, downbeatOnly: true });
  // Fire on the drop frame and hold 4 frames after it; a symmetric window would flash before the sound.
  const drop = momentAt(frame - 2, moments, "drop", 2);
  const bass = curve("bands.low", frame);
  const size = 300 + 400 * bass;
  return (
    <AbsoluteFill style={{ background: drop ? "#f2f2f2" : "#0b0b0f", alignItems: "center", justifyContent: "center" }}>
      <Audio src={staticFile("audio.wav")} />
      <div
        style={{
          width: size,
          height: size,
          borderRadius: "50%",
          background: `rgba(232, 89, 12, ${0.25 + 0.75 * pulse})`,
          border: `${2 + 18 * down}px solid #f2f2f2`,
        }}
      />
      <div style={{ position: "absolute", bottom: 60, color: "#7fd4ff", fontFamily: "monospace", fontSize: 36 }}>
        {`frame ${frame}`}
      </div>
    </AbsoluteFill>
  );
};
