# /listen: giving Claude an ear

Date: 2026-10-10
Status: draft, awaiting review

## 1. Intent

Claude can already see video through `/watch`. It cannot hear. Today it guesses at audio from BPM or metadata, which is not good enough for motion design, videography and production work.

`/listen` gives Claude a real listening ability: hand it a track, a voiceover, an SFX clip or a video, and it can describe what it hears, produce a frame-accurate sync map for animation, give production and mix feedback, and suggest edit decisions.

### What the user asked for

- Listen to music and any other audio, not just estimate BPM.
- All four outputs: sync map for motion, a description of what is heard, production and mix feedback, edit decisions for video.
- Hybrid processing: measurement and visuals run locally, a cloud model that hears audio supplies the descriptions.
- A standalone `/listen` skill, plus hooks into `motiondev:motion` (Remotion) and `/watch`.
- Gemini via a free AI Studio API key as the default hearing backend.

### Assumptions

- Runs on the user's Mac (Apple Silicon). ffmpeg, yt-dlp and Python are already installed.
- Single user, local use. No server, no UI.
- Audio inputs are typically under 10 minutes.

### Success criteria

- Beats on 4/4 music land within 1 frame at 60fps (about 17ms) of ground truth.
- Drops land within 2 frames.
- In a blind listening test the user grades Claude's descriptions 4/5 or higher on average.
- Mix flags match what the user actually hears.
- A Remotion clip driven by `cues.json` visibly hits on the beat and on the drop.

## 2. Approach

A Python toolkit that mirrors `/watch`. Each job uses the best specialised tool, signal processing owns precision, and a hearing model owns perception. Claude cross-checks both against spectrogram images it can see.

Rejected:

- Cloud-first (Gemini does everything): audio language models drift by hundreds of milliseconds on timestamps, which breaks motion sync.
- JavaScript in the Remotion ecosystem (essentia.js): weaker beat tracking and no good stem separation.

## 3. Architecture

### 3.1 Pipeline

```
input (file | URL | video)
  │
  0 ingest     yt-dlp / ffmpeg -> master.wav (44.1k stereo) + speech.wav (16k mono)
  1 stems      demucs -> vocals / drums / bass / other
  2 rhythm     beat_this -> beats, downbeats, BPM, meter
               + per-stem onsets (kick, snare, hat from drums stem; bass; vocal)
  3 structure  sections, energy curve, key and scale, moments
               (drops, impacts, risers, silences, vocal in and out, cut and hold points)
  4 mix        integrated and short-term LUFS, true peak, LRA, spectral balance
               per band, stereo width and low-end mono check, clipping
  5 speech     only when the vocals stem contains voice -> mlx-whisper word timestamps
  6 hear       Gemini gets the audio plus the measured skeleton -> structured
               description, section labels, extra moments, mix impressions
  7 see        spectrogram PNGs and stem energy lanes
  8 assemble   report.md + cues.json + curves.json + images/
```

### 3.2 Rules

- **One stage, one module.** Each stage is a single file exposing one function. It reads earlier stages' JSON from the cache directory and writes its own JSON. Stages can be tested and replaced independently.
- **Cache by content hash.** Every input is hashed. Results live in `~/.cache/listen/<hash>/`. Each stage carries a version number; bumping it reruns only that stage and the stages after it.
- **Measured data is the source of truth for numbers and timestamps.** The hearing model supplies words and labels only. It may flag an approximate time; Claude verifies it against measured data and spectrograms.
- **Content type routing.** A quick classifier (stem energy plus voice activity) labels input as `music`, `speech`, `sfx` or `mixed`. Music emphasises rhythm, sections, key, arc and mix. Speech emphasises transcript, pauses, delivery, room tone and noise. SFX emphasises a timestamped event list.

### 3.3 Tools

| Job | Tool |
|---|---|
| Download and decode | yt-dlp, ffmpeg |
| Stem separation | demucs (htdemucs) |
| Beats, downbeats, tempo | beat_this; librosa as fallback |
| Onsets, structure, energy, key | librosa |
| Loudness | pyloudnorm (ITU-R BS.1770) |
| Speech | mlx-whisper (Apple Silicon, local) |
| Hearing and description | Gemini via pluggable backend (section 6) |
| Plots | matplotlib |
| Environment | uv, pinned to Python 3.12 |

### 3.4 Command line

```
listen <file|url|video> [question]
  --fast            skip stems and the hearing model; quick sync map only
  --start T --end T analyse one section (SS, MM:SS or HH:MM:SS)
  --fps N           frame rate for frame numbers (default 60)
  --ref FILE        run the same measurements on a reference track for comparison
  --no-cloud        never upload audio; measurements and images only
  --stages a,b      rerun only these stages
  --export DIR      copy cues.json and curves.json into DIR (for example a Remotion public/ folder)
```

### 3.5 Code layout

```
cid/ear/
  listen.py            CLI entry point
  setup.py             preflight (--check, exit codes) and installer
  stages/
    ingest.py stems.py rhythm.py structure.py mix.py
    speech.py hear.py see.py assemble.py classify.py
  backends/
    aistudio.py
  remotion/
    cues.ts            useCues, beatPulse, curve, momentAt
  SKILL.md
  tests/
  docs/specs/
  pyproject.toml       uv-managed, Python 3.12
```

## 4. Outputs

### 4.1 cues.json (events, for code)

Every event carries `t` (seconds) and `f` (frame at the target fps). Strengths are normalised 0 to 1.

```json
{
  "version": 1,
  "source": { "file": "track.wav", "hash": "a3f9...", "duration": 184.32, "fps": 60, "kind": "music" },
  "tempo": { "bpm": 124.0, "confidence": 0.94, "meter": "4/4", "changes": [] },
  "key": { "tonic": "A", "mode": "minor", "confidence": 0.81 },
  "beats": [
    { "t": 0.482, "f": 29, "bar": 1, "beat": 1, "strength": 0.91 }
  ],
  "hits": {
    "kick":  [{ "t": 0.482, "f": 29, "v": 0.88 }],
    "snare": [], "hat": [], "bass": [], "vocal": []
  },
  "sections": [
    { "id": "s1", "label": "intro", "start": 0.0, "end": 15.48, "f0": 0, "f1": 929,
      "energy": 0.22, "label_source": "gemini" }
  ],
  "moments": [
    { "t": 30.97, "f": 1858, "type": "drop", "strength": 0.97, "note": "full kit and sub enter" },
    { "t": 23.20, "f": 1392, "type": "riser", "end": 30.90, "strength": 0.70 }
  ],
  "speech": [
    { "t0": 61.0, "t1": 63.4, "text": "Finally home",
      "words": [{ "t0": 61.0, "t1": 61.5, "w": "Finally" }] }
  ],
  "mix": { "lufs": -9.1, "true_peak": -0.3, "lra": 5.2, "clipped_samples": 0,
           "flags": ["low-mid build-up 200 to 400 Hz in drop"] }
}
```

Moment types: `drop`, `impact`, `riser`, `silence`, `vocal_in`, `vocal_out`, `cut`, `hold`. Cut points are ranked by downbeat, phrase boundary and energy change.

### 4.2 curves.json (continuous, for code)

One value per frame at the target fps, normalised 0 to 1: `energy`, `loudness`, `bands.low`, `bands.mid`, `bands.high`, `stems.vocals`, `stems.drums`, `stems.bass`, `stems.other`. Kept separate from `cues.json` so the event file stays readable.

### 4.3 Remotion helper (remotion/cues.ts)

```ts
const { beats, moments, curve } = useCues(cues, curves);
const pulse  = beatPulse(frame, beats, { decay: 8 });  // 1 on the beat, eases to 0
const bass   = curve("bands.low", frame);              // 0..1 for this frame
const isDrop = momentAt(frame, moments, "drop", 2);    // within 2 frames
```

### 4.4 report.md (for Claude)

Target under 2,000 tokens. Sections in order:

1. Header: file, duration, kind, BPM and meter, key with confidence.
2. What it sounds like (heard, from Gemini).
3. Timeline table: section, time range, energy sparkline, what is happening (measured plus heard).
4. Key moments and best cut points.
5. Mix: LUFS, true peak, LRA, flags. With `--ref`, the deltas against the reference.
6. Speech transcript, if any.
7. Conflicts: every place where the hearing model and measurements disagree beyond threshold, with the image to check.
8. Image index.

Each stage that failed appears as `unavailable: <reason>`.

### 4.5 images/

- `spec_full.png`: wide mel spectrogram, log frequency axis labelled in Hz, time in mm:ss, section boundaries as lines, downbeats as ticks.
- `spec_s<n>.png`: zoom per section, up to 8 by default.
- `stems.png`: four stacked energy lanes on a shared time axis.
- `ref.png`: spectral balance overlay against the reference, only with `--ref`.
- Default 6 to 12 images, hard cap 16.

## 5. How Claude listens (SKILL.md behaviour)

1. Run preflight. Silent on success.
2. Run `listen.py`.
3. Read `report.md`.
4. Look at `spec_full.png`.
5. Open zoomed images only where Conflicts list something or where the user's question points.
6. Answer with timestamps, and say which claims are measured and which are heard.
7. Follow-up questions on the same track are answered from the cache and images, never by a new hearing-model call.

## 6. Hearing backend

### 6.1 Backends

Selected by `LISTEN_BACKEND` in `~/.config/listen/.env`.

| Backend | Auth | Status |
|---|---|---|
| `aistudio` | `GEMINI_API_KEY` from Google AI Studio, free tier by default | default |
| none (`--no-cloud`) | none | always available |

Data note: on the free tier Google may use submitted content to improve its products and humans may review it. Unreleased client audio should go through a billing-enabled (paid) project or use `--no-cloud`. The skill states this the first time it uploads.

### 6.2 Request discipline (free tier)

The free tier has a per-day request limit and a per-minute token limit. The hearing stage is designed to spend as little of both as possible.

- **Exactly one request per track** (per chunk for inputs over 10 minutes). No conversational follow-ups.
- **Measured skeleton in the prompt.** BPM, meter, key, section boundaries and moments are given, so the model describes rather than searches.
- **Structured output.** The request sets a JSON response schema. The prompt is fixed, short and versioned; no free-form essay.
- **Small upload.** Audio is downmixed to mono and compressed before upload. Token cost depends on duration, so `--start/--end` is the main lever.
- **Cache first.** A cached hearing result for the same hash and prompt version is never re-requested.
- **Rate limits.** HTTP 429 triggers exponential backoff with a cap, then the stage is marked unavailable rather than failing the run.
- **Usage counter.** A local daily counter in `~/.cache/listen/usage.json` warns before the configured daily cap.
- **Chunking.** Inputs over 10 minutes are split into overlapping chunks; results are merged by time.

### 6.3 Hearing response schema

```json
{
  "kind": "",
  "overall": { "genre": "", "mood": "", "instruments": [], "vocals": "", "production_style": "" },
  "sections": [{ "id": "s1", "label": "", "description": "", "what_changes": "" }],
  "extra_moments": [{ "approx_t": 0.0, "what": "" }],
  "mix_impressions": [],
  "speech": { "tone": "", "emotion": "", "room": "", "noise": "" }
}
```

`speech` is filled only for `speech` or `mixed` inputs.
`kind` is the model's own call (music, speech, sfx or mixed); a disagreement with the measured kind is listed under Conflicts.

## 7. Hooks

- **/watch:** when a video has an audio track, `/watch` calls `listen` on it with `--fps` set to the video frame rate, so Claude sees and hears the video together.
- **motiondev:motion:** imports `cues.json` and `curves.json` through `remotion/cues.ts`. The motion skill reads the cue file instead of guessing timing from BPM.

## 8. Error handling

- Each stage fails independently. The report marks it `unavailable: <reason>`; nothing is guessed or faked.
- Preflight `setup.py --check` with exit codes in the style of `/watch`: missing binaries, missing Python environment, missing key. Installer is idempotent.
- No audio stream in a video: clear message and exit.
- Clips under 3 seconds or silent input: rhythm and structure are skipped.
- Inputs over 20 minutes: warning suggesting `--start/--end`.
- Low beat confidence (ambient, rubato, live playing): beats are marked low confidence and moments fall back to the energy curve.
- Stage version bump invalidates only that stage and its dependants.

## 9. Testing

### 9.1 Automated (offline, deterministic)

Each stage is tested on audio synthesised inside the tests:

- Click track at a known BPM: detected beats within plus or minus 10ms.
- Pink noise at a known LUFS: measured loudness within plus or minus 0.5 LU.
- Sine tones and sweeps: energy lands in the right bands.
- Inserted silence gaps: `silence` moments detected at the right times.
- Deliberately clipped file: clipping flagged.
- Hearing stage: backend mocked; schema validation and conflict detection tested.
- Golden tests on 3 to 5 real tracks with hand-annotated cues.
- One live hearing smoke test, opt-in only, to protect the free quota.

### 9.2 Acceptance (proving Claude can hear)

1. **Blind listening test.** The user picks about 6 clips they know well: 2 music tracks in different genres, 1 voiceover or dialogue, 1 SFX or ambience, 1 video with music and voice, 1 mix they worked on plus its reference. Claude runs `/listen` on each with no other context and answers a fixed question set. The user grades each answer 1 to 5.
2. **Sync test.** A Remotion clip flashes on every beat and hits on the drop, rendered with the original audio. The user watches it; Claude checks it with `/watch`.

Pass bar: the success criteria in section 1.

## 10. Out of scope

- Real-time or live microphone listening.
- Generating or editing audio (mixing, mastering, stem export as a product).
- A GUI.
- Music identification (naming the song or artist).
- Platforms other than macOS on Apple Silicon.

## 11. Build-time spike: result

Tested 2026-10-10. Goal: use a personal Google account instead of an API key for hearing.

- **Antigravity (agy):** not usable. Its file viewer turns audio into a text transcript before the model sees it. It returned the spoken words of a test clip but could not hear voice, tone or anything non-verbal, so it cannot describe music. Words are already covered locally by mlx-whisper.
- **Gemini CLI:** not tested. Sign-in needs an interactive browser flow, and the AI Studio key already works. Skipped by decision.

Result: `aistudio` is the only hearing backend. If the free tier gets tight, enable billing on the same key rather than adding a backend.
