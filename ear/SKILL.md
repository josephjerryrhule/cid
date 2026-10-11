---
name: listen
description: Listen to audio (music, voice, SFX, or a video's soundtrack) from a file or URL. Measures beats, sections, drops, loudness and mix balance, separates stems, transcribes speech, renders spectrograms Claude can see, and asks a hearing model to describe the sound. Writes cues.json and curves.json for Remotion and motion sync. Use for "listen to this", "what does this track sound like", "sync this to the music", "where should I cut", "check my mix", or whenever a video's audio matters.
argument-hint: "<audio-or-video-path-or-url> [question] [--fast] [--start T --end T] [--fps N] [--ref FILE] [--no-cloud] [--export DIR]"
allowed-tools: Bash, Read, AskUserQuestion
user-invocable: true
---

# /listen: Claude listens

You have no audio input. This skill gives you one. A pipeline measures the audio, renders images of it, and asks a hearing model to describe it. You read the report, look at the images, and answer.

## Step 0: preflight (silent on success)

```bash
python3 "${CLAUDE_SKILL_DIR}/setup.py" --check
```

Exit 0 prints nothing: continue without comment. Otherwise:

| Exit | Meaning | Action |
|---|---|---|
| 2 | missing ffmpeg, ffprobe, yt-dlp or uv | run `python3 "${CLAUDE_SKILL_DIR}/setup.py"` |
| 3 | no GEMINI_API_KEY | ask the user to add a free key from aistudio.google.com/apikey to `~/.config/listen/.env` themselves, or continue with `--no-cloud` |
| 4 | both | run the installer, then handle the key as in 3 |
| 5 | Python environment missing | run `python3 "${CLAUDE_SKILL_DIR}/setup.py"` |

Skip Step 0 on later /listen calls in the same session once it returned 0.

## Step 1: run

```bash
uv run --project "${CLAUDE_SKILL_DIR}" python "${CLAUDE_SKILL_DIR}/listen.py" "<source>" [flags]
```

Choose flags:
- Only timing needed (a sync map for motion): `--fast` (no stems, no hearing request).
- The user cares about one part: `--start` and `--end`.
- Animation at a frame rate other than 60: `--fps N`. For a video, use the video's own frame rate.
- Mix feedback against a reference: `--ref <file or url>`.
- Unreleased client audio or the user asks for privacy: `--no-cloud`. If audio looks like unreleased client work and the user has not said, ask once.
- Remotion project: `--export <project>/public`.

The command prints the stage status, the paths, then the full report.

## Step 2: read, then look

1. Read the printed report.
2. Read `spec_full.png` (its path is under Images).
3. Open a `spec_sN.png` only when Conflicts point at it or the question is about that section. Open `stems.png` for arrangement questions, `ref.png` for reference comparisons, and `notes.png` for melody questions.

## Step 3: answer

- Give times as m:ss.xx and say which claims are measured and which are heard.
- Resolve every Conflict before relying on it: trust measured numbers for timing; check the image for anything else.
- If the report has a "Note:" line about Google AI Studio, pass it on once.
- Follow-up questions about the same audio: answer from the report, cues.json and the images. Never rerun the hearing stage for a follow-up. Rerun with `--stages <name>` only if a stage changed.

## Motion work

`cues.json` has beats, hits, sections and moments with `t` (seconds) and `f` (frame). `curves.json` has one 0..1 value per frame for `energy`, `loudness`, `bands.low|mid|high` and `stems.vocals|drums|bass|other`. In a Remotion project, copy `${CLAUDE_SKILL_DIR}/remotion/cues.ts` and `cues-core.ts` into the project and use:

`cues.json` also has `notes.{vocals,bass,other}` (every melody note with frames) and `tags` (instruments, voices and SFX with start and end frames); `curves.json` has `tags.<label>`. Use `noteAt(frame, cues.notes.other)` to hit animation on each note and `tagActive(frame, cues.tags, "Electric guitar")` to drive layers by instrument.

```ts
const { beats, moments, curve } = useCues(cues, curves);
const p = beatPulse(frame, beats, { decay: 8 });
const bass = curve("bands.low", frame);
const isDrop = momentAt(frame, moments, "drop", 2);
```

Never guess timing from BPM when cues.json exists.

## With /watch

When watching a video whose sound matters, run /listen on the same source with `--fps` set to the video's frame rate, then combine what you see in the frames with what the report says you hear.
