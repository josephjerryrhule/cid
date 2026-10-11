# cid

Experiments in giving Claude new senses and skills for creative production work.
Each idea lives in its own folder.

## ear: `/listen`

Claude can see video through `/watch`. `ear/` gives it an ear.

`/listen` takes an audio file, a URL or a video and:

- measures beats, downbeats, tempo, key, sections, drops, risers, silences, cut points and per-instrument hits
- separates stems (demucs) and transcribes speech locally (mlx-whisper)
- measures loudness, true peak, dynamics, spectral balance and stereo image, optionally against a reference track
- transcribes melody to notes per stem (basic-pitch): bass lines, sung vocals, and the hook
- tags what is playing when, per stem and in the mix (PANNs, AudioSet classes): instruments, voices, SFX
- renders spectrograms and stem lanes that Claude reads as images
- asks a hearing model (Gemini) to describe the sound, once per track, with the measured facts as a skeleton
- writes `cues.json` and `curves.json` with frame numbers, plus a Remotion helper, so animation lands on the beat

Measured numbers own the timing. The hearing model owns the words. Where they disagree, the report says so.

Design: [`ear/docs/specs/2026-10-10-listen-design.md`](ear/docs/specs/2026-10-10-listen-design.md).
Build plan: [`ear/docs/plans/2026-10-10-listen.md`](ear/docs/plans/2026-10-10-listen.md).

### Setup (macOS, Apple Silicon)

```bash
cd ear
python3 setup.py            # installs ffmpeg, yt-dlp, uv if missing; builds the Python 3.12 env
# add GEMINI_API_KEY=... to ~/.config/listen/.env (free key: aistudio.google.com/apikey)
ln -s "$PWD" ~/.claude/skills/listen
```

### Use

```bash
uv run python listen.py track.mp3                  # full analysis
uv run python listen.py track.mp3 --fast           # sync map only, no upload
uv run python listen.py mix.wav --ref reference.wav
uv run python listen.py clip.mp4 --fps 30 --export my-remotion-project/public
```

Inside Claude Code: `/listen <file or url> [question]`.

Audio is sent to Google only for the hearing step. Use `--no-cloud` for audio that must stay local.
