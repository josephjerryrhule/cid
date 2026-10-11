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

One command. Paste it into Terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/josephjerryrhule/cid/master/install.sh | bash
```

It installs what is missing (Homebrew, ffmpeg, yt-dlp, uv), builds the Python environment, links the
skill into Claude Code as `/listen`, asks once for a free Gemini key (press Enter to skip), and downloads
the models (about 3 GB, once) so the first run is fast. It is safe to run again; that is also how you update.

Then, in Claude Code: `/listen <audio, video or URL>`.

Already cloned? Run `./install.sh` from the repo. Flags: `--no-models` (download on first use instead),
`--yes` (no prompts), `--dir PATH` (where to clone; default `~/.local/share/cid`).

The Gemini key is optional. Without it everything is measured locally; add `--no-cloud` or put
`GEMINI_API_KEY=...` in `~/.config/listen/.env` later (free key: aistudio.google.com/apikey).

### Use

```bash
uv run python listen.py track.mp3                  # full analysis
uv run python listen.py track.mp3 --fast           # sync map only, no upload
uv run python listen.py mix.wav --ref reference.wav
uv run python listen.py clip.mp4 --fps 30 --export my-remotion-project/public
```

Inside Claude Code: `/listen <file or url> [question]`.

Audio is sent to Google only for the hearing step. Use `--no-cloud` for audio that must stay local.
