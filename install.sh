#!/usr/bin/env bash
# One-command setup for /listen (Claude's ear).
#
#   curl -fsSL https://raw.githubusercontent.com/josephjerryrhule/cid/master/install.sh | bash
#
# or, from a checkout:  ./install.sh
#
# What it does, and it is safe to run again:
#   1. installs Homebrew if missing, then ffmpeg, yt-dlp and uv
#   2. gets the code (clones to ~/.local/share/cid, or uses the checkout you ran it from)
#   3. builds the Python environment
#   4. links the skill into Claude Code as /listen
#   5. asks once for a free Gemini key (press Enter to skip; /listen still works with --no-cloud)
#   6. downloads the models so the first real run is fast
#
# Flags: --no-models (skip step 6), --yes (never prompt), --dir PATH (where to clone)
set -euo pipefail

REPO="https://github.com/josephjerryrhule/cid.git"
DIR="${CID_DIR:-$HOME/.local/share/cid}"
MODELS=1
ASK=1
while [ $# -gt 0 ]; do
  case "$1" in
    --no-models) MODELS=0 ;;
    --yes) ASK=0 ;;
    --dir) DIR="$2"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  ok  %s\n' "$*"; }
fail() { printf '\n  %s\n' "$*" >&2; exit 1; }

# 0. platform: the local speech model (mlx-whisper) needs Apple Silicon
[ "$(uname -s)" = "Darwin" ] || fail "/listen needs macOS on Apple Silicon (M1 or later)."
[ "$(uname -m)" = "arm64" ] || fail "/listen needs Apple Silicon (M1 or later); this Mac is $(uname -m)."

# 1. tools
say "1/6  Tools"
if ! command -v brew >/dev/null 2>&1; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$("$b" shellenv)"; done
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "  Installing Homebrew (it may ask for your Mac password)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/tty
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
need=()
for t in ffmpeg yt-dlp uv git; do command -v "$t" >/dev/null 2>&1 || need+=("$t"); done
if [ ${#need[@]} -gt 0 ]; then brew install "${need[@]}"; fi
ok "ffmpeg, yt-dlp, uv, git"

# 2. code: use this checkout if the script lives in one, otherwise clone or update
say "2/6  Code"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$HERE" ] && [ -f "$HERE/ear/listen.py" ]; then
  DIR="$HERE"
elif [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only --quiet
else
  mkdir -p "$(dirname "$DIR")"
  git clone --quiet "$REPO" "$DIR"
fi
EAR="$DIR/ear"
ok "$EAR"

# 3. Python environment (uv fetches Python 3.12 by itself)
say "3/6  Python environment (a few minutes the first time)"
python3 "$EAR/setup.py" >/dev/null 2>&1 || true
[ -f "$EAR/.venv/.listen-ready" ] || fail "The Python environment did not build. Run: cd \"$EAR\" && uv sync"
ok "built"

# 4. skill link
say "4/6  Claude Code skill"
SKILLS="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"
mkdir -p "$SKILLS"
if [ -e "$SKILLS/listen" ] && [ ! -L "$SKILLS/listen" ]; then
  mv "$SKILLS/listen" "$SKILLS/listen.backup.$(date +%s)"
  echo "  moved your old ~/.claude/skills/listen aside"
fi
ln -sfn "$EAR" "$SKILLS/listen"
ok "/listen -> $EAR"

# 5. hearing key (optional)
say "5/6  Hearing model key (optional)"
ENVF="${LISTEN_CONFIG:-$HOME/.config/listen}/.env"
if grep -qE '^GEMINI_API_KEY=.+' "$ENVF" 2>/dev/null || [ -n "${GEMINI_API_KEY:-}" ]; then
  ok "key already set"
elif [ "$ASK" = 1 ] && [ -r /dev/tty ]; then
  echo "  /listen measures everything locally. A free Gemini key adds a spoken description of the sound."
  echo "  Get one at https://aistudio.google.com/apikey, paste it here, or press Enter to skip."
  printf '  key: '
  read -r -s KEY </dev/tty || KEY=""
  echo
  if [ -n "$KEY" ]; then
    printf 'GEMINI_API_KEY=%s\n' "$KEY" >>"$ENVF"
    chmod 600 "$ENVF"
    ok "saved to $ENVF"
  else
    ok "skipped (use --no-cloud, or add GEMINI_API_KEY to $ENVF later)"
  fi
else
  ok "skipped (add GEMINI_API_KEY to $ENVF any time)"
fi

# 6. models
say "6/6  Models"
if [ "$MODELS" = 1 ]; then
  echo "  Downloading beat, speech, stem, melody and tagging models (about 3 GB, once)..."
  python3 "$EAR/setup.py" --prefetch >/dev/null 2>&1 || true
  ok "downloaded"
else
  ok "skipped (they download on first use)"
fi

if python3 "$EAR/setup.py" --check >/dev/null 2>&1; then
  say "Ready. In Claude Code, type:  /listen <audio, video or URL>"
else
  code=$?
  if [ "$code" = 3 ]; then
    say "Ready. In Claude Code, type:  /listen <audio, video or URL> --no-cloud   (or add a Gemini key for the spoken description)"
  else
    python3 "$EAR/setup.py" --check || true
    fail "Something is still missing; the line above says what."
  fi
fi
