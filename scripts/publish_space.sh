#!/usr/bin/env bash
# Export the index as a static Hugging Face Space (the browser demo) and upload it.
#
#   scripts/publish_space.sh                     # edition 10 EN, e5-base, <user>/image2vienna
#   scripts/publish_space.sh --repo jaimenms/image2vienna-demo --private
#
# Needs `uv sync --all-extras` and `uv run hf auth login` (or HF_TOKEN) once. Static
# Spaces are free; the repo is created on first run. Re-running is idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

EDITION=latest; MODEL=st:intfloat/multilingual-e5-base; REPO=""; VISIBILITY=--public; MESSAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --edition) EDITION=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --private) VISIBILITY=--private; shift ;;
    --message) MESSAGE=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
USER_NAME=$(uv run hf auth whoami 2>&1 | sed -n 's/^[Uu]ser=//p' | head -1)
[[ -n "$USER_NAME" ]] || { echo "not logged in: run 'uv run hf auth login'" >&2; exit 1; }
REPO=${REPO:-$USER_NAME/image2vienna}
OUT=space/$(basename "$REPO")

uv run i2vienna web-export "$OUT" --edition "$EDITION" --model "$MODEL" --repo-id "$REPO"
BUILT=$(python3 -c "import json;print(json.load(open('$OUT/manifest.json'))['index']['edition'])")
MESSAGE=${MESSAGE:-"image2vienna browser demo: Vienna $BUILT ($(git rev-parse --short HEAD 2>/dev/null || echo uncommitted))"}
uv run hf repos create "$REPO" --repo-type space --space-sdk static $VISIBILITY --exist-ok >/dev/null
# --delete "*" keeps the Space an exact mirror of the export
uv run hf upload "$REPO" "$OUT" . --repo-type space --delete "*" --commit-message "$MESSAGE"
TAG=$(scripts/hf_tag.sh "$REPO" space)
echo "published https://huggingface.co/spaces/$REPO (tag $TAG)"
