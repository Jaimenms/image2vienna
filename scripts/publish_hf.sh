#!/usr/bin/env bash
# Export the index as a Hugging Face model repository (Inference Endpoints handler +
# Parquet tables + vendored package) and upload it.
#
#   scripts/publish_hf.sh                      # EN, edition 10, default model, jaimenms/image2vienna-en, public
#   scripts/publish_hf.sh --private --repo <user>/image2vienna-en
#
# Needs `uv run hf auth login` (or HF_TOKEN) once. Re-running is idempotent: unchanged
# files are skipped by the Hub, changed ones become a new commit. Every upload is
# tagged `v<package version>` (plus `-<n>` if that tag exists): `i2vienna download
# --revision v0.1.0` pins it.
set -euo pipefail
cd "$(dirname "$0")/.."

LANG_CODE=EN; EDITION=latest; MODEL=""; REPO=""; VISIBILITY=--public; MESSAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang) LANG_CODE=$2; shift 2 ;;
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
LANG_LOWER=$(echo "$LANG_CODE" | tr '[:upper:]' '[:lower:]')
REPO=${REPO:-$USER_NAME/image2vienna-$LANG_LOWER}
OUT=hf/image2vienna-$LANG_LOWER
MODEL_ARGS=""; [[ -n "$MODEL" ]] && MODEL_ARGS="--model $MODEL"

# shellcheck disable=SC2086
uv run i2vienna hf-export "$OUT" --edition "$EDITION" --lang "$LANG_CODE" --repo-id "$REPO" $MODEL_ARGS
BUILT=$(python3 -c "import json;print(json.load(open('$OUT/image2vienna.json'))['edition'])")
MESSAGE=${MESSAGE:-"image2vienna Vienna $BUILT $LANG_CODE ($(git rev-parse --short HEAD 2>/dev/null || echo uncommitted))"}
uv run hf repos create "$REPO" --repo-type model $VISIBILITY --exist-ok >/dev/null
uv run hf upload "$REPO" "$OUT" . --repo-type model --commit-message "$MESSAGE"
TAG=$(scripts/hf_tag.sh "$REPO" model)
echo "published https://huggingface.co/$REPO (tag $TAG)"
