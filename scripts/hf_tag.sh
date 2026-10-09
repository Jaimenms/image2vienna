#!/usr/bin/env bash
# Tag the current revision of a Hub repo with the package version: v0.1.0, then
# v0.1.0-2, v0.1.0-3 ... when that version was already tagged. Prints the tag.
#   scripts/hf_tag.sh jaimenms/image2vienna-en model
#   scripts/hf_tag.sh jaimenms/image2vienna space
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$1; TYPE=${2:-model}
VERSION=$(uv run python -c "import image2vienna; print(image2vienna.__version__)")
EXISTING=$(uv run hf repos tag list "$REPO" --repo-type "$TYPE" --format json 2>/dev/null | python3 -c "
import sys, json
try: data = json.load(sys.stdin)
except Exception: data = []
tags = [t.get('name', t) if isinstance(t, dict) else t for t in (data if isinstance(data, list) else data.get('tags', []))]
print(' '.join(str(t) for t in tags))" || true)
TAG="v$VERSION"; N=1
while [[ " $EXISTING " == *" $TAG "* ]]; do N=$((N + 1)); TAG="v$VERSION-$N"; done
uv run hf repos tag create "$REPO" "$TAG" --repo-type "$TYPE" -m "image2vienna $VERSION ($(git rev-parse --short HEAD 2>/dev/null || echo uncommitted))" >/dev/null
echo "$TAG"
