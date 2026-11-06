#!/usr/bin/env bash
# Rebuild demo MP4s from base64 part files in this directory.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if ls tiny_demo.mp4.b64.part* >/dev/null 2>&1; then
  cat tiny_demo.mp4.b64.part* | base64 -d > tiny_demo.mp4
  echo "Wrote tiny_demo.mp4 ($(wc -c < tiny_demo.mp4) bytes)"
fi

if ls rabbit_streaming_converter.mp4.b64.part* >/dev/null 2>&1; then
  cat rabbit_streaming_converter.mp4.b64.part* | base64 -d > rabbit_streaming_converter.mp4
  echo "Wrote rabbit_streaming_converter.mp4 ($(wc -c < rabbit_streaming_converter.mp4) bytes)"
fi

# Single-file base64 fallbacks
if [[ -f tiny_demo.mp4.b64 && ! -f tiny_demo.mp4 ]]; then
  base64 -d tiny_demo.mp4.b64 > tiny_demo.mp4
  echo "Wrote tiny_demo.mp4 from tiny_demo.mp4.b64"
fi
