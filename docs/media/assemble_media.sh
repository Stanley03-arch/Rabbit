#!/usr/bin/env bash
# Reconstruct docs/media/rabbit_streaming_converter.mp4 from base64 parts.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
if [[ ! -f rabbit_streaming_converter.mp4.b64.part0 ]]; then
  echo "Missing base64 parts in $DIR" >&2
  exit 1
fi
cat rabbit_streaming_converter.mp4.b64.part0 \
    rabbit_streaming_converter.mp4.b64.part1 \
    rabbit_streaming_converter.mp4.b64.part2 \
    rabbit_streaming_converter.mp4.b64.part3 \
  | base64 -d > rabbit_streaming_converter.mp4
echo "Wrote $DIR/rabbit_streaming_converter.mp4 ($(wc -c < rabbit_streaming_converter.mp4) bytes)"
file rabbit_streaming_converter.mp4 2>/dev/null || true
