#!/usr/bin/env bash
# Minimum setup: tangram-figure task with the Seed-X reasoning translator in the middle.
#   receiver sees the translation + the model's reasoning (--explain cot)
#   add more explanations with --explain, e.g. ./play.sh --explain cot,backtranslation
#
#   ./play.sh                # GPU 0, port 8000
#   GPU=1 PORT=8001 ./play.sh
#   SET=abstract ./play.sh   # tangram sets: classic, abstract (tangram_set.json)
#
# Shared machine: check nvidia-smi first and pick a free GPU (Seed-X needs ~16GB).
# The server listens on all of the machine's addresses; use the one for the
# network you are on (WireGuard on boa: 192.168.50.7).
set -euo pipefail
cd "$(dirname "$0")"
GPU=${GPU:-0}
PORT=${PORT:-8000}
ROOM=${ROOM:-tangram-figure}

HOST=${HOST:-$(ip -4 -brief addr show wg0 2>/dev/null | awk '{print $3}' | cut -d/ -f1)}
HOST=${HOST:-localhost}
echo "Dev view: http://$HOST:$PORT/dev?room=$ROOM&dlang=en&mlang=zh"
echo "Director: http://$HOST:$PORT/?room=$ROOM&role=director&lang=en"
echo "Matcher:  http://$HOST:$PORT/?room=$ROOM&role=matcher&lang=zh"
echo "(change lang= to the player's language: en zh es fr de ja ko)"
echo "Logs: logs/$ROOM.jsonl"

CUDA_VISIBLE_DEVICES=$GPU exec uv run python server.py \
  --model seed-x --explain cot --set "${SET:-classic}" --port "$PORT" "$@"
