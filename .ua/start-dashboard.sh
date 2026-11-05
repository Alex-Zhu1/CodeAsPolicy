#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
plugin_dir="/home/mhzhu/.understand-anything-plugin"
node_bin="/home/mhzhu/.local/bin/node"
if [[ ! -s "$project_dir/.ua/knowledge-graph.json" ]]; then
  printf '%s\n' '尚未生成知识图谱，请先运行 understand。' >&2
  exit 1
fi
cd "$plugin_dir/packages/dashboard"
export GRAPH_DIR="$project_dir"
export BROWSER=none
exec "$node_bin" node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173
