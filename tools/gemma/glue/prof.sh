#!/bin/bash
# tools/gemma/glue/prof.sh <name> <glue_bench args...>: run glue_bench under nsys (graph nodes traced) holding the GPU
# lock, export the sqlite, and print the per-graph summary (graph names from the "graphs:" line glue_bench prints).
set -e -o pipefail
cd "$(dirname "$0")"
mkdir -p prof
name=$1; shift
out=$(flock /root/sg-tools/gpu.lock nsys profile --cuda-graph-trace=node -o prof/$name -f true --stats=false ./glue_bench "$@" 2>&1) || true
echo "$out" | grep -v "^Generat\|nsys-rep\|^\s*$\|Collecting\|Processing\|Creating\|Exporting\|^\[" || true
nsys export -t sqlite -o prof/$name.sqlite -f true prof/$name.nsys-rep >/dev/null 2>&1
names=$(echo "$out" | grep "^graphs: " | tail -1 | sed 's/^graphs: //')
python3 nsys_graphs.py prof/$name.sqlite "$names" $KERNELS
