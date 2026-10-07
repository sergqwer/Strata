#!/usr/bin/env python3
"""routing-26b.json (per layer, the routing mass of the original 128 experts) -> the 96 kept experts' masses of one
layer (the most-routed 96, in ascending original index, as lora/prune_gguf.py keeps them), one float per line: the
harness's skewed routing for the prompt path.
  python3 routing_masses.py /root/sg-tools/routing-26b.json <layer> [keep=96] > masses.txt"""
import json, sys
m = json.load(open(sys.argv[1]))[int(sys.argv[2])]
keep = int(sys.argv[3]) if len(sys.argv) > 3 else 96
top = sorted(sorted(range(len(m)), key=lambda e: -m[e])[:keep])
tot = sum(m[e] for e in top)
for e in top:
    print(f"{m[e] / tot:.9g}")
