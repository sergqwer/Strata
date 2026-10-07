#!/usr/bin/env python3
"""Per-graph timing from an nsys run of glue_bench (nsys profile --cuda-graph-trace=node, nsys export -t sqlite):
graphs are named in creation order by the names glue_bench printed ("graphs: a,b,c"); for each graph, the median
and min over its replays of (a) the summed kernel time and (b) the span from its first kernel's start to its last
kernel's end, and the per-kernel-name medians per replay.
  python3 nsys_graphs.py run.sqlite name1,name2,... [--kernels]"""
import sqlite3, sys, statistics as st
from collections import defaultdict
db = sqlite3.connect(sys.argv[1])
names = sys.argv[2].split(",")
rows = db.execute("""select k.start, k.end, s.value, k.graphId, k.gridX, k.gridY from CUPTI_ACTIVITY_KIND_KERNEL k
                     join StringIds s on s.id = k.shortName where k.graphId is not null and k.graphId != 0 order by k.start""").fetchall()
gids = sorted({r[3] for r in rows})
if len(gids) != len(names):
    print(f"warning: {len(gids)} graphs in the trace, {len(names)} names")
gname = {g: (names[i] if i < len(names) else f"g{g}") for i, g in enumerate(gids)}
# split each graph's kernels into replays: a replay = a maximal run of consecutive kernels of that graph
reps = defaultdict(list)
cur_g, cur = None, []
for r in rows:
    if r[3] != cur_g:
        if cur: reps[cur_g].append(cur)
        cur_g, cur = r[3], []
    cur.append(r)
if cur: reps[cur_g].append(cur)
base = None
for g in gids:
    rl = reps[g]
    n = st.median(len(x) for x in rl)
    rl = [x for x in rl if len(x) == n]
    ksum = [sum(k[1] - k[0] for k in x) / 1e3 for x in rl]
    span = [(x[-1][1] - x[0][0]) / 1e3 for x in rl]
    q1 = lambda v: sorted(v)[len(v) // 4]
    print(f"{gname[g]:34s} {len(rl):3d} replays x {int(n):4d} kernels: kernel sum min {min(ksum):9.2f} p25 {q1(ksum):9.2f}"
          f" med {st.median(ksum):9.2f} us | span min {min(span):9.2f} p25 {q1(span):9.2f} med {st.median(span):9.2f} us")
    if "--kernels" in sys.argv:
        per = defaultdict(list)
        for x in rl:
            acc = defaultdict(float)
            for k in x: acc[(k[2], k[4], k[5])] += (k[1] - k[0]) / 1e3
            for kk, v in acc.items(): per[kk].append(v)
        for kk, v in sorted(per.items(), key=lambda kv: -min(kv[1])):
            print(f"      {kk[0][:44]:44s} {str(kk[1:]):>14s} min {min(v):9.2f} p25 {q1(v):9.2f} us/replay")
