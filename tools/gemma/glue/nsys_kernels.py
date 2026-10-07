#!/usr/bin/env python3
"""Per-kernel durations of a glue_bench run under nsys (`nsys profile --cuda-graph-trace=node`, then
`nsys export -t sqlite`): the last `replays` graph replays, each kernel position's median duration, grouped by kernel
name and grid; the replay span vs the summed kernel time (the gaps); with --seq, the kernel sequence of replay 0.
  python3 nsys_kernels.py prof/old_dec.sqlite [replays] [--seq [first] [count]]"""
import sqlite3, sys, statistics as st
db = sqlite3.connect(sys.argv[1])
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rows = db.execute("""select k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX*k.blockY*k.blockZ
                     from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName order by k.start""").fetchall()
n = len(rows)
per = n // reps
rows = rows[n - per * reps:]
med = []
for p in range(per):
    med.append(st.median(rows[r * per + p][1] - rows[r * per + p][0] for r in range(reps)) / 1e3)
gaps = []
for p in range(1, per):
    gaps.append(st.median(rows[r * per + p][0] - rows[r * per + p - 1][1] for r in range(reps)) / 1e3)
agg = {}
for p in range(per):
    _, _, name, gx, gy, gz, bs = rows[p]
    agg.setdefault((name, gx, gy, gz, bs), []).append(med[p])
print(f"{per} kernels per replay")
print(f"{'kernel':42s} {'grid':>16s} {'blk':>5s} {'n':>4s} {'med us':>8s} {'sum us':>9s}")
for k2, ds in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
    print(f"{k2[0][:42]:42s} {str((k2[1],k2[2],k2[3])):>16s} {k2[4]:5d} {len(ds):4d} {st.median(ds):8.2f} {sum(ds):9.1f}")
spans = [rows[(r + 1) * per - 1][1] - rows[r * per][0] for r in range(reps)]
print(f"kernel time per replay {sum(med)/1e3:.3f} ms, gaps {sum(gaps)/1e3:.3f} ms; replay span min {min(spans)/1e6:.3f} median {st.median(spans)/1e6:.3f} ms")
if "--seq" in sys.argv:
    i = sys.argv.index("--seq")
    first = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 0
    cnt = int(sys.argv[i + 2]) if len(sys.argv) > i + 2 else 40
    for p in range(first, min(per, first + cnt)):
        _, _, name, gx, gy, gz, bs = rows[p]
        print(f"  {p:4d} {name[:40]:40s} {str((gx,gy,gz)):>16s} {bs:5d} {med[p]:8.2f} us  gap {gaps[p-1] if p else 0:6.2f}")
