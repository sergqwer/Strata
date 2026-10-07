"""Synthetic routing for the w4a16 speed harness: per token, 8 distinct experts drawn by Gumbel top-k from the layer's
routing mass in routing-26b.json (the 96 kept experts = the 96 most-routed of 128, in ascending original index), or
uniformly (--uniform). Writes <out>/route-L<il>-T<tokens>[-u].i32 (i64 rows, i64 8, then T x 8 int32).

  python gen_routing.py --routing routing-26b.json --layer 5 --tokens 692 917 1190 1787 --out /dev/shm/w4a16
"""
import argparse, json, struct
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--routing", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--tokens", type=int, nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--uniform", action="store_true")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--experts", type=int, default=96)
a = ap.parse_args()
m = np.array(json.load(open(a.routing))[a.layer], dtype=np.float64)
keep = np.sort(np.argsort(-m)[: a.experts])
p = m[keep] / m[keep].sum()
rng = np.random.default_rng(a.layer)
for T in a.tokens:
    logp = np.zeros(a.experts) if a.uniform else np.log(p)
    sel = np.argsort(-(logp[None, :] + rng.gumbel(size=(T, a.experts))), axis=1)[:, : a.k].astype(np.int32)
    c = np.bincount(sel.ravel(), minlength=a.experts)
    name = Path(a.out) / f"route-L{a.layer}-T{T}{'-u' if a.uniform else ''}.i32"
    with open(name, "wb") as f:
        f.write(struct.pack("2q", T, a.k))
        f.write(sel.tobytes())
    print(name.name, "rows/expert max", c.max(), "mean", round(c.mean(), 1), "min", c.min())
