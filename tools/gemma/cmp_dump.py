"""Compare two dump directories (ref_logits --dump vs STRATA_DUMP): per tensor, max/rms difference relative to the ref."""
import struct, sys, os, re
import numpy as np

def load(p):
    with open(p, "rb") as f:
        ne = struct.unpack("<4q", f.read(32))
        a = np.frombuffer(f.read(), dtype=np.float32)
    return ne, a

ref, ours = sys.argv[1], sys.argv[2]
names = sorted(set(os.listdir(ref)) & set(os.listdir(ours)), key=lambda n: (int(re.search(r"-(\d+)\.bin$", n).group(1)) if re.search(r"-(\d+)\.bin$", n) else -1, n))
for n in names:
    (ne_r, r), (ne_o, o) = load(os.path.join(ref, n)), load(os.path.join(ours, n))
    if r.size != o.size:
        print(f"{n:32s} size {ne_r} vs {ne_o}")
        continue
    d = np.abs(r - o)
    print(f"{n:32s} ref rms {np.sqrt((r*r).mean()):9.4f}  max|d| {d.max():9.4f}  rms d {np.sqrt((d*d).mean()):8.5f}  rel {np.sqrt((d*d).mean())/max(np.sqrt((r*r).mean()),1e-9):.2e}")
