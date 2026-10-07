"""Real activations + an fp64 reference for one MoE layer, for the w4a16 harness's numerics check.

Reads a dump of llama.cpp's MoE inputs (ffn_norm_2-<il>.b*.bin = the expert input g, ffn_moe_logits-<il>.b*.bin =
the router logits; tools/gemma/w4a16/dump_moe.cpp writes them) and the layer's Q4_0 expert tensors
(extract_layer.py), and writes to <out>/:
  L<il>.g.f32      T x D            the expert inputs (f32, as the engine's g_)
  L<il>.ids.i32    T x 8            the chosen experts (top-8 of the logits)
  L<il>.href.f32   T*8 x FF         geglu(W_gu[e] g) per (token, slot), computed in fp64 from the exact Q4_0 weights
  L<il>.yref.f32   T*8 x D          W_down[e] href (fp64)
Header of each file: i64 rows, i64 cols, then the values.

  python mkref.py --dump dsel --layers layers --layer 5 --out ref
"""
import argparse, struct, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--layers", required=True)
ap.add_argument("--layer", type=int, nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def load_dump(name):
    parts = []
    for b in range(16):
        p = Path(a.dump) / f"{name}.b{b}.bin"
        if not p.exists():
            break
        raw = p.read_bytes()
        ne = struct.unpack("4q", raw[:32])
        parts.append(np.frombuffer(raw[32:], dtype=np.float32).reshape(-1, ne[0]))
    return np.concatenate(parts, 0)


def deq_q4_0(raw, rows, cols):
    """Q4_0 bytes of a [rows x cols] matrix -> fp64 [rows, cols]: value j<16 = (qs[j]&15)-8, j>=16 = (qs[j-16]>>4)-8, x d"""
    nb = cols // 32
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(rows, nb, 18)
    d = blk[:, :, :2].copy().view(np.float16).astype(np.float64)[:, :, 0]
    qs = blk[:, :, 2:]
    lo = (qs & 15).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    v = np.concatenate([lo, hi], axis=2).astype(np.float64)  # [rows, nb, 32]
    return (v * d[:, :, None]).reshape(rows, cols)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.79788456080286535587989211986876 * x * (1.0 + 0.044715 * x * x)))


def save(path, arr, dtype):
    arr = np.ascontiguousarray(arr, dtype=dtype)
    with open(path, "wb") as f:
        f.write(struct.pack("2q", arr.shape[0], arr.shape[1] if arr.ndim > 1 else 1))
        f.write(arr.tobytes())


for il in a.layer:
    meta = json.loads((Path(a.layers) / f"L{il}.json").read_text())
    D, FF2, E = meta["gate_up"]["ne"]
    FF = FF2 // 2
    g = load_dump(f"ffn_norm_2-{il}").astype(np.float32)
    logits = load_dump(f"ffn_moe_logits-{il}")
    T = g.shape[0]
    ids = np.argsort(-logits, axis=1, kind="stable")[:, : a.k].astype(np.int32)
    gu_raw = (Path(a.layers) / f"L{il}.gate_up").read_bytes()
    dn_raw = (Path(a.layers) / f"L{il}.down").read_bytes()
    gu_eb = len(gu_raw) // E
    dn_eb = len(dn_raw) // E
    href = np.zeros((T * a.k, FF), np.float64)
    yref = np.zeros((T * a.k, D), np.float64)
    g64 = g.astype(np.float64)
    for e in range(E):
        pairs = np.nonzero(ids.reshape(-1) == e)[0]
        if len(pairs) == 0:
            continue
        x = g64[pairs // a.k]
        wgu = deq_q4_0(gu_raw[e * gu_eb:(e + 1) * gu_eb], FF2, D)
        gu = x @ wgu.T
        h = gelu_tanh(gu[:, :FF]) * gu[:, FF:]
        wd = deq_q4_0(dn_raw[e * dn_eb:(e + 1) * dn_eb], D, FF)
        href[pairs] = h
        yref[pairs] = h @ wd.T
    save(out / f"L{il}.g.f32", g, np.float32)
    save(out / f"L{il}.ids.i32", ids, np.int32)
    save(out / f"L{il}.href.f32", href, np.float32)
    save(out / f"L{il}.yref.f32", yref, np.float32)
    c = np.bincount(ids.ravel(), minlength=E)
    print(f"L{il}: T {T}, max/mean rows per expert {c.max()}/{c.mean():.1f}, |h| max {np.abs(href).max():.2f} "
          f"rms {np.sqrt((href**2).mean()):.4f}, |y| rms {np.sqrt((yref**2).mean()):.4f}")
