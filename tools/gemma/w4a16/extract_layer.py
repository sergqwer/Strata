"""Write one layer's expert tensors of a Gemma 4 GGUF as raw files for the w4a16 harness:
<out>/L<il>.gate_up (Q4_0 bytes, ne 2816 x 1408 x E), .down (Q4_0, 704 x 2816 x E), .down_scale (f32 x E, if present),
plus <out>/L<il>.json (shapes, types, sha256 of each file).

  python extract_layer.py --gguf model.gguf --layer 5 --out /dev/shm/w4a16
"""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "src/llama.cpp-latest/gguf-py"))
sys.path.insert(0, "/root/llama.cpp/gguf-py")
import gguf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--layer", type=int, nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hash-only", action="store_true")
a = ap.parse_args()
r = gguf.GGUFReader(a.gguf)
by = {t.name: t for t in r.tensors}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for il in a.layer:
    meta = {}
    for key, name in (("gate_up", "ffn_gate_up_exps.weight"), ("down", "ffn_down_exps.weight"),
                      ("down_scale", "ffn_down_exps.scale")):
        t = by.get(f"blk.{il}.{name}")
        if t is None:
            continue
        b = t.data.tobytes()
        meta[key] = {"ne": [int(x) for x in t.shape], "type": int(t.tensor_type), "bytes": len(b),
                     "sha256": hashlib.sha256(b).hexdigest()}
        if not a.hash_only:
            (out / f"L{il}.{key}").write_bytes(b)
    (out / f"L{il}.json").write_text(json.dumps(meta, indent=1))
    print(il, {k: (v["ne"], v["type"], v["sha256"][:16]) for k, v in meta.items()})
