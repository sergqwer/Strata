# tools/gemma/vis_gemm_data.py - one encoder layer's matrices from a gemma4v mmproj as raw files for vis_gemm_bench:
#   python vis_gemm_data.py <layer> <out dir> [mmproj.gguf]   (gguf-py from llama.cpp)
import gguf, numpy as np, sys, os
r = gguf.GGUFReader(sys.argv[3] if len(sys.argv) > 3 else "/home/ubuntu/models/gemma-4-26B-A4B-qat/gemma-4-26B-it-mmproj.gguf")
T = {t.name: t for t in r.tensors}
L = int(sys.argv[1]); out = sys.argv[2]
def raw(name):
    t = T[name]; a = np.asarray(t.data)
    print(name, t.tensor_type.name, a.dtype, a.shape)
    return a
p = f'v.blk.{L}.'
qkv = np.concatenate([raw(p+'attn_q.weight'), raw(p+'attn_k.weight'), raw(p+'attn_v.weight')], axis=0)
gu = np.concatenate([raw(p+'ffn_gate.weight'), raw(p+'ffn_up.weight')], axis=0)
for nm, a in [('wqkv', qkv), ('wo', raw(p+'attn_out.weight')), ('wgu', gu), ('wdown', raw(p+'ffn_down.weight')),
              ('ln1', raw(p+'ln1.weight')), ('ln2', raw(p+'ln2.weight')), ('proj', raw('mm.input_projection.weight')),
              ('patch', raw('v.patch_embd.weight'))]:
    a = np.ascontiguousarray(a); a.tofile(os.path.join(out, nm + '.bin')); print(' ->', nm, a.dtype, a.shape, a.nbytes)
