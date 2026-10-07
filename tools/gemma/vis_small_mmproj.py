# tools/gemma/vis_small_mmproj.py - a copy of a gemma4v mmproj with only its first N encoder layers and the position
# table cut to 128 rows, for vis_tune_test on little VRAM:  python vis_small_mmproj.py <src.gguf> <dst.gguf> <N>
import gguf, numpy as np, sys
src, dst, nl = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(dst, 'clip')
for k, f in r.fields.items():
    if k.startswith('GGUF.') or k == 'general.architecture': continue
    t = f.types[0]
    v = f.contents()
    if k == 'clip.vision.block_count': v = nl
    if t == gguf.GGUFValueType.ARRAY:
        w.add_array(k, v)
    elif t == gguf.GGUFValueType.STRING:
        w.add_string(k, v)
    elif t == gguf.GGUFValueType.BOOL:
        w.add_bool(k, v)
    elif t == gguf.GGUFValueType.FLOAT32:
        w.add_float32(k, v)
    elif t == gguf.GGUFValueType.UINT32:
        w.add_uint32(k, v)
    elif t == gguf.GGUFValueType.INT32:
        w.add_int32(k, v)
    elif t == gguf.GGUFValueType.UINT64:
        w.add_uint64(k, v)
    else:
        w.add_key_value(k, v, t)
for t in r.tensors:
    if t.name.startswith('v.blk.') and int(t.name.split('.')[2]) >= nl: continue
    ne = [int(x) for x in t.shape]
    if t.name == 'v.position_embd.weight':
        a = np.asarray(t.data).reshape(ne[2], ne[1], ne[0])[:, :128, :].copy()
        w.add_tensor(t.name, a)
    elif t.tensor_type == gguf.GGMLQuantizationType.BF16:
        w.add_tensor(t.name, np.asarray(t.data).reshape(list(reversed(ne))[:-1] + [ne[0] * 2]), raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        w.add_tensor(t.name, np.asarray(t.data).reshape(list(reversed(ne))))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
r2 = gguf.GGUFReader(dst)
print(len(r2.tensors), 'tensors;', [(t.name, list(t.shape), t.tensor_type.name) for t in r2.tensors if not t.name.startswith('v.blk.1')][:8])
print({k: r2.fields[k].contents() for k in ['clip.vision.block_count', 'clip.vision.projector_type', 'clip.vision.attention.layer_norm_epsilon']})
