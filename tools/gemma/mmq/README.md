# tools/gemma/mmq - the prompt path's MMQ products, old vs new, bit for bit

`mmq_bench.cu` (CMake target `strata-gemma-mmq-bench`, not built by default) runs the engine's prompt-path MMQ work
on one layer's real weights from the GGUF, the old code (layer_big before the tile list, call for call) against the
new, and compares every output buffer **byte for byte** (memcmp; the destination is filled with 0xff before each
run, so a value the new code does not write cannot pass). Timing: loops of ~`--loop-ms` of back-to-back calls, old
and new interleaved, `--reps` times; min and median of the per-call mean. The card is the live production card:
absolute times swing ~2x with production's load, the old/new ratios much less; `ncu --clock-control none -k
regex:"mul_mat_q|tiles_kernel"` on a `--check-only` run gives kernel-only times.

```
make -C build -j4 strata-gemma-mmq-bench
flock /root/sg-tools/gpu.lock build/strata-gemma-mmq-bench --layer 0 --tokens 692,917 --what moe --reps 6 --loop-ms 50
flock /root/sg-tools/gpu.lock build/strata-gemma-mmq-bench --layer 0 --tokens 692,1787 --what moe,mlp --check-only
flock /root/sg-tools/gpu.lock build/strata-gemma-mmq-bench --layer 0 --tokens 692 --what moe --variants "jset=0;c0=64;c0=192"
```

- `--what moe`: quantize of the expert rows (gathered row by row vs once per token, `mmq::quantize_scatter`), the
  gate_up product (llama.cpp's grid sized by a host-synced max_rows vs `Context::run_tiles`; also with the old bounds
  copy + stream sync included), geglu + quantize (two kernels vs `mmq::geglu_quantize`), the down product. Routing per
  token (Gumbel top-8) from the layer's routing mass in `/root/sg-tools/routing-26b.json` (`skew`) or equal weights
  (`uniform`); activations are synthetic (normal, per-channel scales, rare outliers).
- `--what mlp`: the dense MLP's geglu + quantize, two kernels vs `mmq::geglu_quantize`.
- `--what dense`: qkv / attn_out / dense MLP products, llama.cpp's launch vs the tile path, for information: the
  dense products keep llama.cpp's launch (the tile path matches it bit for bit where llama.cpp tiles, not where its
  stream-k split sums partials in another order, and is no faster).
- VRAM: the process stays under `--vram-mb` (450, the CUDA context and code take ~175 MiB). Where all 96 experts do
  not fit, the MoE test takes every 2nd expert by routing mass with top-4 (`--expert-frac 2`): the same rows per
  expert and tiles of the same shapes, half the weights and rows (its times are about half a layer's).
- `--variants "jset=0;c0=192"`: time run_tiles' knobs against the old code (`mmq::knob`).

`epilogue_peak.cu`: IMMA m16n8k32 throughput from registers with the float epilogue MMQ runs per 32-value block
(sum += float(c) * dA * dB) and cheaper variants. It is why no faster bit-identical tile was found: that epilogue
caps any kernel which keeps llama.cpp's per-block arithmetic at ~51 TOPS with I2FP (llama.cpp) or ~75 TOPS with a
conversion-free float(c), against ~115 for the IMMA alone, and llama.cpp's kernel already reaches ~43 TOPS on the
dense products. Two such rewrites (16 warps with a cp.async pipeline; llama.cpp's tile with the conversion-free
epilogue) were bit-identical and no faster; they are in the branch history (commit d1d7ff1).
