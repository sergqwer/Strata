# tools/gemma/mmq - the prompt path's MMQ products, old vs new, bit for bit

`mmq_bench.cu` (CMake target `strata-gemma-mmq-bench`, not built by default) runs the engine's prompt-path products
on one layer's real weights from the GGUF, the old code against the new, and compares every output buffer
**byte for byte** (memcmp; the destination is filled with 0xff before each run). Timing: loops of ~`--loop-ms` of
back-to-back calls, old and new interleaved, `--reps` times; min and median of the per-call mean.

```
make -C build -j4 strata-gemma-mmq-bench
flock /root/sg-tools/gpu.lock build/strata-gemma-mmq-bench --layer 0 --tokens 692,917 --what moe --reps 4
flock /root/sg-tools/gpu.lock build/strata-gemma-mmq-bench --layer 5 --tokens 692 --what dense --check-only
```

- `--what moe`: quantize of the expert rows (gathered row by row vs once per token, `mmq::quantize_scatter`), the
  gate_up product (llama.cpp's grid sized by a host-synced max_rows vs `Context::run_tiles`), geglu + quantize (two
  kernels vs `mmq::geglu_quantize`), the down product. Routing per token (Gumbel top-8) from the layer's routing mass
  in `/root/sg-tools/routing-26b.json` (`skew`) or equal weights (`uniform`); activations are synthetic.
- `--what dense`: qkv / attn_out / dense MLP products, old vs the tile path (informational: the dense products keep
  llama.cpp's launch, whose stream-k split for some token counts sums partials in another order).
- VRAM: the process stays under `--vram-mb` (450). Where all 96 experts do not fit, the MoE test takes every 2nd
  expert by routing mass with top-4 (`--expert-frac 2`): the same rows per expert, half the weights and rows.
- `--variants 0:64,0:128,1:32`: time run_tiles' knobs (width set, fixed tile cost) against the old code.
