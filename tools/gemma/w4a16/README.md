Harness for the w4a16 expert GEMMs (`src/gemma/moe_w4a16.cuh`, `--moe-w4a16`): the prompt path's MoE gate_up +
GeGLU and down on the fp16 tensor cores straight from the Q4_0 expert tensors, against the engine's q8_1 + MMQ path.

Data (one layer, real weights; nothing here needs the engine):

    python extract_layer.py --gguf model.gguf --layer 5 10 --out /dev/shm/w4a16      # L<il>.gate_up / .down (+ sha256)
    tools/gemma/w4a16/dump_moe ...    (a copy of tools/gemma/ref_logits.cpp whose dump callback keeps every ubatch:
                                       ffn_norm_2-<il> = the expert input g, ffn_moe_logits-<il>, ffn_moe_geglu-<il>)
    python mkref.py --dump <dump dir> --layers <dir> --layer 5 10 --out <dir>        # g, top-8 ids, fp64 hidden and y
    python gen_routing.py --routing routing-26b.json --layer 5 --tokens 692 917 1190 1787 --out <dir> [--uniform]

The expert tensors of the production GGUF are the base e96 ones (LoRA touches attention and the dense MLP only):
layer 5's sha256 matched, so the dump and the reference can be made on any machine with the base GGUF.

Build (on the VPS, after `make -C build strata_mmq ggml-base` of this tree):

    nvcc -O3 -arch=sm_86 -use_fast_math -extended-lambda -std=c++17 -Iinclude tools/gemma/w4a16/harness.cu \
         build/libstrata_mmq.a build/llama.cpp/ggml/src/libggml-base.a -lcuda -o harness
    flock /root/sg-tools/gpu.lock ./harness num   /dev/shm/w4a16 5                       # numerics vs fp64
    flock /root/sg-tools/gpu.lock ./harness speed /dev/shm/w4a16 5 route-L5-T692.i32 4 30 [GU2P2,DN2P2,...]
    ncu --clock-control none --kernel-name regex:gemm ./harness prof ...                  # one launch each

`speed` times every candidate and the old path's kernels in interleaved ~4 ms loops (30-60 rounds; min / p10 /
median): the card is shared with production, whose load swings by 2-3x within seconds, so only the minimum of many
short samples is comparable (the old path's minimum matches the live engine's phase times). The weights of the speed
runs are 32 experts' worth of physical memory mapped three times over the virtual range (CUDA VMM): the DRAM traffic
stays real at a third of the VRAM. GPU memory: ~290 MiB (num), ~130-260 MiB (speed). Ablation variants (A*, R*, V*:
no MMA / no copies / copies only ...) give wrong results on purpose and only measure where the time goes.

Results (A4000 under production load, 2026-10-07; layer 5, routing drawn from routing-26b.json; ms a layer, min of
60 interleaved ~4 ms loops; old = q8_1 gather + MMQ gate_up | GeGLU + q8_1 + MMQ down, new = fp16 scatter +
gate_up/GeGLU | down; the old minimums match the live engine's phases, e.g. 68 / 48 ms x30 at 692 tokens):

| tokens | old gather + gate_up | new scatter + gate_up | old down phase | new down | old total | new total | |
|---|---|---|---|---|---|---|---|
| 692 | 0.10 + 2.26 | 0.10 + 1.24 (35 TFLOPS) | 1.58 | 0.80 | 3.94 | 2.15 | 1.83x |
| 917 | 0.15 + 3.01 | 0.15 + 1.51 | 1.64 | 0.89 | 4.80 | 2.54 | 1.89x |
| 1190 | 0.20 + 3.37 | 0.20 + 1.89 (40 TFLOPS) | 1.91 | 1.08 | 5.48 | 3.17 | 1.73x |
| 1787 | 0.36 + 3.23 | 0.27 + 2.64 (43 TFLOPS) | 2.97 | 1.56 | 6.56 | 4.46 | 1.47x |
| 692 uniform | 0.10 + 1.39 | 0.10 + 1.13 | 1.04 | 0.67 | 2.52 | 1.91 | 1.32x |

(the old path also synchronizes with the host once a layer for the bounds; the new one does not). Most of the gain is
MMQ's cost on skewed routing (its grid spans the largest expert); at 1787 tokens gate_up runs at ~43 TFLOPS, near the
~45-48 the power cap sustains for fp16 MMA. Numerics, real activations (an area page, 747 tokens) vs fp64, ey rel. RMS
error old -> new: layer 0 2.6e-3 -> 7.2e-5, 5 5.8e-3 -> 1.5e-4, 10 8.4e-3 -> 2.9e-4, 16 1.1e-2 -> 2.8e-4,
29 1.2e-2 -> 3.6e-4 (worst row 1.6-5.5e-2 -> 4.7-6.5e-4); bf16 was not needed (largest |g| 38, |hidden| 291, fp16
max 65504; both are clamped). Block scales on the fp32 partial sums (weights exact) gave the same error as scales
folded into fp16 weights (2.449e-4 vs 2.448e-4) and were slower, so production folds them.
