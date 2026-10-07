Standalone harness for the vision encoder's attention (head_dim 72): the old `vit_fa2_kernel` (`cur.cuh`), the
candidates (`fa3.cuh`, `fa4.cuh`) and the kernel as it went into `vision.cu` (`prod.cuh`), checked against an FP32
(double-sum) reference and timed (min over reps). `peak_mma.cu` measures the card's m16n8k16 ceiling at its clock.

    nvcc -O3 -arch=sm_86 -use_fast_math -std=c++17 harness.cu -o harness
    flock /root/sg-tools/gpu.lock ./harness <sigma> <reps> "<kernel filter>" "B:n,..."    # e.g. 1 40 "" "1:4959,9:630"

A4000 (2026-10-07): 1 x 4959 patches 3.11 -> 1.67 ms (36 -> 67 TFLOPS), 9 x 630 0.52 -> 0.29 ms; same bf16 results
as the old kernel (1 ulp on rare elements), equal error vs the reference. The kernel uses 255 registers: after any
edit check `cuobjdump --dump-resource-usage` for LOCAL:0 (a 44 B spill cost ~20%).
