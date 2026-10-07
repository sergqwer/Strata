// tools/gemma/mmq/epilogue_peak.cu - why a bit-identical MMQ kernel cannot go much faster on the A4000.
//
//   nvcc -O3 -arch=sm_86 -use_fast_math -o epilogue_peak epilogue_peak.cu && flock /root/sg-tools/gpu.lock ./epilogue_peak
//
// IMMA m16n8k32 (int8) throughput from registers only, with the float epilogue llama.cpp's MMQ runs after every
// 32-value block (sum += float(c) * dA * dB) and cheaper variants.  Measured 2026-10-07 (TOPS, 2*4096 ops per IMMA):
// IMMA + a dead FFMA 114-122, + FFMA 112, + FMUL+FFMA 100, + FADD+FMUL+FFMA (c from an accumulator started at
// 0x4B400000) 75, + IADD+I2FP+FMUL+FFMA (llama.cpp's conversion) 51.  llama.cpp's kernel reaches ~43 TOPS on the dense
// products, so any kernel that keeps its per-block arithmetic (bit-identity) has at most ~1.5x left even in theory.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
template <int EPI, int NACC>
__global__ void __launch_bounds__(256, 2) k(int iters, float* out, const float* sc) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u}, b0 = threadIdx.x * 11u, b1 = threadIdx.x * 13u;
    float acc[NACC][4] = {};
    int d[NACC][4];
    const float dA = sc[threadIdx.x & 7], dB = sc[8 + (threadIdx.x & 7)];
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int n = 0; n < NACC; ++n) {
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                         : "=r"(d[n][0]), "=r"(d[n][1]), "=r"(d[n][2]), "=r"(d[n][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0 + n), "r"(b1), "r"(0x4B400000));
            if (EPI == 1) {
#pragma unroll
                for (int l = 0; l < 4; ++l) acc[n][l] = __fmaf_rn(__fmul_rn(__fsub_rn(__int_as_float(d[n][l]), 12582912.f), dA), dB, acc[n][l]);
            } else if (EPI == 2) {
#pragma unroll
                for (int l = 0; l < 4; ++l) acc[n][l] = __fmaf_rn(__fmul_rn(__int2float_rn(d[n][l] - 0x4B400000), dA), dB, acc[n][l]);
            } else if (EPI == 3) {
#pragma unroll
                for (int l = 0; l < 4; ++l) acc[n][l] = __fmaf_rn(__int_as_float(d[n][l]), dB, acc[n][l]);
            } else if (EPI == 4) {
#pragma unroll
                for (int l = 0; l < 4; ++l) acc[n][l] = __fmaf_rn(__fmul_rn(__int_as_float(d[n][l]), dA), dB, acc[n][l]);
            } else {
#pragma unroll
                for (int l = 0; l < 4; ++l) acc[n][l] += __int_as_float(d[n][l]) * 0.f;
            }
        }
        b0 += 1;
    }
    float s = 0;
#pragma unroll
    for (int n = 0; n < NACC; ++n) for (int l = 0; l < 4; ++l) s += acc[n][l];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int EPI, int NACC> void run(const char* nm, float* out, float* sc) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int iters = 4096, blocks = 96;
    k<EPI, NACC><<<blocks, 256>>>(10, out, sc);
    float best = 1e9;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(e0); k<EPI, NACC><<<blocks, 256>>>(iters, out, sc); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    double imma = (double) blocks * 8 * iters * NACC;
    printf("%-28s %.3f ms  %.1f TOPS (int8, 2*4096 ops per IMMA)\n", nm, best, imma * 8192 / (best * 1e-3) / 1e12);
}
int main() {
    float *out, *sc; cudaMalloc(&out, 1 << 20); cudaMalloc(&sc, 64 * 4); cudaMemset(sc, 0, 256);
    run<0, 8>("IMMA + 1 FFMA (zero)", out, sc);
    run<3, 8>("IMMA + FFMA", out, sc);
    run<4, 8>("IMMA + FMUL+FFMA", out, sc);
    run<1, 8>("IMMA + FADD+FMUL+FFMA", out, sc);
    run<2, 8>("IMMA + IADD,I2FP+FMUL+FFMA", out, sc);
    run<0, 8>("IMMA + 1 FFMA (zero)", out, sc);
    return 0;
}
