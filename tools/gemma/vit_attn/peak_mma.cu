// tensor-core ceiling at the current clock: register-only mma.sync m16n8k16 f16->f32, 8 independent accumulators a warp
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
__global__ void k(float* out, int iters, uint32_t seed) {
    uint32_t a[4] = {seed, seed * 3, seed * 5, seed * 7}, b0 = seed * 11, b1 = seed * 13;
    float c[8][4] = {};
    for (int i = 0; i < iters; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    float s = 0;
    for (int j = 0; j < 8; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    if (s == 12345.f) out[0] = s;
}
int main() {
    float* o; cudaMalloc(&o, 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int blocks = 48 * 4, threads = 256, iters = 2000;
    float best = 1e9;
    for (int r = 0; r < 20; ++r) {
        cudaEventRecord(e0); k<<<blocks, threads>>>(o, iters, 0x3c003c00u); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    const double flop = 2.0 * 16 * 8 * 16 * 8 * (double) iters * blocks * (threads / 32);
    printf("mma peak: %.3f ms  %.1f TFLOPS\n", best, flop / best / 1e9);
}
