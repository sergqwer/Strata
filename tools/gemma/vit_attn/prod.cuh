#pragma once
// extracted verbatim from src/gemma/vision.cu
namespace prod {
// ---- the attention for head_dim 72 (Gemma 4's encoder), ~1.8x the kernel above on the A4000 at the same math.
// A block owns 128 query rows of one head: 4 warps x 2 m-tiles of 16 rows, so every K / V fragment read from shared
// memory feeds two MMAs. Only the 72 real columns are computed: Q K^T as 4 k16 steps + one m16n8k8, P V over 9
// n-tiles of 8 (the zero padding 72..79 is never read). K / V tiles of 64 keys are copied with cp.async into two
// stages (the next tile streams in while this one is computed), stored row-major and dense (72 halves = 144 B a row:
// conflict-free for the cp.async writes and for ldmatrix); K's B fragments come from ldmatrix, V's from ldmatrix.trans
// (no transpose through shared memory). The softmax is the kernel above's: p = exp(s - m) as ex2((s - m) * log2 e),
// f32 row sums of the unrounded p (kept per lane, reduced once at the end), p rounded to f16 for P V.
constexpr int FA3_HD = 72, FA3_BQ = 128, FA3_BK = 64, FA3_THREADS = 128;

__device__ __forceinline__ void mma1688(float (&c)[4], const uint32_t (&a)[2], uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b0));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t& r0, uint32_t& r1, const void* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int bytes) {   // bytes < 16: zero-filled
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// Q, K, V: [image][head][n][HDP] f16 (HDP >= 72; columns past 72 are not read); O: [image][n][H * 72] bf16.
template <int HDP>
__global__ void __launch_bounds__(FA3_THREADS, 2) vit_fa3_kernel(const __half* __restrict__ Q,
                                                                 const __half* __restrict__ K,
                                                                 const __half* __restrict__ V, bf16* __restrict__ O,
                                                                 int n, int H) {
    constexpr int HD = FA3_HD, BK = FA3_BK, MT = 2, LS = HD, CH = HD / 8, NT = BK / 8, KJ = BK / 16;
    constexpr int TILE = BK * LS, ST = 2;
    constexpr float L2E = 1.4426950408889634f;
    static_assert(HDP >= HD && HDP % 8 == 0 && CH % 2 == 1, "144-B rows: ldmatrix rows hit distinct bank groups");
    __shared__ __align__(128) __half smem[ST * 2 * TILE];   // [stage][K | V][BK][LS]: 36 KB
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
    {   // image blockIdx.z of a batch: its own Q / K / V blocks and output rows (each image attends only to itself)
        const size_t z = blockIdx.z;
        Q += z * H * n * HDP;
        K += z * H * n * HDP;
        V += z * H * n * HDP;
        O += z * n * H * HD;
    }
    const __half* Qh = Q + (size_t) h * n * HDP;
    const __half* Kh = K + (size_t) h * n * HDP;
    const __half* Vh = V + (size_t) h * n * HDP;
    const int q0 = blockIdx.x * FA3_BQ + warp * 16 * MT;

    auto load_tile = [&](int stage, int k0) {   // rows past n are zero-filled (their scores are masked)
        __half* ks = smem + stage * 2 * TILE;
        __half* vs = ks + TILE;
#pragma unroll
        for (int i0 = 0; i0 < BK * CH; i0 += FA3_THREADS) {
            const int i = i0 + tid;
            if (BK * CH % FA3_THREADS == 0 || i < BK * CH) {
                const int r = i / CH, c = i % CH;
                const bool ok = k0 + r < n;
                const size_t off = (size_t) (ok ? k0 + r : 0) * HDP + c * 8;
                cp_async16(ks + i * 8, Kh + off, ok ? 16 : 0);
                cp_async16(vs + i * 8, Vh + off, ok ? 16 : 0);
            }
        }
    };
    const int ntiles = (n + BK - 1) / BK;
    if (0 < ntiles) load_tile(0, 0);   // (this exact form: the register allocation at 255 is touchy, others spill)
    cp_async_commit();

    uint32_t qa[MT][4][4], qb[MT][2];   // A fragments: columns 0..63 (k16 steps), 64..71 (the k8 step)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int ra = q0 + m * 16 + g, rb = ra + 8;
        const __half* pa = Qh + (size_t) ra * HDP + 2 * t;
        const __half* pb = Qh + (size_t) rb * HDP + 2 * t;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            qa[m][kk][0] = ra < n ? *reinterpret_cast<const uint32_t*>(pa + kk * 16) : 0u;
            qa[m][kk][1] = rb < n ? *reinterpret_cast<const uint32_t*>(pb + kk * 16) : 0u;
            qa[m][kk][2] = ra < n ? *reinterpret_cast<const uint32_t*>(pa + kk * 16 + 8) : 0u;
            qa[m][kk][3] = rb < n ? *reinterpret_cast<const uint32_t*>(pb + kk * 16 + 8) : 0u;
        }
        qb[m][0] = ra < n ? *reinterpret_cast<const uint32_t*>(pa + 64) : 0u;
        qb[m][1] = rb < n ? *reinterpret_cast<const uint32_t*>(pb + 64) : 0u;
    }
    float o[MT][9][4], mrow[MT][2], lrow[MT][2];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int d = 0; d < 9; ++d) o[m][d][0] = o[m][d][1] = o[m][d][2] = o[m][d][3] = 0.f;
        mrow[m][0] = mrow[m][1] = -INFINITY;
        lrow[m][0] = lrow[m][1] = 0.f;
    }
    const int lr = lane & 7, lj = lane >> 3;   // ldmatrix: lanes 8j..8j+7 address the rows of matrix j

    for (int it = 0; it < ntiles; ++it) {
        cp_async_wait<0>();
        __syncthreads();   // tile it landed for every thread; every warp is done with tile it - 1's stage
        {
            const int tn = it + ST - 1;   // the next tile streams in while this one is computed
            if (tn < ntiles) load_tile(tn % ST, tn * BK);
            cp_async_commit();
        }
        const __half* ks = smem + (it % ST) * 2 * TILE;
        const __half* vs = ks + TILE;
        const int k0 = it * BK;

        float s[MT][NT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) s[m][nt][0] = s[m][nt][1] = s[m][nt][2] = s[m][nt][3] = 0.f;
#pragma unroll
        for (int n4 = 0; n4 < NT / 4; ++n4) {
            uint32_t c8[4];   // columns 64..71 of key n-tiles n4 * 4 .. n4 * 4 + 3
            ldsm_x4(c8[0], c8[1], c8[2], c8[3], ks + ((n4 * 4 + lj) * 8 + lr) * LS + 64);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int nt = n4 * 4 + j;
#pragma unroll
                for (int hb = 0; hb < 2; ++hb) {   // columns 32 hb .. 32 hb + 31: two k16 steps
                    uint32_t b[4];
                    ldsm_x4(b[0], b[1], b[2], b[3], ks + (nt * 8 + lr) * LS + hb * 32 + lj * 8);
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        mma16816(s[m][nt], qa[m][2 * hb], b[0], b[1]);
                        mma16816(s[m][nt], qa[m][2 * hb + 1], b[2], b[3]);
                    }
                }
#pragma unroll
                for (int m = 0; m < MT; ++m) mma1688(s[m][nt], qb[m], c8[j]);
            }
        }
        if (k0 + BK > n) {
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int col = k0 + nt * 8 + 2 * t;
                    if (col >= n) s[m][nt][0] = s[m][nt][2] = -INFINITY;
                    if (col + 1 >= n) s[m][nt][1] = s[m][nt][3] = -INFINITY;
                }
        }
        uint32_t pa[MT][KJ][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float mx0 = mrow[m][0], mx1 = mrow[m][1];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                mx0 = fmaxf(mx0, fmaxf(s[m][nt][0], s[m][nt][1]));
                mx1 = fmaxf(mx1, fmaxf(s[m][nt][2], s[m][nt][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float c0 = ex2_approx((mrow[m][0] - mx0) * L2E), c1 = ex2_approx((mrow[m][1] - mx1) * L2E);
            mrow[m][0] = mx0;
            mrow[m][1] = mx1;
            float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const float p00 = ex2_approx((s[m][nt][0] - mx0) * L2E), p01 = ex2_approx((s[m][nt][1] - mx0) * L2E);
                const float p10 = ex2_approx((s[m][nt][2] - mx1) * L2E), p11 = ex2_approx((s[m][nt][3] - mx1) * L2E);
                sum0 += p00 + p01;
                sum1 += p10 + p11;
                pa[m][nt / 2][(nt & 1) * 2 + 0] = pack_h2(p00, p01);
                pa[m][nt / 2][(nt & 1) * 2 + 1] = pack_h2(p10, p11);
            }
            lrow[m][0] = lrow[m][0] * c0 + sum0;
            lrow[m][1] = lrow[m][1] * c1 + sum1;
#pragma unroll
            for (int d = 0; d < 9; ++d) {
                o[m][d][0] *= c0;
                o[m][d][1] *= c0;
                o[m][d][2] *= c1;
                o[m][d][3] *= c1;
            }
        }
#pragma unroll
        for (int j = 0; j < KJ; ++j) {   // keys 16 j .. 16 j + 15: B fragments of V via ldmatrix.trans
            const __half* vrow = vs + (j * 16 + lr + (lj & 1) * 8) * LS;
#pragma unroll
            for (int dp = 0; dp < 4; ++dp) {   // output columns 16 dp .. 16 dp + 15
                uint32_t v0, v1, v2, v3;
                ldsm_x4_t(v0, v1, v2, v3, vrow + (dp * 2 + (lj >> 1)) * 8);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    mma16816(o[m][2 * dp], pa[m][j], v0, v1);
                    mma16816(o[m][2 * dp + 1], pa[m][j], v2, v3);
                }
            }
            uint32_t v0, v1;   // columns 64..71
            ldsm_x2_t(v0, v1, vrow + 64);
#pragma unroll
            for (int m = 0; m < MT; ++m) mma16816(o[m][8], pa[m][j], v0, v1);
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        float l0 = lrow[m][0], l1 = lrow[m][1];
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        const float i0 = 1.f / l0, i1 = 1.f / l1;
        const int ra = q0 + m * 16 + g, rb = ra + 8;
#pragma unroll
        for (int d = 0; d < 9; ++d) {
            const int col = d * 8 + 2 * t;
            if (ra < n)
                *reinterpret_cast<__nv_bfloat162*>(O + (size_t) ra * H * HD + (size_t) h * HD + col) =
                    __floats2bfloat162_rn(o[m][d][0] * i0, o[m][d][1] * i0);
            if (rb < n)
                *reinterpret_cast<__nv_bfloat162*>(O + (size_t) rb * H * HD + (size_t) h * HD + col) =
                    __floats2bfloat162_rn(o[m][d][2] * i1, o[m][d][3] * i1);
        }
    }
}

}  // namespace prod
