// src/gemma/moe_w4a16.cuh - the device code of the w4a16 expert GEMMs (see include/strata/gemma/moe_w4a16.hpp):
// templated on the tile shape and the scale handling so that tools/gemma/w4a16/harness.cu can measure the variants;
// src/gemma/moe_w4a16.cu instantiates the production ones.
//
// Operand roles (mma.m16n8k16, f16 inputs, f32 accumulators): A = the weights (16 weight rows), B = the activations
// (8 rows of one expert). One 32-bit word of a Q4_0 block - qs[4t .. 4t+3] for quad lane t - holds the values
// v = 4t..4t+3 (low nibbles) and 16+4t..16+4t+3 (high nibbles); `dq` turns it into the A registers of both k16 steps
// of the block: step 0 pairs (4t, 4t+2) -> k (2t, 2t+1) and (4t+1, 4t+3) -> k (2t+8, 2t+9), step 1 the same + 16.
// The MMA sums over k, so any order of k works as long as both operands use it: the activations are stored in that
// order ("fragment-contiguous", fpos / fval below), so one 16-byte load gives a lane both k16 steps' B registers.
//
// A Q4_0 block is 18 bytes, so in a 4-byte aligned row its 16 nibble bytes start 2 bytes off a word boundary in every
// other block: a pair of blocks (36 bytes, 9 words) is [d0 q0 q1 | q2-5 | q6-9 | q10-13 | q14 q15 d1 | q'0-3 | ...]:
// the even block's word for lane t is bytes 2..5 of words t, t+1 (one PRMT), the odd block's is word 5+t.
//
// The design and its measurements are in the kernels' comment below; tools/gemma/w4a16/v1.cuh is the first version
// (shared-memory staging of both operands), kept for the harness.
#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace strata::gemma::w4a16::dev {

__device__ __forceinline__ float gelu_tanh(float x) {   // = k::gelu_tanh (ggml_cuda_op_gelu_single)
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ __half clamp_half(float v) {   // fp16 has no headroom check of its own: saturate, never inf
    return __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
}


__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }

template <int BYTES> __device__ __forceinline__ void cp_async(uint32_t dst, const void* src, bool pred) {
    const int n = pred ? BYTES : 0;   // 0: zero-fill, nothing read
    if constexpr (BYTES == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(dst), "l"(src), "n"(BYTES), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// the same into fresh accumulators (c = a b)
__device__ __forceinline__ void mma16816_z(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %10, %10, %10};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {   // (a & b) | c
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// one word of Q4_0 nibbles -> the four half2 A registers of a row: l0 / l1 for k16 step 0 (k 2t,2t+1 / 2t+8,2t+9),
// h0 / h1 for step 1, each value n - 8 exactly (1024 + n is an exact fp16; so is 1024 + 16 n)
__device__ __forceinline__ void dq(uint32_t q, uint32_t& l0, uint32_t& l1, uint32_t& h0, uint32_t& h1) {
    const uint32_t q8 = q >> 8;
    const uint32_t a = lop3_and_or(q, 0x000f000fu, 0x64006400u);
    const uint32_t b = lop3_and_or(q8, 0x000f000fu, 0x64006400u);
    const uint32_t c = lop3_and_or(q, 0x00f000f0u, 0x64006400u);
    const uint32_t d = lop3_and_or(q8, 0x00f000f0u, 0x64006400u);
    const __half2 sub = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));   // 1032
    const __half2 mul = __halves2half2(__ushort_as_half(0x2c00), __ushort_as_half(0x2c00));   // 1/16
    const __half2 add = __halves2half2(__ushort_as_half(0xd480), __ushort_as_half(0xd480));   // -72
    __half2 r;
    r = __hsub2(*reinterpret_cast<const __half2*>(&a), sub);
    l0 = *reinterpret_cast<uint32_t*>(&r);
    r = __hsub2(*reinterpret_cast<const __half2*>(&b), sub);
    l1 = *reinterpret_cast<uint32_t*>(&r);
    r = __hfma2(*reinterpret_cast<const __half2*>(&c), mul, add);
    h0 = *reinterpret_cast<uint32_t*>(&r);
    r = __hfma2(*reinterpret_cast<const __half2*>(&d), mul, add);
    h1 = *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint32_t hmul2u(uint32_t x, __half2 s) {
    __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&x), s);
    return *reinterpret_cast<uint32_t*>(&r);
}

// ------------------------------------------------------------------------------------------------ the GEMMs

struct Args {
    const uint8_t* w = nullptr;   // Q4_0 expert tensor
    size_t row_bytes = 0, expert_bytes = 0;
    int K = 0;                    // values a row: d (gate_up) or ff (down)
    int ff = 0;                   // gate_up: rows of gate (the up rows follow)
    const __half* x = nullptr;    // [rows x K], sorted by expert, reordered per 16
    const int32_t* bounds = nullptr;
    int n_expert = 0;
    __half* hidden = nullptr;     // gate_up output [rows x ff]
    float* ey = nullptr;          // down output [rows x ld_ey]
    int ld_ey = 0;
};

// ================================================================================================ the kernels
//
// Measured on the first version (tools/gemma/w4a16/v1.cuh, harness ablations): the copies, not the MMAs, set the
// time - the activation tile went through the L2 once per 128-row weight tile (11x for gate_up, 22x for down: ~64% of
// the L2 traffic), and the weight k-slices (36 bytes of 128 scattered rows a k-tile) ran the DRAM at ~200 GB/s. Here:
//  - 352 weight rows a block (11 warps x 32 rows): the activations pass the L2 4x (gate_up) / 8x (down);
//  - every warp streams its own 32 weight rows through a 256-byte ring per row in shared memory, a whole 128-byte line
//    at a time (one request, DRAM-friendly) and LW k-tiles ahead; no block barriers in the main loop;
//  - the activations go straight from global memory into the B fragments (one 16-byte load per n8 tile and k32 block;
//    the 11 warps of a block read the same bytes, the L1 serves most), stored "fragment-contiguous": per 32-value
//    block, lane t's 16 bytes are [v4t, v4t+2, v4t+1, v4t+3, v16+4t, v18+4t, v17+4t, v19+4t] (fpos / fval);
//  - gate_up: a warp's 16 gate rows (and the matching up rows) are the 16 hidden units one half of a hidden block
//    holds, so its GeGLU output is 32 contiguous bytes a row of the hidden, already in down's B layout;
//  - the main loop is instantiated per count of n8 tiles that hold rows (2/4/6/8): no per-tile checks inside.
// ABL (ablations: parts of the work skipped, wrong results) exists for the harness only; production uses 0.

__host__ __device__ constexpr int fpos(int v) {   // storage position of value v (0..31) in its 32-value block
    return 8 * ((v >> 2) & 3) + 4 * (v >> 4) + ((v & 3) == 1 ? 2 : (v & 3) == 2 ? 1 : (v & 3));
}
__host__ __device__ constexpr int fval(int p) {   // the value at storage position p
    return 16 * ((p >> 2) & 1) + 4 * (p >> 3) + ((p & 3) == 1 ? 2 : (p & 3) == 2 ? 1 : (p & 3));
}

// one thread per (token, 32-value block): 32 floats -> 32 fp16 in fpos order, to each of the token's k sorted rows
__global__ void scatter_kernel(const float* __restrict__ g, int d, const int32_t* __restrict__ inv, int tokens, int k,
                                __half* __restrict__ xs) {
    const int blocks = d / 32;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) tokens * blocks) return;
    const int t = (int) (i / blocks), blk = (int) (i % blocks);
    const float4* src = reinterpret_cast<const float4*>(g + (size_t) t * d + (size_t) blk * 32);
    float v[32];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float4 f = src[j];
        v[4 * j] = f.x;
        v[4 * j + 1] = f.y;
        v[4 * j + 2] = f.z;
        v[4 * j + 3] = f.w;
    }
    __align__(16) __half h[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) h[fpos(j)] = clamp_half(v[j]);
    for (int j = 0; j < k; ++j) {
        const int r = inv[(size_t) t * k + j];
        uint4* dst = reinterpret_cast<uint4*>(xs + (size_t) r * d + (size_t) blk * 32);
#pragma unroll
        for (int c = 0; c < 4; ++c) dst[c] = reinterpret_cast<const uint4*>(h)[c];
    }
}

template <int MODE_, int WM_, int NT_, int LW_, bool SACC_, int ABL_ = 0, int BP_ = 0, int SYNC_ = 0, int RF_ = 0>
struct Cfg {
    static constexpr int RF = RF_;              // refills: 0 warp-cooperative (8 lanes a line), 1 a lane its own row
    static constexpr int MODE = MODE_, WM = WM_, NT = NT_, LW = LW_, ABL = ABL_;
    static constexpr int BP = BP_;              // B lines prefetched into the L1 this many k32 blocks ahead (0: none)
    static constexpr int SYNC = SYNC_;          // a block barrier every SYNC k-tiles (0: none)
    static constexpr bool SACC = SACC_;
    static constexpr int MT = 2;                // m16 tiles a warp (gate_up: gate + up)
    static constexpr int RW = 32;               // weight rows a warp
    static constexpr int BM = WM * RW;          // weight rows a block
    static constexpr int BN = NT * 8;           // activation rows a block
    static constexpr int THREADS = WM * 32;
    static constexpr int RB = 256;              // ring bytes a row
    static constexpr int WROW = 36;             // bytes of a row's 64-value k-tile
    static constexpr int SMEM = BM * RB;
    static_assert(WROW * LW + WROW - 1 + 128 <= RB, "ring: the live span");
};

template <class C>
__global__ void __launch_bounds__(C::THREADS, 1) gemm_kernel(const Args a) {
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int s_tile[3];
    constexpr int NT = C::NT;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wt = blockIdx.x, tile = blockIdx.y;
    if (warp == 0) {
        if (lane == 0) s_tile[0] = -1;
        __syncwarp();
        int base = 0;
        for (int e0 = 0; e0 < a.n_expert; e0 += 32) {
            const int e = e0 + lane;
            int lo = 0, hi = 0;
            if (e < a.n_expert) {
                lo = a.bounds[e];
                hi = a.bounds[e + 1];
            }
            const int nt = (hi - lo + C::BN - 1) / C::BN;
            int inc = nt;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, inc, o);
                if (lane >= o) inc += v;
            }
            const int ex = base + inc - nt;
            if (tile >= ex && tile < ex + nt) {
                const int r0 = lo + (tile - ex) * C::BN;
                s_tile[0] = e;
                s_tile[1] = r0;
                s_tile[2] = min(C::BN, hi - r0);
            }
            base += __shfl_sync(0xffffffffu, inc, 31);
            if (base > tile) break;
        }
    }
    __syncthreads();
    const int e = s_tile[0];
    if (e < 0) return;
    const int row0 = s_tile[1], cnt = s_tile[2];
    const int NK = a.K / 64;
    const int rbytes = (int) a.row_bytes;

    // the warp's 32 weight rows: local rows 0..15 = m16 tile 0, 16..31 = m16 tile 1. Lane l owns local row l for the
    // refills: its pointer, its offset in a 128-byte line (ro), the next line to fetch and the k-tile that first needs it
    const int hb = wt * C::WM + warp;   // gate_up: the half hidden block this warp produces
    int growl;
    if constexpr (C::MODE == 0) {
        const int u = 32 * (hb >> 1) + fval(16 * (hb & 1) + (lane & 15));
        growl = lane < 16 ? u : a.ff + u;
    } else {
        growl = wt * C::BM + warp * C::RW + lane;
    }
    const uint8_t* myrow = a.w + (size_t) e * a.expert_bytes + (size_t) growl * a.row_bytes;
    const int myro = (int) (reinterpret_cast<uintptr_t>(myrow) & 127);
    // line l holds row bytes [128 l - ro, 128 l - ro + 128); k-tile k reads bytes [36 k, 36 k + 36)
    auto kneed = [&](int l) {
        if (128 * l - myro >= rbytes) return 1 << 30;   // past the row
        const int x = 128 * l - myro - 35;
        return x <= 0 ? 0 : (x + 35) / 36;
    };
    int nl = 0, kn = 0;   // kneed(0) = 0
    uint8_t* ring = smem + warp * C::RW * C::RB;
    const uint32_t ring_s = smem_u32(ring);
    // row byte b of local row r: 16-byte unit (b >> 4) & 15 of the row's ring, XOR-swizzled by r & 7
    auto ring_pos = [](int r, int b) { return r * C::RB + ((((b >> 4) & 15) ^ (r & 7)) << 4) + (b & 15); };
    // copy the lines first needed by k-tile j: rows in groups of 4, 8 lanes a row (one request a line)
    constexpr int UNIT = C::MODE == 0 ? 16 : 4;   // gate_up rows are 16-byte aligned, down rows only 4-byte
    auto issue = [&](int j) {
        if constexpr (C::ABL == 4 || C::ABL == 7 || C::ABL == 10) return;
        unsigned mask = __ballot_sync(0xffffffffu, kn <= j);
        while (mask) {
            // the next (up to) four rows of the mask; lanes 8 q .. 8 q + 7 copy row rq
            int rq = -1;
            unsigned m = mask;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = m ? __ffs(m) - 1 : -1;
                if (q == (lane >> 3)) rq = r;
                if (m) m &= m - 1;
            }
            const int src = rq < 0 ? 0 : rq;
            const unsigned long long rp = __shfl_sync(0xffffffffu, (unsigned long long) myrow, src);
            const int ro = __shfl_sync(0xffffffffu, myro, src), l = __shfl_sync(0xffffffffu, nl, src);
            if (rq >= 0) {
                const int b0 = 128 * l - ro;
#pragma unroll
                for (int c = 0; c < 16 / UNIT; ++c) {
                    const int b = b0 + (UNIT == 16 ? 16 * (lane & 7) : 4 * ((lane & 7) + 8 * c));
                    if (b >= 0 && b < rbytes)
                        cp_async<UNIT>(ring_s + ring_pos(rq, b), reinterpret_cast<const uint8_t*>(rp) + b, true);
                }
            }
            mask = m;
        }
        if (kn <= j) {   // the rows just copied move on to their next line
            ++nl;
            kn = kneed(nl);
        }
    };
    // RF 1 (16-byte rows): each lane copies its own row's line, 8 units in 8 instructions - no ballot or shuffles
    auto issue_own = [&](int j) {
        if constexpr (C::ABL == 4 || C::ABL == 7 || C::ABL == 10) return;
        while (kn <= j) {   // twice at most (k-tile 0)
            const int b0 = 128 * nl - myro;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int b = b0 + 16 * u;
                if (b >= 0 && b < rbytes) cp_async<16>(ring_s + ring_pos(lane, b), myrow + b, true);
            }
            ++nl;
            kn = kneed(nl);
        }
    };
    // k-tile 0's lines (one or two a row) are issued with two calls; afterwards one line a row at most a k-tile
    auto issue_all = [&](int j) {
        if constexpr (C::RF == 1 && UNIT == 16) {
            issue_own(j);
        } else {
            issue(j);
            if (__any_sync(0xffffffffu, kn <= j)) issue(j);
        }
    };

    const int g = lane >> 2, t = lane & 3;
    // B: n8 tile j holds activation rows 8 j + g; rows past the tile read row cnt - 1 (finite, discarded)
    const __half* bbase = a.x + (size_t) row0 * a.K + 8 * t;
    uint32_t boffs[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) boffs[j] = (uint32_t) (min(8 * j + g, cnt - 1) * a.K);

    float acc[2][NT][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[mt][j][q] = 0.f;

    auto mainloop = [&](auto nv_tag) {
        constexpr int NV = decltype(nv_tag)::value;   // n8 tiles computed (the rest hold no row of this tile)
        auto nvalid = [&](int j) { return j < NV; };
        auto load_b = [&](uint32_t (&bf)[NT][4], int k32) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (!nvalid(j)) break;
                if constexpr (C::ABL == 3 || C::ABL == 5 || C::ABL == 7 || C::ABL == 9) {
                    bf[j][0] = bf[j][1] = bf[j][2] = bf[j][3] = (uint32_t) k32 + j;
                    continue;
                }
                const uint4 v = __ldg(reinterpret_cast<const uint4*>(bbase + boffs[j] + 32 * k32));
                bf[j][0] = v.x;
                bf[j][1] = v.y;
                bf[j][2] = v.z;
                bf[j][3] = v.w;
            }
        };
        auto prefetch_b = [&](int k32) {
            if constexpr (C::BP == 0) return;
            if (k32 >= 2 * NK || (k32 & 1)) return;   // a 128-byte line: two k32 blocks of a row
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (!nvalid(j)) break;
                if (t == 0) asm volatile("prefetch.global.L1 [%0];" ::"l"(bbase + boffs[j] + 32 * k32));
            }
        };
        auto compute = [&](int kt, int kb, const uint32_t (&bf)[NT][4]) {
            const int bo = C::WROW * kt + 18 * kb;   // the block's first byte in a row
            // the same ring offsets for this lane's four rows (all of them have r & 7 == g)
            auto off = [&](int b) { return ((((b >> 4) & 15) ^ g) << 4) + (b & 15); };
            int o0, o1 = 0;
            const int od = off(bo);
            if (kb == 0) {
                o0 = off(bo + 4 * t);
                o1 = off(bo + 4 * t + 4);
            } else {
                o0 = off(bo + 2 + 4 * t);
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                uint32_t qv[2];
                __half dv[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint8_t* rowp = ring + (16 * mt + g + 8 * h) * C::RB;
                    if (C::ABL == 2 || C::ABL == 8 || C::ABL == 9 || C::ABL == 10) {
                        qv[h] = (uint32_t) (bo * 0x01010101);
                        dv[h] = __ushort_as_half(0x3c00);
                    } else if (kb == 0) {
                        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(rowp + o0);
                        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(rowp + o1);
                        qv[h] = __byte_perm(w0, w1, 0x5432);
                        dv[h] = *reinterpret_cast<const __half*>(rowp + od);
                    } else {
                        qv[h] = *reinterpret_cast<const uint32_t*>(rowp + o0);
                        dv[h] = *reinterpret_cast<const __half*>(rowp + od);
                    }
                }
                if constexpr (C::ABL == 8 || C::ABL == 9 || C::ABL == 10) {
                    acc[mt][0][0] += __uint_as_float(qv[0] ^ qv[1] ^ bf[0][0]);
                    continue;
                }
                uint32_t l0[2], l1[2], h0[2], h1[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    dq(qv[h], l0[h], l1[h], h0[h], h1[h]);
                    if constexpr (!C::SACC) {
                        const __half2 s2 = __half2half2(dv[h]);
                        l0[h] = hmul2u(l0[h], s2);
                        l1[h] = hmul2u(l1[h], s2);
                        h0[h] = hmul2u(h0[h], s2);
                        h1[h] = hmul2u(h1[h], s2);
                    }
                }
                const uint32_t A0[4] = {l0[0], l0[1], l1[0], l1[1]};
                const uint32_t A1[4] = {h0[0], h0[1], h1[0], h1[1]};
                const float d0 = __half2float(dv[0]), d1 = __half2float(dv[1]);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    if (!nvalid(j)) break;
                    if constexpr (C::SACC) {
                        float tmp[4];
                        mma16816_z(tmp, A0, bf[j][0], bf[j][1]);
                        mma16816(tmp, A1, bf[j][2], bf[j][3]);
                        acc[mt][j][0] = fmaf(d0, tmp[0], acc[mt][j][0]);
                        acc[mt][j][1] = fmaf(d0, tmp[1], acc[mt][j][1]);
                        acc[mt][j][2] = fmaf(d1, tmp[2], acc[mt][j][2]);
                        acc[mt][j][3] = fmaf(d1, tmp[3], acc[mt][j][3]);
                    } else if (C::ABL == 6) {
                        acc[mt][j][0] += __uint_as_float(A0[0] ^ bf[j][0]);
                        acc[mt][j][1] += __uint_as_float(A1[1] ^ bf[j][3]);
                    } else {
                        mma16816(acc[mt][j], A0, bf[j][0], bf[j][1]);
                        mma16816(acc[mt][j], A1, bf[j][2], bf[j][3]);
                    }
                }
            }
        };

        // the pipeline: k-tile j's lines are issued LW k-tiles ahead (one cp.async group a k-tile); B is loaded one
        // k32 block at a time into one buffer (the scoreboard orders the reuse), its lines prefetched BP blocks ahead
#pragma unroll
        for (int j = 0; j < C::LW; ++j) {
            if (j < NK) issue_all(j);
            cp_async_commit();
        }
#pragma unroll
        for (int k = 0; k < C::BP; ++k) prefetch_b(k);
        uint32_t B0[NT][4];
        for (int kt = 0; kt < NK; ++kt) {
            if constexpr (C::SYNC > 0)
                if (kt % C::SYNC == C::SYNC - 1) __syncthreads();   // keeps the warps (and the L1) together
            if (kt + C::LW < NK) issue_all(kt + C::LW);
            cp_async_commit();
            prefetch_b(2 * kt + C::BP);
            load_b(B0, 2 * kt);
            cp_async_wait<C::LW>();
            __syncwarp();
            compute(kt, 0, B0);
            prefetch_b(2 * kt + 1 + C::BP);
            load_b(B0, 2 * kt + 1);
            compute(kt, 1, B0);
        }
    };
    // the n8 tiles that hold a row, rounded up to a pair (a partial tile's missing rows read row cnt - 1, discarded):
    // one instantiation per count, no per-tile checks in the loop
    static_assert(NT == 8, "the dispatch below covers NT = 8");
    const int nv = (cnt + 7) / 8;
    if (nv > 6) mainloop(std::integral_constant<int, 8>{});
    else if (nv > 4) mainloop(std::integral_constant<int, 6>{});
    else if (nv > 2) mainloop(std::integral_constant<int, 4>{});
    else mainloop(std::integral_constant<int, 2>{});

    cp_async_wait<0>();
    __syncwarp();
    if constexpr (C::MODE == 0) {
        // GeGLU, staged in the warp's ring (fp16, [token][16]), then 32 bytes a row of the hidden
        __half* hs = reinterpret_cast<__half*>(ring);
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int tok = 8 * j + 2 * t + (q & 1);
                hs[tok * 16 + g + 8 * (q >> 1)] = clamp_half(gelu_tanh(acc[0][j][q]) * acc[1][j][q]);
            }
        __syncwarp();
        __half* dst = a.hidden + (size_t) row0 * a.ff + 32 * (hb >> 1) + 16 * (hb & 1);
        for (int tok = lane >> 1; tok < cnt; tok += 16)
            *reinterpret_cast<uint4*>(dst + (size_t) tok * a.ff + 8 * (lane & 1)) =
                *reinterpret_cast<const uint4*>(hs + tok * 16 + 8 * (lane & 1));
    } else {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int tok = 8 * j + 2 * t + (q & 1);
                    if (tok < cnt)
                        a.ey[(size_t) (row0 + tok) * a.ld_ey + wt * C::BM + warp * C::RW + 16 * mt + g + 8 * (q >> 1)] =
                            acc[mt][j][q];
                }
    }
}

template <class C> inline dim3 grid_of(int64_t rows, int n_expert, int n_out_rows) {
    const int wtiles = C::MODE == 0 ? n_out_rows / (C::BM / 2) : n_out_rows / C::BM;
    return dim3((unsigned) wtiles, (unsigned) ((rows + C::BN - 1) / C::BN + n_expert));
}

}  // namespace strata::gemma::w4a16::dev
