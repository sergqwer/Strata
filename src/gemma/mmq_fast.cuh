// src/gemma/mmq_fast.cuh - the prompt path's Q4_0 / Q8_0 x q8_1 products, faster than llama.cpp's MMQ kernel and
// bit-identical to it.  Included by mmq.cu only.
//
// The contract (llama.cpp's mul_mat_q on sm_80+, tiling path, vec_dot_q8_0_q8_1_mma):  for weight row i and
// activation column j,
//     sum = 0
//     for kb in 0 .. KB-1 (every 32-value block of the row, KB rounded up to a multiple of 8 as MMQ's 256-value
//                          iterations read them - the blocks past the row end are read from where MMQ reads them):
//         c   = dot(int8 weights of block kb, int8 activations of block kb)      (exact int32)
//         sum = fma(float(c) * dA[i][kb], dB[j][kb], sum)                         (FMUL.FTZ, FFMA.FTZ)
//     dst[ids[j]][i] = sum
// with dA = the weight block's half scale as float and dB = the q8_1 block's scale (float in the D4 layout, the low
// half of half2 in DS4).  Every value is computed in this order and nothing else enters it, so how the tile is cut,
// staged or scheduled does not change a bit; only splitting k does (llama.cpp's stream-k), which this code mirrors
// where llama.cpp does it (Tile::seg).
//
// What makes it faster than llama.cpp's kernel (one block of 8 warps per SM that loads, syncs and computes in
// lockstep):
//  - 16 warps; the activations stream through a 3-stage cp.async ring, the weights through registers a stage ahead;
//  - float(c) without a conversion: the IMMA accumulator starts at 0x4B400000, the float with those bits is
//    1.5 * 2^23 + c exactly (|c| < 2^22), and one FADD takes the 1.5 * 2^23 off;
//  - Q4_0 nibbles go to the tensor cores unsigned (u8 x s8): the -8 of every weight is folded into the same
//    accumulator start, 0x4B400000 - 8 * (sum of the block's activations), computed once per stage and column, so
//    a weight word costs 3 integer ops instead of 7 and the int32 c is still exactly dot(q - 8, y);
//  - the activation scales (DS4 halves) are made floats once per stage and column, not once per warp.
// The tile is 128 weight rows x J activation columns, k in stages of 128 values (4 blocks).
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace strata::gemma::mmq::fast {

constexpr int kThreads = 512;       // 16 warps, one tile per SM
constexpr int kBM = 128;            // weight rows per tile
constexpr int kRowB = 144;          // shared bytes per row of a stage: A = 128 int8 + 4 float scales, Y = the q8_1 block
constexpr int kAStages = 2, kYStages = 3, kPStages = 2;
constexpr int kMagic = 0x4B400000;  // 1.5 * 2^23

template <ggml_type T> struct WTraits;
template <> struct WTraits<GGML_TYPE_Q4_0> {   // {half d; uint8 qs[16]}: 18 bytes
    static constexpr int bytes = 18, words = 5;   // a thread stages one block from 20 aligned bytes
    static constexpr bool q4 = true;              // q8_1 activations in the DS4 layout (half2 d, s)
};
template <> struct WTraits<GGML_TYPE_Q8_0> {   // {half d; int8 qs[32]}: 34 bytes
    static constexpr int bytes = 34, words = 9;
    static constexpr bool q4 = false;             // D4 layout (float d)
};

// shared memory: A stages, Y stages, [Q4_0: per stage the accumulator starts and float scales of every (block,
// column)], the tile's dst rows
template <ggml_type T, bool U8 = false> __host__ __device__ constexpr int smem_bytes(int J) {
    return kAStages * kBM * kRowB + kYStages * J * kRowB + (WTraits<T>::q4 && U8 ? kPStages * 4 * J * 8 : 0) + J * 4;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// D = A (16x32, row) * B (32x8 s8, col) + C, C per column: {c0, c1, c0, c1} (rows g and g+8 of columns 2t, 2t+1)
template <bool U8>
__device__ __forceinline__ void imma(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int c0, int c1) {
    if constexpr (U8)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                     "{%10, %11, %10, %11};\n"
                     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    else
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                     "{%10, %11, %10, %11};\n"
                     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}
// Q4_0 weights as signed int8 (llama.cpp's __vsubss4(q & 15, 8), no value saturates): b + 0x78 never carries out of a
// byte, ^0x80 then subtracts 128
__device__ __forceinline__ uint32_t q4_lo(uint32_t q) { return ((q & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u; }
__device__ __forceinline__ uint32_t q4_hi(uint32_t q) { return (((q >> 4) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u; }

// the int32 dot as float, exactly what llama.cpp's I2FP gives: d = c + 0x4B400000 as bits = 1.5 * 2^23 + c
__device__ __forceinline__ float c_float(int d) { return __fsub_rn(__int_as_float(d), 12582912.0f); }

// a work item: weight rows [it*128, +128) of matrix `expert`, activation columns [col0, col0 + ncols) through a tile
// J wide; seg > 0: llama.cpp's stream-k split this tile at k block `seg` (its two partials meet as its fixup adds them)
struct Tile {
    int expert, it, col0, ncols, seg;
};

struct Args {
    const char* x;          // weights
    const char* y;          // q8_1 activations, MMQ layout: [k group of 128][column] blocks of 144 bytes
    const int32_t* ids;     // dst row of activation column (sorted row) c
    float* dst;
    int nrows;              // weight rows (the rows past it in the last row tile are clamped, never written)
    int stride_row;         // blocks per weight row
    long long stride_expert;   // blocks per expert matrix
    int ncols_y;            // columns of y (its k-group stride)
    int ld_dst;
    int nkt;                // k stages of 128 values: ceil(blocks / 8) * 2, i.e. MMQ's 256-value iterations
};

// warps over the tile: 4 x 4 (warp tiles of 32 rows) where J/4 is a multiple of 8, else 8 x 2 (16 rows)
template <int J> struct Shape {
    static_assert(J % 16 == 0 && J >= 16 && J <= 128, "tile width");
    static constexpr int WN = J % 32 == 0 ? 4 : 2, WM = 16 / WN;
    static constexpr int WROWS = kBM / WM, WCOLS = J / WN;
    static constexpr int MT = WROWS / 16, NT = WCOLS / 8;   // m16 / n8 fragments per warp
};

// One tile.  The caller has __syncthreads()-ed since the last use of smem.
template <ggml_type T, int J, bool SEG, bool U8 = false>
__device__ __forceinline__ void tile(const Args& a, const Tile& t, unsigned char* smem) {
    using W = WTraits<T>;
    using S = Shape<J>;
    constexpr int MT = S::MT, NT = S::NT;
    constexpr bool Q4 = W::q4 && U8;   // the unsigned-nibble path with the per-column accumulator start
    constexpr int kYChunks = J * (kRowB / 16), kYPer = (kYChunks + kThreads - 1) / kThreads;

    unsigned char* As = smem;
    unsigned char* Ys = As + kAStages * kBM * kRowB;
    int* Pc = (int*) (Ys + kYStages * J * kRowB);             // [stage][kb][col] accumulator start (Q4_0)
    float* Pd = (float*) (Pc + (Q4 ? kPStages * 4 * J : 0));  // [stage][kb][col] activation scale (Q4_0)
    int32_t* ids_s = (int32_t*) (Pd + (Q4 ? kPStages * 4 * J : 0));

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / S::WN, wn = warp % S::WN;
    const int g = lane >> 2, tq = lane & 3;

    for (int j = tid; j < J; j += kThreads) ids_s[j] = a.ids[t.col0 + min(j, t.ncols - 1)];

    // weights: thread -> (row r, block b of the stage's 4).  A block pair is 4-byte aligned (2*18, 2*34 bytes): the
    // even block is read from the pair's start, the odd one from 16 / 32 bytes in, and its qs words line up there
    const int r = tid >> 2, b = tid & 3, odd = b & 1;
    const uint32_t* xsrc = (const uint32_t*) (a.x + ((long long) t.expert * a.stride_expert +
                                                      (long long) min(t.it * kBM + r, a.nrows - 1) * a.stride_row) * W::bytes +
                                              (b >> 1) * 2 * W::bytes + odd * (W::q4 ? 16 : 32));
    const int xsh = odd ? 32 : 16;   // funnel shift: 16 for the even block (qs at byte 2), 32 (the high word) for the odd
    unsigned char* xdst = As + r * kRowB + b * 32;
    uint32_t xw[W::words];
    auto load_x = [&](int kt) {
        const uint32_t* p = xsrc + kt * (W::bytes);   // a stage is 4 blocks = W::bytes words
#pragma unroll
        for (int i = 0; i < W::words; ++i) xw[i] = __ldg(p + i);
    };
    auto store_x = [&](int buf) {
        unsigned char* d = xdst + buf * (kBM * kRowB);
        const float dd = __half2float(__ushort_as_half((unsigned short) ((odd ? xw[0] >> 16 : xw[0]) & 0xFFFFu)));
        if constexpr (W::q4) {   // 16 qs bytes: low nibbles are values 0-15, high ones 16-31
            uint32_t q[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) q[i] = __funnelshift_rc(xw[i], xw[i + 1], xsh);
            if constexpr (U8) {
                const uint32_t m = 0x0F0F0F0Fu;
                *(uint4*) (d) = make_uint4(q[0] & m, q[1] & m, q[2] & m, q[3] & m);
                *(uint4*) (d + 16) = make_uint4((q[0] >> 4) & m, (q[1] >> 4) & m, (q[2] >> 4) & m, (q[3] >> 4) & m);
            } else {
                *(uint4*) (d) = make_uint4(q4_lo(q[0]), q4_lo(q[1]), q4_lo(q[2]), q4_lo(q[3]));
                *(uint4*) (d + 16) = make_uint4(q4_hi(q[0]), q4_hi(q[1]), q4_hi(q[2]), q4_hi(q[3]));
            }
        } else {                 // 32 int8
            uint32_t q[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) q[i] = __funnelshift_rc(xw[i], xw[i + 1], xsh);
            *(uint4*) (d) = make_uint4(q[0], q[1], q[2], q[3]);
            *(uint4*) (d + 16) = make_uint4(q[4], q[5], q[6], q[7]);
        }
        *(float*) (As + buf * (kBM * kRowB) + r * kRowB + 128 + 4 * b) = dd;
    };
    // activations: the stage's J blocks of 144 bytes are contiguous in y
    const char* ysrc = a.y + (long long) t.col0 * kRowB + tid * 16;
    const long long ystage = (long long) a.ncols_y * kRowB;
    const uint32_t ydst = smem_u32(Ys) + tid * 16;
    auto load_y = [&](int kt, int buf) {
        const char* src = ysrc + kt * ystage;
#pragma unroll
        for (int u = 0; u < kYPer; ++u)
            if (tid + u * kThreads < kYChunks) cp_async16(ydst + buf * (J * kRowB) + u * kThreads * 16, src + u * kThreads * 16);
    };
    // Q4_0: per (block, column) of a landed stage: the accumulator start (folds the weights' -8) and the float scale
    auto prep_y = [&](int buf, int pbuf) {
        if constexpr (Q4) {
            for (int p = tid; p < 4 * J; p += kThreads) {
                const int kb = p / J, c = p - kb * J;
                const unsigned char* blk = Ys + buf * (J * kRowB) + c * kRowB;
                const uint4 w0 = *(const uint4*) (blk + 16 + kb * 32), w1 = *(const uint4*) (blk + 16 + kb * 32 + 16);
                const int ones = 0x01010101;
                int sy = __dp4a((int) w0.x, ones, 0);
                sy = __dp4a((int) w0.y, ones, sy);
                sy = __dp4a((int) w0.z, ones, sy);
                sy = __dp4a((int) w0.w, ones, sy);
                sy = __dp4a((int) w1.x, ones, sy);
                sy = __dp4a((int) w1.y, ones, sy);
                sy = __dp4a((int) w1.z, ones, sy);
                sy = __dp4a((int) w1.w, ones, sy);
                Pc[pbuf * 4 * J + p] = kMagic - 8 * sy;
                Pd[pbuf * 4 * J + p] = __low2float(*(const half2*) (blk + kb * 4));
            }
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int l = 0; l < 4; ++l) acc[i][j][l] = 0.0f;
    float keep[SEG ? MT : 1][SEG ? NT : 1][4];   // stream-k: the partial of the tile's first segment
    const int seg_kt = SEG && t.seg > 0 ? t.seg / 4 : -1;

    // prologue: stages 0 and 1 in flight, stage 0 staged
    load_y(0, 0);
    cp_async_commit();
    if (a.nkt > 1) load_y(1, 1);
    cp_async_commit();
    load_x(0);
    store_x(0);
    if (a.nkt > 1) load_x(1);
    if constexpr (Q4) {
        cp_async_wait<1>();
        __syncthreads();
        prep_y(0, 0);
    }

    const int arow0 = wm * S::WROWS, bcol0 = wn * S::WCOLS;
    const uint32_t As_u = smem_u32(As), Ys_u = smem_u32(Ys);
    const uint32_t a_ld = As_u + (arow0 + (lane & 7) + 8 * ((lane >> 3) & 1)) * kRowB + 16 * (lane >> 4);
    const uint32_t b_ld = Ys_u + (bcol0 + (lane & 7) + 8 * (lane >> 4)) * kRowB + 16 + 16 * ((lane >> 3) & 1);
    const uint32_t b_ld2 = Ys_u + (bcol0 + (lane & 7)) * kRowB + 16 + 16 * ((lane >> 3) & 1);
    const unsigned char* a_sc = As + (arow0 + g) * kRowB + 128;
    const int p_col = bcol0 + 2 * tq;

    for (int kt = 0; kt < a.nkt; ++kt) {
        // Q4_0: stage kt+1 must have landed (it is prepared in this iteration); otherwise stage kt
        if constexpr (Q4) cp_async_wait<0>();
        else cp_async_wait<1>();
        __syncthreads();
        if (kt + 1 < a.nkt) {
            store_x((kt + 1) & 1);
            prep_y((kt + 1) % kYStages, (kt + 1) & 1);
        }
        if (kt + 2 < a.nkt) {
            load_x(kt + 2);
            load_y(kt + 2, (kt + 2) % kYStages);
        }
        cp_async_commit();

        if (SEG && kt == seg_kt) {   // llama.cpp's stream-k: the k range [0, seg) was another block's partial
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        if constexpr (SEG) keep[i][j][l] = acc[i][j][l];
                        acc[i][j][l] = 0.0f;
                    }
        }

        const uint32_t abuf = (kt & 1) * (kBM * kRowB), ybuf = (kt % kYStages) * (J * kRowB);
        const int pbuf = (kt & 1) * 4 * J;
        // the weight scales of this warp's rows for the stage's 4 blocks
        float4 da4[MT][2];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            da4[i][0] = *(const float4*) (a_sc + abuf + i * 16 * kRowB);
            da4[i][1] = *(const float4*) (a_sc + abuf + (i * 16 + 8) * kRowB);
        }
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            uint32_t af[MT][4];
            float da[MT][2];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                ldsm_x4(a_ld + abuf + i * 16 * kRowB + kb * 32, af[i][0], af[i][1], af[i][2], af[i][3]);
                da[i][0] = kb == 0 ? da4[i][0].x : kb == 1 ? da4[i][0].y : kb == 2 ? da4[i][0].z : da4[i][0].w;
                da[i][1] = kb == 0 ? da4[i][1].x : kb == 1 ? da4[i][1].y : kb == 2 ? da4[i][1].z : da4[i][1].w;
            }
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                const bool two = j + 1 < NT;
                uint32_t b00, b01, b10 = 0, b11 = 0;
                if (two) ldsm_x4(b_ld + ybuf + j * 8 * kRowB + kb * 32, b00, b01, b10, b11);
                else ldsm_x2(b_ld2 + ybuf + j * 8 * kRowB + kb * 32, b00, b01);
                int2 cs[2] = {make_int2(kMagic, kMagic), make_int2(kMagic, kMagic)};   // accumulator starts, columns 2t, 2t+1
                float2 db[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};          // their activation scales
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    if (u == 1 && !two) continue;
                    const int col = p_col + (j + u) * 8;
                    if constexpr (Q4) {
                        cs[u] = *(const int2*) (Pc + pbuf + kb * J + col);
                        db[u] = *(const float2*) (Pd + pbuf + kb * J + col);
                    } else if constexpr (W::q4) {
                        const unsigned char* hp = Ys + ybuf + col * kRowB + kb * 4;
                        db[u] = make_float2(__low2float(*(const half2*) hp), __low2float(*(const half2*) (hp + kRowB)));
                    } else {
                        const unsigned char* hp = Ys + ybuf + col * kRowB + kb * 4;
                        db[u] = make_float2(*(const float*) hp, *(const float*) (hp + kRowB));
                    }
                }
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    int c0[4], c1[4];
                    imma<Q4>(c0, af[i], b00, b01, cs[0].x, cs[0].y);
                    if (two) imma<Q4>(c1, af[i], b10, b11, cs[1].x, cs[1].y);
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const float dbv0 = (l & 1) ? db[0].y : db[0].x;
                        acc[i][j][l] = __fmaf_rn(__fmul_rn(c_float(c0[l]), da[i][l >> 1]), dbv0, acc[i][j][l]);
                        if (two) {
                            const float dbv1 = (l & 1) ? db[1].y : db[1].x;
                            acc[i][j + 1][l] = __fmaf_rn(__fmul_rn(c_float(c1[l]), da[i][l >> 1]), dbv1, acc[i][j + 1][l]);
                        }
                    }
                }
            }
        }
    }

    if constexpr (SEG) {
        if (seg_kt >= 0) {   // llama.cpp's fixup: dst = last segment + (0 + first segment)
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int l = 0; l < 4; ++l) acc[i][j][l] = __fadd_rn(acc[i][j][l], __fadd_rn(0.0f, keep[i][j][l]));
        }
    }

    // write back: element l of fragment (i, j) is weight row g (+8 for l >= 2), column 2*tq (+1 for odd l)
    const int rbase = t.it * kBM + arow0 + g;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int cc = bcol0 + j * 8 + 2 * tq + v;
            if (cc >= t.ncols) continue;
            float* drow = a.dst + (long long) ids_s[cc] * a.ld_dst;
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int rr = rbase + i * 16 + 8 * u;
                    if (rr < a.nrows) drow[rr] = acc[i][j][2 * u + v];
                }
        }
}

}  // namespace strata::gemma::mmq::fast
