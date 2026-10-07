// tools/gemma/w4a16/v1.cuh - the first w4a16 expert GEMM (kept for the harness's comparisons; production runs the
// kernel in src/gemma/moe_w4a16.cuh): 128 weight rows x 64 activation rows a block, 8 warps, both operands staged
// through shared memory with a block-wide cp.async pipeline (64-value k-tiles), activations stored with every 16
// values as evens then odds (one ldmatrix gives a B fragment). Its measured limits (activations through the L2 11x /
// 22x, 36-byte weight k-slices of 128 scattered rows: DRAM at ~200 GB/s) led to the production design.
#pragma once

#include "../../../src/gemma/moe_w4a16.cuh"

namespace strata::gemma::w4a16::dev::v1 {

// storage position of value v (0..15) inside its 16-value group: evens first, then odds
__host__ __device__ constexpr int perm16(int v) { return (v & 1) ? 8 + (v >> 1) : (v >> 1); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}


// ------------------------------------------------------------------------------------------------ the scatter

// one thread per (token, 16-value group): read 16 floats, write them as fp16 (evens, then odds) to each of the token's
// k sorted rows
__global__ void scatter_kernel(const float* __restrict__ g, int d, const int32_t* __restrict__ inv, int tokens, int k,
                               __half* __restrict__ xs) {
    const int groups = d / 16;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) tokens * groups) return;
    const int t = (int) (i / groups), grp = (int) (i % groups);
    const float4* src = reinterpret_cast<const float4*>(g + (size_t) t * d + (size_t) grp * 16);
    float v[16];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 f = src[j];
        v[4 * j] = f.x;
        v[4 * j + 1] = f.y;
        v[4 * j + 2] = f.z;
        v[4 * j + 3] = f.w;
    }
    __align__(16) __half h[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) h[perm16(j)] = clamp_half(v[j]);
    const uint4 o0 = *reinterpret_cast<const uint4*>(h), o1 = *reinterpret_cast<const uint4*>(h + 8);
    for (int j = 0; j < k; ++j) {
        const int r = inv[(size_t) t * k + j];
        uint4* dst = reinterpret_cast<uint4*>(xs + (size_t) r * d + (size_t) grp * 16);
        dst[0] = o0;
        dst[1] = o1;
    }
}


// MODE 0: gate_up + GeGLU -> fp16 hidden; MODE 1: down -> f32 ey.
// KB: Q4_0 blocks per k-tile (even); CPW: bytes per cp.async of weights (the row alignment decides: 4 for 4-byte
// aligned rows, up to 2*KB for 16-byte aligned ones); WM x WN warps, each MT m16 tiles (weight rows) x NT n8 tiles
// (activation rows). gate_up with MT == 1: the first WM/2 warps hold gate rows, the others the matching up rows (paired
// through shared memory in the epilogue); with even MT a warp holds both. SACC: the block scale multiplies the fp32
// partial sum of the block's 32 products (weights exact), else it is folded into the fp16 weights (one rounding).
template <int MODE_, int KB_, int CPW_, int WM_, int WN_, int MT_, int NT_, int STAGES_, bool SACC_, int MINB_ = 1,
          int ABL_ = 0, bool RING_ = false>
struct Cfg {
    // RING: each weight row streams through a ring of RB = 256 bytes in shared memory, refilled a whole 128-byte line at
    // a time and every line once, LW k-tiles ahead of the activations (a k-tile's 36 bytes a row, copied per k-tile,
    // cost 2-3 partial sectors of 128 scattered rows: the DRAM delivered ~200 GB/s for that pattern); 16-byte units,
    // XOR-swizzled by row (bank-conflict free A loads); needs 16-byte aligned rows and 64-value k-tiles
    static constexpr bool RING = RING_;
    static constexpr int RB = 256, LW = 1;
    static constexpr int MODE = MODE_, KB = KB_, CPW = CPW_, WM = WM_, WN = WN_, MT = MT_, NT = NT_, STAGES = STAGES_;
    // ablation for the harness (wrong results): 1 no dequant, 2 no A loads, 3 no B loads, 4 no weight copies,
    // 5 no activation copies, 6 no MMA, 7 no copies at all, 8 copies only, 9 weight copies only, 10 activations only
    static constexpr int ABL = ABL_;
    static constexpr bool SACC = SACC_;
    static constexpr int MINB = MINB_;
    static constexpr int BM = WM * MT * 16;   // weight rows a block (gate_up: half gate, half up)
    static constexpr int BN = WN * NT * 8;    // activation rows a block (one expert)
    static constexpr int KT = KB * 32;        // values a k-tile
    static constexpr int WROW = KB * 18;      // bytes of a weight row's k-tile
    // padded so that the 8 rows (lanes g) x 4 words (lanes t) of one A load hit 32 distinct banks
    static constexpr int WSTRIDE = KB == 2 ? 48 : KB == 4 ? 80 : KB == 8 ? 144 : 0;
    static constexpr int ACH = KT / 8;        // 16-byte chunks of an activation row's k-tile (swizzled: c ^ (row & 7))
    static constexpr int ASTRIDE = KT * 2;
    static constexpr int WBYTES = RING ? BM * RB : BM * WSTRIDE;   // RING: one ring for all stages
    static constexpr int ABYTES = BN * ASTRIDE;
    static constexpr int STAGE = RING ? ABYTES : WBYTES + ABYTES;   // stage s: weights at s * STAGE, activations + WBYTES
    static constexpr int THREADS = WM * WN * 32;
    // copies, lanes along a row (coalesced): NWC weight chunks (WCH = 9 a row) and ACT activation chunks a thread
    static constexpr int WCH = RING ? 1 : WROW / CPW;
    static constexpr int NWC = (BM * WCH + THREADS - 1) / THREADS;
    static constexpr int ACT = (BN * ACH + THREADS - 1) / THREADS;
    // gate_up epilogue: the up half as f32 (MT == 1), then the hidden as fp16, both [BN rows] in shared memory
    static constexpr int USTRIDE = BM / 2 + 4, HSTRIDE = BM / 2 + 8;
    static constexpr int EPI_U = MODE == 0 && MT == 1 ? BN * USTRIDE * 4 : 0;
    static constexpr int EPI = MODE == 0 ? EPI_U + BN * HSTRIDE * 2 : 0;
    static constexpr int MAIN = RING ? WBYTES + STAGES * ABYTES : STAGES * STAGE;
    static constexpr int SMEM = MAIN > EPI ? MAIN : EPI;
    static_assert(!RING || (KB == 2 && WROW * (STAGES + LW - 1) + WROW - 1 + 128 <= RB), "ring: 64-value k-tiles");
    static_assert(KB % 2 == 0 && WSTRIDE >= WROW, "KB: 2, 4 or 8");
    static_assert(RING || WROW % CPW == 0, "weight copy width");
    static_assert(NT % 2 == 0, "ldmatrix.x4 covers two n8 tiles");
    static_assert(MODE == 1 || MT % 2 == 0 || (MT == 1 && WM % 2 == 0), "gate_up: gate and up tiles in pairs");
    static_assert(ACH >= 8, "the swizzle needs 8 chunks a row");
};

template <class C>
__global__ void __launch_bounds__(C::THREADS, C::MINB) gemm_kernel(const Args a) {
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int s_tile[3];
    constexpr int MT = C::MT, NT = C::NT;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % C::WM, wn = warp / C::WM;
    const int wt = blockIdx.x, tile = blockIdx.y;

    // which expert and rows token tile `tile` is: experts in order, ceil(rows_e / BN) tiles each
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
    const int NK = a.K / C::KT;

    // the copies of a stage, as fixed per-thread offsets (+ kt * the k-tile's bytes); consecutive lanes copy consecutive
    // chunks of a row, so a warp's copy touches few cache lines
    const uint8_t* wexp = a.w + (size_t) e * a.expert_bytes;
    uint32_t wsrc[C::NWC], wdst[C::NWC];
#pragma unroll
    for (int j = 0; j < C::NWC; ++j) {
        const int i = tid + j * C::THREADS;
        const int r = min(i / C::WCH, C::BM - 1), c = i % C::WCH;
        int grow;
        if constexpr (C::MODE == 0) grow = r < C::BM / 2 ? wt * (C::BM / 2) + r : a.ff + wt * (C::BM / 2) + r - C::BM / 2;
        else grow = wt * C::BM + r;
        wsrc[j] = (uint32_t) (grow * a.row_bytes) + c * C::CPW;
        wdst[j] = r * C::WSTRIDE + c * C::CPW;
    }
    const __half* asrc[C::ACT];
    uint32_t adst[C::ACT];
    bool aok[C::ACT];
#pragma unroll
    for (int j = 0; j < C::ACT; ++j) {
        const int i = min(tid + j * C::THREADS, C::BN * C::ACH - 1);   // the surplus re-copies the last chunk
        const int r = i / C::ACH, c = i % C::ACH;
        aok[j] = r < cnt;
        asrc[j] = a.x + (size_t) (row0 + (aok[j] ? r : 0)) * a.K + c * 8;
        adst[j] = C::WBYTES + r * C::ASTRIDE + ((c ^ (r & 7)) << 4);
    }
    const uint32_t sbase = smem_u32(smem);
    // RING: 8 lanes copy one row's 128-byte line (one request); a pass covers THREADS / 8 rows
    constexpr int RPP = C::RING ? C::THREADS / 8 : 1;   // rows a pass
    constexpr int RPASS = C::RING ? (C::BM + RPP - 1) / RPP : 0;
    const uint8_t* rptr[RPASS > 0 ? RPASS : 1];
    int roff[RPASS > 0 ? RPASS : 1];
    if constexpr (C::RING) {
#pragma unroll
        for (int i = 0; i < RPASS; ++i) {
            const int r = min((tid >> 3) + i * RPP, C::BM - 1);
            int grow;
            if constexpr (C::MODE == 0) grow = r < C::BM / 2 ? wt * (C::BM / 2) + r : a.ff + wt * (C::BM / 2) + r - C::BM / 2;
            else grow = wt * C::BM + r;
            rptr[i] = wexp + (size_t) grow * a.row_bytes;
            roff[i] = (int) (reinterpret_cast<uintptr_t>(rptr[i]) & 127);
        }
    }
    // the line of row r holding byte b goes to ring units ((b >> 4) & 15) ^ (r & 7)
    auto ring_unit = [](int r, int b) { return r * C::RB + ((((b >> 4) & 15) ^ (r & 7)) << 4); };
    auto load_stage = [&](int stage, int kt) {
        const uint32_t st = sbase + stage * C::STAGE;
        if constexpr (C::RING) {
            if (C::ABL != 4 && C::ABL != 7 && C::ABL != 10) {
                // the lines needed through k-tile kt + LW that k-tile kt - 1's copies did not fetch (k-tile 0: all)
#pragma unroll
                for (int i = 0; i < RPASS; ++i) {
                    const int r = (tid >> 3) + i * RPP;
                    if (C::BM % RPP != 0 && r >= C::BM) break;
                    const int ro = roff[i];
                    const int last = (ro + C::WROW * (kt + C::LW) + C::WROW - 1) >> 7;
                    const int first = kt == 0 ? 0 : ((ro + C::WROW * (kt + C::LW) - 1) >> 7) + 1;
#pragma unroll
                    for (int l = 0; l < 2; ++l) {
                        const int b = 128 * (first + l) - ro + 16 * (tid & 7);   // byte of the row
                        if ((l == 0 || kt == 0) && first + l <= last && b >= 0 && b < (int) a.row_bytes)
                            cp_async<16>(sbase + ring_unit(r, b), rptr[i] + b, true);
                    }
                }
            }
        } else {
            const uint8_t* ws = wexp + (size_t) kt * C::WROW;
#pragma unroll
            for (int j = 0; j < C::NWC; ++j)
                if (C::ABL != 4 && C::ABL != 7 && ((C::BM * C::WCH) % C::THREADS == 0 || j + 1 < C::NWC || tid + j * C::THREADS < C::BM * C::WCH))
                    cp_async<C::CPW>(st + wdst[j], ws + wsrc[j], true);
        }
#pragma unroll
        for (int j = 0; j < C::ACT; ++j)
            if (C::ABL != 5 && C::ABL != 7 && C::ABL != 9) cp_async<16>(st + adst[j], asrc[j] + (size_t) kt * C::KT, aok[j]);
    };

    const int g = lane >> 2, t = lane & 3;
    auto mrow = [&](int mt) {
        if constexpr (C::MODE == 0 && MT == 1) {
            return (wm < C::WM / 2 ? 0 : C::BM / 2) + (wm % (C::WM / 2)) * 16;
        } else if constexpr (C::MODE == 0) {
            const int i = mt % (MT / 2);
            return (mt < MT / 2 ? 0 : C::BM / 2) + (wm * (MT / 2) + i) * 16;
        } else {
            return (wm * MT + mt) * 16;
        }
    };
    uint32_t aoff[MT][2];   // this thread's A words: row g / g+8 of each m16 tile, word t (RING: the row's ring)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h)
            aoff[mt][h] = C::RING ? (mrow(mt) + g + 8 * h) * C::RB : (mrow(mt) + g + 8 * h) * C::WSTRIDE + 4 * t;
    // the warp's n8 tiles are wn, wn + WN, wn + 2 WN, ... (interleaved, so that a short tile still spreads over the
    // warps); n8 tile j holds activation rows 8 (wn + WN j) .. +7. ldmatrix pair jp loads tiles 2 jp and 2 jp + 1.
    const uint32_t boff = C::WBYTES + (8 * wn + 8 * C::WN * (lane >> 4) + (lane & 7)) * C::ASTRIDE;
    constexpr uint32_t BPAIR = 16 * C::WN * C::ASTRIDE;
    const int bl7 = lane & 7, bhi = (lane >> 3) & 1;
    auto nvalid = [&](int j) { return 8 * (wn + C::WN * j) < cnt; };   // warp-uniform
    const bool active = nvalid(0) && C::ABL != 8 && C::ABL != 9 && C::ABL != 10;

    // the fragments of one k32 block (one Q4_0 block of every weight row): raw nibble words and scales, B fragments
    struct Frag {
        uint32_t q[MT][2];
        __half d[MT][2];
        uint32_t b[2][NT][2];   // [k16 step][n8 tile][b0b1 / b2b3]
    };
    auto load_frag = [&](Frag& f, int stage, int kt, int kb) {
        const uint8_t* ws = smem + stage * C::STAGE;
        const int p = kb >> 1;
        // RING: the pair's bytes sit at the same ring bytes in every row, up to the row's unit swizzle
        int o_q0 = 0, o_q1 = 0, o_d = 0;
        if constexpr (C::RING) {
            const int b0 = C::WROW * kt + 36 * p;
            if (kb & 1) {
                o_q0 = b0 + 20 + 4 * t;
                o_d = b0 + 18;
            } else {
                o_q0 = b0 + 4 * t;
                o_q1 = b0 + 4 * t + 4;
                o_d = b0;
            }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint8_t* rp = ws + aoff[mt][h] + 36 * p;   // word t of the pair
                if (C::RING && C::ABL != 2) {
                    const int rr = mrow(mt) + g + 8 * h;
                    auto at = [&](int b) { return smem + ring_unit(rr, b) + (b & 15); };
                    if (kb & 1) {
                        f.q[mt][h] = *reinterpret_cast<const uint32_t*>(at(o_q0));
                    } else {
                        const uint32_t w0 = *reinterpret_cast<const uint32_t*>(at(o_q0));
                        const uint32_t w1 = *reinterpret_cast<const uint32_t*>(at(o_q1));
                        f.q[mt][h] = __byte_perm(w0, w1, 0x5432);
                    }
                    f.d[mt][h] = *reinterpret_cast<const __half*>(at(o_d));
                } else if (C::ABL == 2) {
                    f.q[mt][h] = aoff[mt][h] * 0x01010101u + kb;
                    f.d[mt][h] = __ushort_as_half((unsigned short) (0x3c00 + h));
                } else if (kb & 1) {
                    f.q[mt][h] = *reinterpret_cast<const uint32_t*>(rp + 20);
                    f.d[mt][h] = *reinterpret_cast<const __half*>(rp - 4 * t + 18);
                } else {
                    const uint32_t w0 = *reinterpret_cast<const uint32_t*>(rp);
                    const uint32_t w1 = *reinterpret_cast<const uint32_t*>(rp + 4);
                    f.q[mt][h] = __byte_perm(w0, w1, 0x5432);
                    f.d[mt][h] = *reinterpret_cast<const __half*>(rp - 4 * t);
                }
            }
        const uint32_t as = sbase + stage * C::STAGE + boff;
#pragma unroll
        for (int jp = 0; jp < NT / 2; ++jp) {
            if (!nvalid(2 * jp)) break;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int ks = 2 * kb + s;
                if (C::ABL == 3) {
                    f.b[s][2 * jp][0] = f.b[s][2 * jp][1] = f.b[s][2 * jp + 1][0] = f.b[s][2 * jp + 1][1] = boff + ks;
                    continue;
                }
                ldmatrix_x4(f.b[s][2 * jp][0], f.b[s][2 * jp][1], f.b[s][2 * jp + 1][0], f.b[s][2 * jp + 1][1],
                            as + jp * BPAIR + (((2 * ks + bhi) ^ bl7) << 4));
            }
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[mt][j][q] = 0.f;

    auto compute = [&](const Frag& f) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            uint32_t l0[2], l1[2], h0[2], h1[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (C::ABL == 1) {
                    l0[h] = f.q[mt][h];
                    l1[h] = f.q[mt][h] ^ 0x1111u;
                    h0[h] = f.q[mt][h] ^ 0x2222u;
                    h1[h] = f.q[mt][h] ^ 0x3333u;
                } else {
                    dq(f.q[mt][h], l0[h], l1[h], h0[h], h1[h]);
                }
                if constexpr (!C::SACC && C::ABL != 1) {
                    const __half2 s2 = __half2half2(f.d[mt][h]);
                    l0[h] = hmul2u(l0[h], s2);
                    l1[h] = hmul2u(l1[h], s2);
                    h0[h] = hmul2u(h0[h], s2);
                    h1[h] = hmul2u(h1[h], s2);
                }
            }
            const uint32_t A0[4] = {l0[0], l0[1], l1[0], l1[1]};
            const uint32_t A1[4] = {h0[0], h0[1], h1[0], h1[1]};
            float d0 = 0.f, d1 = 0.f;
            if constexpr (C::SACC) {
                d0 = __half2float(f.d[mt][0]);
                d1 = __half2float(f.d[mt][1]);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (!nvalid(j)) break;
                if constexpr (C::SACC) {
                    float tmp[4];
                    mma16816_z(tmp, A0, f.b[0][j][0], f.b[0][j][1]);
                    mma16816(tmp, A1, f.b[1][j][0], f.b[1][j][1]);
                    acc[mt][j][0] = fmaf(d0, tmp[0], acc[mt][j][0]);
                    acc[mt][j][1] = fmaf(d0, tmp[1], acc[mt][j][1]);
                    acc[mt][j][2] = fmaf(d1, tmp[2], acc[mt][j][2]);
                    acc[mt][j][3] = fmaf(d1, tmp[3], acc[mt][j][3]);
                } else if (C::ABL == 6) {
                    acc[mt][j][0] += __uint_as_float(A0[0] ^ f.b[0][j][0]);
                    acc[mt][j][1] += __uint_as_float(A0[1] ^ f.b[0][j][1]);
                    acc[mt][j][2] += __uint_as_float(A1[2] ^ f.b[1][j][0]);
                    acc[mt][j][3] += __uint_as_float(A1[3] ^ f.b[1][j][1]);
                } else {
                    mma16816(acc[mt][j], A0, f.b[0][j][0], f.b[0][j][1]);
                    mma16816(acc[mt][j], A1, f.b[1][j][0], f.b[1][j][1]);
                }
            }
        }
    };

    // STAGES k-tiles in flight; one barrier a k-tile, after which the buffer just consumed is refilled; the fragments
    // of the next k32 block are loaded before the current one is computed
#pragma unroll
    for (int s = 0; s < C::STAGES; ++s) {
        if (s < NK) load_stage(s, s);
        cp_async_commit();
    }
    cp_async_wait<C::STAGES - 1>();
    __syncthreads();
    Frag F[2];
    if (active) load_frag(F[0], 0, 0, 0);
    for (int kt = 0; kt < NK; ++kt) {
        const int stage = kt % C::STAGES;
#pragma unroll
        for (int kb = 0; kb < C::KB; ++kb) {
            if (kb == C::KB - 1) {
                cp_async_wait<C::STAGES - 2>();
                __syncthreads();
                if (kt + C::STAGES < NK) load_stage(stage, kt + C::STAGES);
                cp_async_commit();
                if (active && kt + 1 < NK) load_frag(F[(kb + 1) & 1], (kt + 1) % C::STAGES, kt + 1, 0);
            } else if (active) {
                load_frag(F[(kb + 1) & 1], stage, kt, kb + 1);
            }
            if (active) compute(F[kb & 1]);
        }
    }
    const int nval = active ? cnt : 0;   // epilogue: rows of this tile (0: the warp holds none)

    if constexpr (C::MODE == 0) {
        // gate row u and up row u give hidden unit u: GeGLU, staged in shared memory as fp16 (reordered per 16), then
        // written out a 16-byte chunk at a time
        cp_async_wait<0>();
        __syncthreads();
        __half* hs = reinterpret_cast<__half*>(smem + C::EPI_U);
        if constexpr (MT == 1) {
            float* us = reinterpret_cast<float*>(smem);
            const int u0 = (wm % (C::WM / 2)) * 16;
            if (wm >= C::WM / 2 && nval > 0) {
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        us[(8 * (wn + C::WN * j) + 2 * t + (q & 1)) * C::USTRIDE + u0 + g + 8 * (q >> 1)] = acc[0][j][q];
            }
            __syncthreads();
            if (wm < C::WM / 2 && nval > 0) {
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int u = u0 + g + 8 * (q >> 1), tok = 8 * (wn + C::WN * j) + 2 * t + (q & 1);
                        const float hv = gelu_tanh(acc[0][j][q]) * us[tok * C::USTRIDE + u];
                        hs[tok * C::HSTRIDE + ((u & ~15) | perm16(u & 15))] = clamp_half(hv);
                    }
            }
        } else if (nval > 0) {
#pragma unroll
            for (int i = 0; i < MT / 2; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int u = (wm * (MT / 2) + i) * 16 + g + 8 * (q >> 1);
                        const int tok = 8 * (wn + C::WN * j) + 2 * t + (q & 1);
                        const float hv = gelu_tanh(acc[i][j][q]) * acc[MT / 2 + i][j][q];
                        hs[tok * C::HSTRIDE + ((u & ~15) | perm16(u & 15))] = clamp_half(hv);
                    }
        }
        __syncthreads();
        constexpr int CH = (C::BM / 2) / 8;
        for (int i = tid; i < cnt * CH; i += C::THREADS) {
            const int tok = i / CH, c = i % CH;
            *reinterpret_cast<uint4*>(a.hidden + (size_t) (row0 + tok) * a.ff + wt * (C::BM / 2) + c * 8) =
                *reinterpret_cast<const uint4*>(hs + tok * C::HSTRIDE + c * 8);
        }
    } else {
        if (nval <= 0) return;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int tok = 8 * (wn + C::WN * j) + 2 * t + (q & 1);
                    if (tok < cnt)
                        a.ey[(size_t) (row0 + tok) * a.ld_ey + wt * C::BM + mrow(mt) + g + 8 * (q >> 1)] = acc[mt][j][q];
                }
    }
}


// the grid of a GEMM over `rows` sorted rows: x = weight tiles, y = token tiles (an upper bound: every expert may end
// in a partial tile; the surplus blocks find no tile and return)
template <class C> inline dim3 grid_of(int64_t rows, int n_expert, int n_out_rows) {
    const int wtiles = C::MODE == 0 ? n_out_rows / (C::BM / 2) : n_out_rows / C::BM;
    return dim3((unsigned) wtiles, (unsigned) ((rows + C::BN - 1) / C::BN + n_expert));
}


}  // namespace strata::gemma::w4a16::dev::v1
