// src/gemma/mmq.cu - see include/strata/gemma/mmq.hpp (adapted from Strata's src/prefill/moe_mmq.cu).  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/gemma/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "strata/gemma/mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include "mmq_fast.cuh"
#include "mmq_magic.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::gemma::mmq {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

// ------------------------------------------------------------------------------------------------ the tile list
//
// llama.cpp launches the MoE product as a grid of (row tile, column tile, expert) with ceil(max_rows / J) column
// tiles for EVERY expert and one tile width J for all of them, chosen for the busiest expert: max_rows has to come
// back to the host (a stream sync per product), most blocks of the small experts exit empty, and with routing as
// skewed as Gemma 4's (the busiest expert gets ~6x the mean) the tiles of the rest are mostly padding (2.4x the real
// columns at 692 tokens).  Here a one-block kernel builds the list of non-empty (expert, column block) tiles on the
// device, each with the width of its own expert's rows, and a persistent grid takes (tile, row block) pairs off an
// atomic counter, biggest tiles first.  Each output value is still computed by llama.cpp's mul_mat_q_process_tile
// over the whole k range in the same order (the tiling-mode path of mul_mat_q, which every MoE product takes: its
// tile count is a multiple of the SM count, so it never splits a tile with stream-k), so it is bit-identical: the
// value of a column depends on that column of the activations and the expert's rows only, not on the tile's width,
// its other columns or where the tile runs.

constexpr int kMaxExperts = 256;
constexpr int kTileCap = 4096;   // tiles in the list: at most total_rows / J_min + n_expert (checked on the host)

struct TileItem {
    int32_t expert, col0, ncols, jc;   // jc: index into the width set
};

// the width sets, widest first (the list is ordered by set index, so the widest tiles are taken first)
//   set 0: up to 128 columns, one block per SM (the J=128 tile needs 57 KB of shared memory)
//   set 1: up to 64 columns, two blocks per SM
__host__ __device__ constexpr int jset_n(int s) { return s == 0 ? 6 : 4; }
__host__ __device__ constexpr int jset_J(int s, int i) {
    return s == 0 ? (i == 0 ? 128 : i == 1 ? 96 : i == 2 ? 64 : i == 3 ? 48 : i == 4 ? 32 : 16)
                  : (i == 0 ? 64 : i == 1 ? 48 : i == 2 ? 32 : 16);
}
constexpr int kJmin = 16;

// tile width per expert: the J of the set with the least ceil(rows / J) * (c0 + J) - c0 is a tile's fixed cost (its
// weight rows are loaded and converted whatever its width) in column units; ties keep the wider tile
__global__ void build_tiles_kernel(const int32_t* __restrict__ bounds, int n_expert, int jset, int c0,
                                   TileItem* __restrict__ items, int cap, int32_t* __restrict__ ctl) {
    __shared__ int s_nt[kMaxExperts], s_jc[kMaxExperts];
    const int nj = jset_n(jset);
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        const int r = bounds[e + 1] - bounds[e];
        int best_jc = 0, best_nt = 0;
        long long best_cost = LLONG_MAX;
        if (r > 0) {
            for (int c = 0; c < nj; ++c) {
                const int J = jset_J(jset, c), nt = (r + J - 1) / J;
                const long long cost = (long long) nt * (c0 + J);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_jc = c;
                    best_nt = nt;
                }
            }
        }
        s_nt[e] = best_nt;
        s_jc[e] = best_jc;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        const int jc = s_jc[e], nt = s_nt[e];
        if (nt == 0) continue;
        int off = 0;   // tiles before this expert's: wider sets first, then lower expert ids
        for (int f = 0; f < n_expert; ++f)
            if (s_jc[f] < jc || (s_jc[f] == jc && f < e)) off += s_nt[f];
        const int J = jset_J(jset, jc), r0 = bounds[e], r = bounds[e + 1] - r0;
        for (int t = 0; t < nt && off + t < cap; ++t) items[off + t] = TileItem{e, r0 + t * J, min(J, r - t * J), jc};
    }
    if (threadIdx.x == 0) {
        int tot = 0;
        for (int f = 0; f < n_expert; ++f) tot += s_nt[f];
        ctl[0] = min(tot, cap);
        ctl[1] = 0;   // the work counter of the product that follows
    }
}

// one tile through llama.cpp's own tile code (the non-stream-k path of mul_mat_q: kb0 from 0 to the end); MAGIC:
// with the conversion-free epilogue of mmq_magic.cuh
template <ggml_type type, int J, bool MAGIC>
__device__ __forceinline__ void tile_one(int* smem, const char* __restrict__ x, const int* __restrict__ y,
                                         const int32_t* __restrict__ ids, float* __restrict__ dst, const TileItem t,
                                         const int it, const int nrows_x, const int stride_row_x, const int ncols_y,
                                         const int stride_col_dst, const int stride_expert_x, const int kb_stop) {
    constexpr bool fallback = false;
    constexpr int nthreads = ggml_cuda_mmq_get_nthreads(type, J, fallback);
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int j = tid; j < J; j += nthreads) smem[j] = ids[t.col0 + min(j, t.ncols - 1)];
    __syncthreads();
    if constexpr (MAGIC)
        magic::process_tile<type, J, fallback, false>(
            x, t.expert * stride_expert_x + it * I * stride_row_x, y + t.col0 * (int) (sizeof(block_q8_1_mmq) / sizeof(int)),
            smem, dst + it * I, stride_row_x, ncols_y, stride_col_dst, nrows_x - it * I - 1, t.ncols - 1, kb_stop, 0);
    else
        mul_mat_q_process_tile<type, J, fallback, false>(
            x, t.expert * stride_expert_x + it * I * stride_row_x, y + t.col0 * (int) (sizeof(block_q8_1_mmq) / sizeof(int)),
            smem, dst + it * I, nullptr, nullptr, stride_row_x, ncols_y, stride_col_dst, nrows_x - it * I - 1, t.ncols - 1,
            0, kb_stop);
}

template <ggml_type type, int S, int MINB, bool MAGIC = false>
__launch_bounds__(256, MINB) __global__
void tiles_kernel(const char* __restrict__ x, const int* __restrict__ y, const int32_t* __restrict__ ids,
                  const TileItem* __restrict__ items, int32_t* __restrict__ ctl, float* __restrict__ dst,
                  const int nrows_x, const int stride_row_x, const int ncols_y, const int stride_col_dst,
                  const int stride_expert_x, const int kb_stop, const int nty) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    extern __shared__ int smem[];
    __shared__ int s_g;
    const int total = ctl[0] * nty;
    for (;;) {
        if (threadIdx.x == 0 && threadIdx.y == 0) s_g = atomicAdd(&ctl[1], 1);
        __syncthreads();
        const int g = s_g;
        if (g >= total) break;
        const int ti = g / nty, it = g - ti * nty;
        const TileItem t = items[ti];
#define STRATA_TILE(Jv) tile_one<type, Jv, MAGIC>(smem, x, y, ids, dst, t, it, nrows_x, stride_row_x, ncols_y, stride_col_dst, stride_expert_x, kb_stop)
        if constexpr (S == 0) {
            switch (t.jc) {
                case 0: STRATA_TILE(128); break;
                case 1: STRATA_TILE(96); break;
                case 2: STRATA_TILE(64); break;
                case 3: STRATA_TILE(48); break;
                case 4: STRATA_TILE(32); break;
                default: STRATA_TILE(16); break;
            }
        } else {
            switch (t.jc) {
                case 0: STRATA_TILE(64); break;
                case 1: STRATA_TILE(48); break;
                case 2: STRATA_TILE(32); break;
                default: STRATA_TILE(16); break;
            }
        }
#undef STRATA_TILE
        __syncthreads();   // the next tile reuses the shared memory and s_g
    }
#else
    NO_DEVICE_CODE;
#endif
}

struct TilesLaunch {
    void (*fn)(const char*, const int*, const int32_t*, const TileItem*, int32_t*, float*, int, int, int, int, int, int, int) = nullptr;
    int smem = 0, grid = 0;
};

template <ggml_type type, int S, int MINB, bool MAGIC = false>
TilesLaunch tiles_launch() {
    static TilesLaunch L = [] {
        TilesLaunch l;
        l.fn = tiles_kernel<type, S, MINB, MAGIC>;
        const int id = ggml_cuda_get_device();
        const int cc = ggml_cuda_info().devices[id].cc, nsm = ggml_cuda_info().devices[id].nsm;
        for (int i = 0; i < jset_n(S); ++i)
            l.smem = std::max(l.smem, (int) mmq_get_nbytes_shared(ggml_cuda_mmq_get_config(type, jset_J(S, i), false, cc), cc));
        ck(cudaFuncSetAttribute(l.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, l.smem), "tiles smem");
        int per_sm = 0;
        ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, l.fn, 256, l.smem), "tiles occupancy");
        l.grid = nsm * std::max(per_sm, 1);
        return l;
    }();
    return L;
}

// ------------------------------------------------------------------------------------------------ fast tile drivers

// the MoE tile list worked off by a persistent grid, as tiles_kernel, through fast::tile
template <ggml_type type, bool U8>
__launch_bounds__(fast::kThreads, 1) __global__
void fast_moe_kernel(const fast::Args a, const TileItem* __restrict__ items, int32_t* __restrict__ ctl, const int nty) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    extern __shared__ __align__(16) unsigned char fsmem[];
    __shared__ int s_g;
    const int total = ctl[0] * nty;
    for (;;) {
        if (threadIdx.x == 0) s_g = atomicAdd(&ctl[1], 1);
        __syncthreads();
        const int gi = s_g;
        if (gi >= total) break;
        const int ti = gi / nty;
        const TileItem it = items[ti];
        const fast::Tile t{it.expert, gi - ti * nty, it.col0, it.ncols, 0};
        switch (it.jc) {   // width set 0
            case 0: fast::tile<type, 128, false, U8>(a, t, fsmem); break;
            case 1: fast::tile<type, 96, false, U8>(a, t, fsmem); break;
            case 2: fast::tile<type, 64, false, U8>(a, t, fsmem); break;
            case 3: fast::tile<type, 48, false, U8>(a, t, fsmem); break;
            case 4: fast::tile<type, 32, false, U8>(a, t, fsmem); break;
            default: fast::tile<type, 16, false, U8>(a, t, fsmem); break;
        }
        __syncthreads();   // the next tile reuses the shared memory and s_g
    }
#else
    NO_DEVICE_CODE;
#endif
}

// a dense product on llama.cpp's own tile grid (J, ntx x nty tiles, tile L = it * ntx + jt); with stream-k (G > 0
// blocks over tot = ntiles * bpn k blocks, each block's start rounded down to bpi blocks) a tile the split cuts
// gets its cut point, so its two partial sums meet as llama.cpp's fixup adds them
template <ggml_type type, int J, bool SEG>
__launch_bounds__(fast::kThreads, 1) __global__
void fast_dense_kernel(const fast::Args a, const int ncols, const int ntx, const int G, const long long tot, const int bpn,
                       const int bpi) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    extern __shared__ __align__(16) unsigned char fsmem[];
    const int L = blockIdx.x, jt = L % ntx, it = L / ntx;
    int seg = 0;
    if constexpr (SEG) {
        const long long lo = (long long) L * bpn, hi = lo + bpn;
        const int b0 = (int) (lo * G / tot);
        for (int b = max(b0, 1); b <= min(b0 + 2, G - 1); ++b) {
            long long B = (long long) b * tot / G;
            B -= (B % bpn) % bpi;
            if (B > lo && B < hi) seg = (int) (B - lo);
        }
    }
    const fast::Tile t{0, it, jt * J, min(J, ncols - jt * J), seg};
    fast::tile<type, J, SEG>(a, t, fsmem);
#else
    NO_DEVICE_CODE;
#endif
}

// a dense product on llama.cpp's own tile grid through mmq_magic.cuh's tile (llama.cpp's code, conversion-free
// epilogue); stream-k (G > 0) mirrored as in fast_dense_kernel
template <ggml_type type, int J, bool fallback, bool SEG>
__launch_bounds__(256, 1) __global__
void magic_dense_kernel(const char* __restrict__ x, const int* __restrict__ y, const int32_t* __restrict__ ids,
                        float* __restrict__ dst, const int nrows_x,
                        const int stride_row_x, const int ncols, const int stride_col_dst, const int bpn, const int ntx,
                        const int G, const long long tot, const int bpi) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    extern __shared__ int msmem[];
    constexpr int I = 128;
    const int L = blockIdx.x, jt = L % ntx, it = L / ntx;
    int seg = 0;
    if constexpr (SEG) {
        const long long lo = (long long) L * bpn, hi = lo + bpn;
        const int b0 = (int) (lo * G / tot);
        for (int b = max(b0, 1); b <= min(b0 + 2, G - 1); ++b) {
            long long B = (long long) b * tot / G;
            B -= (B % bpn) % bpi;
            if (B > lo && B < hi) seg = (int) (B - lo);
        }
    }
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int j = tid; j < J; j += 256) msmem[j] = ids[jt * J + min(j, ncols - 1 - jt * J)];
    __syncthreads();
    magic::process_tile<type, J, fallback, SEG>(x, it * I * stride_row_x, y + jt * J * (int) (sizeof(block_q8_1_mmq) / sizeof(int)),
                                                msmem, dst + it * I, stride_row_x, ncols, stride_col_dst, nrows_x - it * I - 1,
                                                ncols - jt * J - 1, bpn, seg);
#else
    NO_DEVICE_CODE;
#endif
}

using magic_dense_fn = void (*)(const char*, const int*, const int32_t*, float*, int, int, int, int, int, int, int, long long, int);

template <ggml_type type, int J, bool fallback, bool SEG> magic_dense_fn magic_dense_get(int cc) {
    static magic_dense_fn f = [cc] {
        magic_dense_fn k = magic_dense_kernel<type, J, fallback, SEG>;
        const int smem = (int) mmq_get_nbytes_shared(ggml_cuda_mmq_get_config(type, J, fallback, cc), cc);
        ck(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "magic dense smem");
        return k;
    }();
    return f;
}

template <ggml_type type, bool fallback> magic_dense_fn magic_dense_pick(int J, bool seg, int cc) {
#define STRATA_MD(Jv) case Jv: return seg ? magic_dense_get<type, Jv, fallback, true>(cc) : magic_dense_get<type, Jv, fallback, false>(cc)
    if constexpr (fallback) {
        switch (J) {
            STRATA_MD(64);
            STRATA_MD(128);
            default: return nullptr;
        }
    } else {
        switch (J) {
            STRATA_MD(64);
            STRATA_MD(80);
            STRATA_MD(96);
            STRATA_MD(112);
            STRATA_MD(128);
            default: return nullptr;
        }
    }
#undef STRATA_MD
}

using fast_dense_fn = void (*)(const fast::Args, int, int, int, long long, int, int);

template <ggml_type type, int J, bool SEG> fast_dense_fn fast_dense_get() {
    static fast_dense_fn f = [] {
        fast_dense_fn k = fast_dense_kernel<type, J, SEG>;
        ck(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, fast::smem_bytes<type>(J)), "fast dense smem");
        return k;
    }();
    return f;
}

template <ggml_type type> fast_dense_fn fast_dense_pick(int J, bool seg) {
    switch (J) {
        case 64: return seg ? fast_dense_get<type, 64, true>() : fast_dense_get<type, 64, false>();
        case 80: return seg ? fast_dense_get<type, 80, true>() : fast_dense_get<type, 80, false>();
        case 96: return seg ? fast_dense_get<type, 96, true>() : fast_dense_get<type, 96, false>();
        case 112: return seg ? fast_dense_get<type, 112, true>() : fast_dense_get<type, 112, false>();
        case 128: return seg ? fast_dense_get<type, 128, true>() : fast_dense_get<type, 128, false>();
        default: return nullptr;
    }
}

int g_fast = -1;   // STRATA_MMQ_FAST: 2 (default) llama.cpp's tile with the conversion-free epilogue (mmq_magic.cuh),
                   // 1 = mmq_fast.cuh's tile, 0 = llama.cpp's tile code unchanged
int g_u8 = -1;     // STRATA_MMQ_U8 (default 0): Q4_0 fast tiles with unsigned nibbles

int env_int(const char* name, int dflt) {
    const char* e = std::getenv(name);
    return e && *e ? std::atoi(e) : dflt;
}
int g_jset = -1, g_c0 = -1;   // the width set and the fixed tile cost (STRATA_MMQ_JSET / STRATA_MMQ_C0)

// ------------------------------------------------------------------------------------------------ fused quantizers

__device__ __forceinline__ float gelu_tanh_f(float x) {   // kernels.cu's gelu_tanh, letter for letter
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

// quantize.cu's quantize_mmq_q8_1 (D4 / DS4 layouts) with its 4 inputs computed as k::geglu computes h: the product
// is a __fmul_rn so that it can never be contracted into the block sum below (a stored h never is)
template <mmq_q8_1_ds_layout ds_layout>
__global__ void geglu_quantize_kernel(const float* __restrict__ gate, const float* __restrict__ up, const int64_t ld,
                                      void* __restrict__ vy, const int64_t ne00, const int64_t ne0, const int ne1) {
    static_assert(ds_layout != MMQ_Q8_1_DS_LAYOUT_D2S6, "D2S6 is not covered");
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;
    const int64_t r = blockIdx.x;
    block_q8_1_mmq* y = (block_q8_1_mmq*) vy;
    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs = i0 % QK8_1_MMQ;

    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < ne00) {
        const float4 g4 = *(const float4*) (gate + r * ld + i0);
        const float4 u4 = *(const float4*) (up + r * ld + i0);
        xi.x = __fmul_rn(gelu_tanh_f(g4.x), u4.x);
        xi.y = __fmul_rn(gelu_tanh_f(g4.y), u4.y);
        xi.z = __fmul_rn(gelu_tanh_f(g4.z), u4.z);
        xi.w = __fmul_rn(gelu_tanh_f(g4.w), u4.w);
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32 / 8; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = 32 / 8; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;
    const int64_t ib = k_block * ne1 + blockIdx.x;
    char4* yqs4 = (char4*) y[ib].qs;
    yqs4[iqs / 4] = q;
    if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

}  // namespace

bool built() { return true; }

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

bool fits(int t, int64_t w_rows) {
    if (!supported(t)) return false;
    // mul_mat_q_case's choice: the "fallback" configs when the rows are not a multiple of 128; then
    // mul_mat_q_switch_J's loop - a tile size whose config exists for this card and fits its shared memory
    const bool fallback = w_rows % 128 != 0;
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, fallback, cc);
            any = c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo;
        }
        if (!any) {
            static bool said[GGML_TYPE_COUNT] = {};
            if (t >= 0 && t < GGML_TYPE_COUNT && !said[t]) {
                said[t] = true;
                std::fprintf(stderr, "strata: prompt kernels: llama.cpp's MMQ has no tile for %s (%lld rows) on GPU %d "
                                     "(cc %d, %zu bytes of shared memory per block): that product takes the non-MMQ path "
                                     "(#420)\n", ggml_type_name((ggml_type) t), (long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

bool legacy() {
    static const bool v = env_int("STRATA_MMQ_LEGACY", 0) != 0;
    return v;
}

void quantize_scatter(const float* x, const int32_t* inv, void* xq, int t, int64_t cols, int64_t ld, int n_tok, int k,
                      int64_t rows, void* stream) {
    if (n_tok <= 0 || rows <= 0) return;
    quantize_scatter_mmq_q8_1_cuda(x, inv, xq, (ggml_type) t, cols, ld, pad512(cols), n_tok, rows, k, (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize_scatter");
}

bool geglu_quantize(const float* gate, const float* up, int64_t ld, void* xq, int t, int64_t cols, int64_t rows,
                    void* stream) {
    const mmq_q8_1_ds_layout layout = mmq_get_q8_1_ds_layout((ggml_type) t);
    if (layout != MMQ_Q8_1_DS_LAYOUT_D4 && layout != MMQ_Q8_1_DS_LAYOUT_DS4) return false;
    if (cols % 4 != 0 || ld % 4 != 0 || ((uintptr_t) gate | (uintptr_t) up) % 16 != 0) return false;
    if (rows <= 0) return true;
    const int64_t ne0 = pad512(cols);
    const dim3 grid((unsigned) rows, (unsigned) ((ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ)), 1);
    const cudaStream_t s = (cudaStream_t) stream;
    switch (mmq_get_q8_1_ds_layout((ggml_type) t)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            geglu_quantize_kernel<MMQ_Q8_1_DS_LAYOUT_D4><<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, s>>>(gate, up, ld, xq, cols, ne0, (int) rows);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            geglu_quantize_kernel<MMQ_Q8_1_DS_LAYOUT_DS4><<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, s>>>(gate, up, ld, xq, cols, ne0, (int) rows);
            break;
        default:
            return false;
    }
    ck(cudaGetLastError(), "geglu_quantize");
    return true;
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
    ck(cudaMalloc(&items_, (size_t) kTileCap * sizeof(TileItem)), "tile list");
    ck(cudaMalloc(&ctl_, 4 * sizeof(int32_t)), "tile list");
    ck(cudaMemset(ctl_, 0, 4 * sizeof(int32_t)), "tile list");
}
Context::~Context() {
    delete (ggml_backend_cuda_context*) ctx_;
    if (items_) cudaFree(items_);
    if (ctl_) cudaFree(ctl_);
    if (h_bounds_) cudaFreeHost(h_bounds_);
}

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk;
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, nullptr,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
        case GGML_TYPE_Q4_0: mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, a, s); break;
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
}

void tile_tuning(int jset, int c0) {
    g_jset = jset == 1 ? 1 : 0;
    g_c0 = std::max(c0, 0);
}

void knob(const char* name, int value) {
    const std::string n = name;
    if (g_jset < 0) {
        g_jset = env_int("STRATA_MMQ_JSET", 0) == 1 ? 1 : 0;
        g_c0 = std::max(env_int("STRATA_MMQ_C0", 64), 0);
    }
    if (g_fast < 0) g_fast = env_int("STRATA_MMQ_FAST", 2);
    if (g_u8 < 0) g_u8 = env_int("STRATA_MMQ_U8", 0) != 0;
    if (n == "jset") g_jset = value == 1 ? 1 : 0;
    else if (n == "c0") g_c0 = std::max(value, 0);
    else if (n == "fast") g_fast = value;
    else if (n == "u8") g_u8 = value != 0;
    else std::fprintf(stderr, "mmq: unknown knob %s\n", name);
}

void Context::run_tiles(const Product& p, void* stream) {
    if (p.n <= 0 || p.total_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const cudaStream_t s = (cudaStream_t) stream;
    if (g_jset < 0) {
        g_jset = env_int("STRATA_MMQ_JSET", 0) == 1 ? 1 : 0;
        g_c0 = std::max(env_int("STRATA_MMQ_C0", 64), 0);
    }
    const int64_t qk = ggml_blck_size(t);
    const bool covered = (t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0) && p.w_rows % 128 == 0 && p.w_cols % qk == 0 &&
                         p.n <= kMaxExperts && p.total_rows / kJmin + p.n <= kTileCap &&
                         p.expert_bytes % ggml_type_size(t) == 0 &&
                         (int64_t) (p.expert_bytes / ggml_type_size(t)) * p.n < INT_MAX &&
                         (p.total_rows + 128) * (p.w_cols / 128 + 1) * 36 < INT_MAX && p.ld_dst * (p.total_rows + 1) < INT_MAX;
    if (!covered) {   // llama.cpp's grid needs max_rows: the bounds come back to the host
        if (h_bounds_n_ < p.n + 1) {
            if (h_bounds_) cudaFreeHost(h_bounds_);
            ck(cudaMallocHost(&h_bounds_, (size_t) (p.n + 1) * sizeof(int32_t)), "bounds");
            h_bounds_n_ = p.n + 1;
        }
        ck(cudaMemcpyAsync(h_bounds_, p.bounds, (size_t) (p.n + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, s), "bounds");
        ck(cudaStreamSynchronize(s), "bounds");
        Product q = p;
        q.max_rows = 1;
        for (int e = 0; e < p.n; ++e) q.max_rows = std::max<int64_t>(q.max_rows, h_bounds_[e + 1] - h_bounds_[e]);
        run(q, stream);
        return;
    }
    if (g_fast < 0) g_fast = env_int("STRATA_MMQ_FAST", 2);
    const bool use_fast = g_fast == 1 && g_jset == 0 && (p.w_cols / qk) * ggml_type_size(t) % 4 == 0 &&
                          ((uintptr_t) p.w | (uintptr_t) p.xq) % 16 == 0 && p.expert_bytes % 4 == 0;
    build_tiles_kernel<<<1, 128, 0, s>>>(p.bounds, p.n, g_jset, g_c0, (TileItem*) items_, kTileCap, ctl_);
    ck(cudaGetLastError(), "build_tiles");
    if (use_fast) {
        if (g_u8 < 0) g_u8 = env_int("STRATA_MMQ_U8", 0) != 0;
        const int u8 = g_u8;
        auto kern = t == GGML_TYPE_Q4_0 ? (u8 ? fast_moe_kernel<GGML_TYPE_Q4_0, true> : fast_moe_kernel<GGML_TYPE_Q4_0, false>)
                                        : fast_moe_kernel<GGML_TYPE_Q8_0, false>;
        const int smem = t == GGML_TYPE_Q4_0 ? (u8 ? fast::smem_bytes<GGML_TYPE_Q4_0, true>(128) : fast::smem_bytes<GGML_TYPE_Q4_0>(128))
                                             : fast::smem_bytes<GGML_TYPE_Q8_0>(128);
        static int grid = 0;
        static bool attr[3] = {false, false, false};
        const int ai = t == GGML_TYPE_Q4_0 ? (u8 ? 1 : 0) : 2;
        if (!attr[ai]) {
            ck(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "fast smem");
            attr[ai] = true;
        }
        if (!grid) {
            int per_sm = 0;
            ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, fast::kThreads, smem), "fast occupancy");
            grid = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm * std::max(per_sm, 1);
        }
        fast::Args fa;
        fa.x = (const char*) p.w;
        fa.y = (const char*) p.xq;
        fa.ids = p.ids;
        fa.dst = p.dst;
        fa.nrows = (int) p.w_rows;
        fa.stride_row = (int) (p.w_cols / qk);
        fa.stride_expert = (long long) (p.expert_bytes / ggml_type_size(t));
        fa.ncols_y = (int) p.total_rows;
        fa.ld_dst = (int) p.ld_dst;
        fa.nkt = (int) ((p.w_cols / qk + 7) / 8 * 2);
        const int nty = (int) (p.w_rows / 128);
        kern<<<grid, fast::kThreads, smem, s>>>(fa, (const TileItem*) items_, ctl_, nty);
        ck(cudaGetLastError(), "fast moe");
        return;
    }
    TilesLaunch L;
    const bool mg = g_fast == 2;
    if (t == GGML_TYPE_Q4_0)
        L = g_jset == 0 ? (mg ? tiles_launch<GGML_TYPE_Q4_0, 0, 1, true>() : tiles_launch<GGML_TYPE_Q4_0, 0, 1>()) : tiles_launch<GGML_TYPE_Q4_0, 1, 2>();
    else
        L = g_jset == 0 ? (mg ? tiles_launch<GGML_TYPE_Q8_0, 0, 1, true>() : tiles_launch<GGML_TYPE_Q8_0, 0, 1>()) : tiles_launch<GGML_TYPE_Q8_0, 1, 2>();
    const char* x = (const char*) p.w;
    const int* y = (const int*) p.xq;
    const int32_t* ids = p.ids;
    const TileItem* items = (const TileItem*) items_;
    int32_t* ctl = ctl_;
    float* dst = p.dst;
    int nrows_x = (int) p.w_rows, stride_row_x = (int) (p.w_cols / qk), ncols_y = (int) p.total_rows,
        stride_col_dst = (int) p.ld_dst, stride_expert_x = (int) (p.expert_bytes / ggml_type_size(t)),
        kb_stop = (int) (p.w_cols / qk), nty = (int) (p.w_rows / 128);
    void* args[] = {&x, &y, &ids, &items, &ctl, &dst, &nrows_x, &stride_row_x, &ncols_y, &stride_col_dst,
                    &stride_expert_x, &kb_stop, &nty};
    ck(cudaLaunchKernel((const void*) L.fn, dim3(L.grid), dim3(32, 8, 1), args, (size_t) L.smem, s), "mmq tiles");
}

void Context::run_dense(const Product& p, void* stream) {
    if (p.n != 1 || p.total_rows <= 0) {
        run(p, stream);
        return;
    }
    const ggml_type t = (ggml_type) p.type;
    if (g_fast < 0) g_fast = env_int("STRATA_MMQ_FAST", 2);
    const int64_t qk = ggml_blck_size(t);
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc, nsm = ggml_cuda_info().devices[id].nsm;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    bool ok = g_fast && t == GGML_TYPE_Q8_0 && p.w_cols % qk == 0 && (p.w_cols / qk) * ggml_type_size(t) % 4 == 0 &&
              ((uintptr_t) p.w | (uintptr_t) p.xq) % 16 == 0 && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE &&
              p.total_rows * (p.w_cols / 128 + 1) * 144 < INT_MAX && p.ld_dst * (p.total_rows + 1) < INT_MAX;
    // llama.cpp's launch for this product (mul_mat_q_case -> mul_mat_q_switch_J -> launch_mul_mat_q)
    const bool fallback = p.w_rows % 128 != 0;
    int J = 0, best = INT_MAX;
    for (int j = 8; j <= 128 && best > 1; j += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, j, fallback, cc);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
        if (c.I != 128 || c.nthreads != 256 || c.K_vram != MMQ_ITER_K || !c.stream_k) ok = false;
        const int nt = (int) ((p.max_rows + j - 1) / j);
        if (nt < best) {
            J = j;
            best = nt;
        }
    }
    const int ntx = (int) ((p.total_rows + J - 1) / std::max(J, 1)), nty = (int) ((p.w_rows + 127) / 128);
    const int ntiles = ntx * nty, waves = (ntiles + nsm - 1) / nsm;
    const bool streamk = 100 * ntiles / (nsm * waves) < 90;
    const int bpn = (int) (p.w_cols / qk), bpi = MMQ_ITER_K / (int) qk;
    const long long tot = (long long) ntiles * bpn;
    if (ok && streamk) {   // the fast tile mirrors a split into at most two segments per tile
        long long prev = -1;
        for (int b = 1; b < nsm && ok; ++b) {
            long long B = (long long) b * tot / nsm;
            B -= (B % bpn) % bpi;
            if (B % bpn != 0 && prev >= 0 && prev / bpn == B / bpn) ok = false;
            if (B % bpn != 0) prev = B;
        }
    }
    if (ok && g_fast == 2 && p.max_rows == p.total_rows && p.ids) {
        const magic_dense_fn mfn = fallback ? magic_dense_pick<GGML_TYPE_Q8_0, true>(J, streamk, cc)
                                            : magic_dense_pick<GGML_TYPE_Q8_0, false>(J, streamk, cc);
        if (mfn) {
            const char* x = (const char*) p.w;
            const int* y = (const int*) p.xq;
            const int32_t* ids = p.ids;
            float* dst = p.dst;
            int nrows_x = (int) p.w_rows, stride_row_x = bpn, ncols = (int) p.total_rows, stride_col_dst = (int) p.ld_dst,
                bpn_ = bpn, ntx_ = ntx, G = streamk ? nsm : 0, bpi_ = bpi;
            long long tot_ = tot;
            void* args[] = {&x, &y, &ids, &dst, &nrows_x, &stride_row_x, &ncols, &stride_col_dst, &bpn_, &ntx_, &G, &tot_, &bpi_};
            const int smem = (int) mmq_get_nbytes_shared(ggml_cuda_mmq_get_config(t, J, fallback, cc), cc);
            ck(cudaLaunchKernel((const void*) mfn, dim3(ntiles), dim3(32, 8, 1), args, (size_t) smem, (cudaStream_t) stream),
               "magic dense");
            return;
        }
    }
    const fast_dense_fn fn = ok && g_fast == 1 ? fast_dense_pick<GGML_TYPE_Q8_0>(J, streamk) : nullptr;
    if (!fn || p.max_rows != p.total_rows) {
        run(p, stream);
        return;
    }
    fast::Args fa;
    fa.x = (const char*) p.w;
    fa.y = (const char*) p.xq;
    fa.ids = p.ids;
    fa.dst = p.dst;
    fa.nrows = (int) p.w_rows;
    fa.stride_row = bpn;
    fa.stride_expert = 0;
    fa.ncols_y = (int) p.total_rows;
    fa.ld_dst = (int) p.ld_dst;
    fa.nkt = (bpn + 7) / 8 * 2;
    int ncols = (int) p.total_rows, G = streamk ? nsm : 0;
    long long tot_ = tot;
    int bpn_ = bpn, bpi_ = bpi, ntx_ = ntx;
    void* args[] = {&fa, &ncols, &ntx_, &G, &tot_, &bpn_, &bpi_};
    ck(cudaLaunchKernel((const void*) fn, dim3(ntiles), dim3(fast::kThreads), args, (size_t) fast::smem_bytes<GGML_TYPE_Q8_0>(J),
                        (cudaStream_t) stream), "fast dense");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

}  // namespace strata::gemma::mmq
