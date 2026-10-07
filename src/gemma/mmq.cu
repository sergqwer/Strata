// src/gemma/mmq.cu - see include/strata/gemma/mmq.hpp (adapted from Strata's src/prefill/moe_mmq.cu).  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/gemma/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "strata/gemma/mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

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
// columns at 692 tokens).  Here a persistent grid works off the non-empty (expert, column block) tiles, each as wide
// as its own expert's rows need, widest first, taking (tile, row block) pairs off an atomic counter; every block
// derives the tile order from the bounds itself (no list in memory, no host sync, no extra launch), and the last
// block to finish resets the counter for the next product.  Each output value is still computed by llama.cpp's
// mul_mat_q_process_tile over the whole k range in the same order (the tiling-mode path of mul_mat_q, which every
// MoE product takes: its tile count is a multiple of the SM count, so it never splits a tile with stream-k), so it
// is bit-identical: the value of a column depends on that column of the activations and the expert's rows only, not
// on the tile's width, its other columns or where the tile runs.

constexpr int kMaxExperts = 256;

// the width sets, widest first: 0 = 6 widths, 1 = every width llama.cpp has a Q4_0/Q8_0 tile for from 16 up (10)
__host__ __device__ constexpr int jset_n(int s) { return s == 0 ? 6 : 10; }
__host__ __device__ constexpr int jset_J(int s, int i) {
    return s == 0 ? (i == 0 ? 128 : i == 1 ? 96 : i == 2 ? 64 : i == 3 ? 48 : i == 4 ? 32 : 16)
                  : (i == 0 ? 128 : i == 1 ? 112 : i == 2 ? 96 : i == 3 ? 80 : i == 4 ? 64 : i == 5 ? 48 : i == 6 ? 40
                     : i == 7 ? 32 : i == 8 ? 24 : 16);
}

// one tile through llama.cpp's own tile code (the non-stream-k path of mul_mat_q: kb0 from 0 to the end)
template <ggml_type type, int J>
__device__ __forceinline__ void tile_one(int* smem, const char* __restrict__ x, const int* __restrict__ y,
                                         const int32_t* __restrict__ ids, float* __restrict__ dst, const int expert,
                                         const int col0, const int ncols, const int it, const int nrows_x,
                                         const int stride_row_x, const int ncols_y, const int stride_col_dst,
                                         const int stride_expert_x, const int kb_stop) {
    constexpr bool fallback = false;
    constexpr int nthreads = ggml_cuda_mmq_get_nthreads(type, J, fallback);
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int j = tid; j < J; j += nthreads) smem[j] = ids[col0 + min(j, ncols - 1)];
    __syncthreads();
    mul_mat_q_process_tile<type, J, fallback, false>(
        x, expert * stride_expert_x + it * I * stride_row_x, y + col0 * (int) (sizeof(block_q8_1_mmq) / sizeof(int)),
        smem, dst + it * I, nullptr, nullptr, stride_row_x, ncols_y, stride_col_dst, nrows_x - it * I - 1, ncols - 1, 0,
        kb_stop);
}

// ctl[0]: the work counter, ctl[1]: blocks done (both back to 0 when the last block leaves)
template <ggml_type type, int S>
__launch_bounds__(256, 1) __global__
void tiles_kernel(const char* __restrict__ x, const int* __restrict__ y, const int32_t* __restrict__ ids,
                  const int32_t* __restrict__ bounds, const int n_expert, const int c0, int32_t* __restrict__ ctl,
                  float* __restrict__ dst, const int nrows_x, const int stride_row_x, const int ncols_y,
                  const int stride_col_dst, const int stride_expert_x, const int kb_stop, const int nty) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    extern __shared__ int smem[];
    __shared__ int s_nt[kMaxExperts], s_jc[kMaxExperts], s_rank[kMaxExperts], s_e[kMaxExperts + 1],
        s_off[kMaxExperts + 1];
    __shared__ int s_g, s_n;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // each expert's tile width: the J of the set with the least ceil(rows / J) * (c0 + J) - c0 is a tile's fixed
    // cost (its weight rows are loaded and converted whatever its width) in column units; ties keep the wider tile
    for (int e = tid; e < n_expert; e += 256) {
        const int r = bounds[e + 1] - bounds[e];
        int best_jc = 0, best_nt = 0, best_cost = INT_MAX;
        if (r > 0) {
            for (int c = 0; c < jset_n(S); ++c) {
                const int J = jset_J(S, c), nt = (r + J - 1) / J, cost = nt * (c0 + J);
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
    // tile order: wider widths first, then lower expert ids; s_e[k] = the k-th expert with tiles, s_off its first tile
    for (int e = tid; e < n_expert; e += 256) {
        int rank = 0;
        for (int f = 0; f < n_expert; ++f)
            rank += s_nt[f] > 0 && (s_jc[f] < s_jc[e] || (s_jc[f] == s_jc[e] && f < e));
        s_rank[e] = s_nt[e] > 0 ? rank : -1;
    }
    __syncthreads();
    for (int e = tid; e < n_expert; e += 256)
        if (s_rank[e] >= 0) s_e[s_rank[e]] = e;
    __syncthreads();
    if (tid == 0) {
        int n = 0, off = 0;
        for (int e = 0; e < n_expert; ++e) n += s_nt[e] > 0;
        for (int i = 0; i < n; ++i) {
            s_off[i] = off;
            off += s_nt[s_e[i]];
        }
        s_off[n] = off;
        s_n = n;
    }
    __syncthreads();
    const int n = s_n, total = s_off[n] * nty;
    for (;;) {
        if (tid == 0) s_g = atomicAdd(&ctl[0], 1);
        __syncthreads();
        const int g = s_g;
        if (g >= total) break;
        const int ti = g / nty, it = g - ti * nty;
        int lo = 0, hi = n - 1;   // the expert whose tiles hold ti: s_off[lo] <= ti < s_off[lo + 1]
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (s_off[mid] <= ti) lo = mid;
            else hi = mid - 1;
        }
        const int e = s_e[lo], jc = s_jc[e], J = jset_J(S, jc), t = ti - s_off[lo];
        const int col0 = bounds[e] + t * J, ncols = min(J, bounds[e + 1] - col0);
#define STRATA_TILE(Jv) tile_one<type, Jv>(smem, x, y, ids, dst, e, col0, ncols, it, nrows_x, stride_row_x, ncols_y, stride_col_dst, stride_expert_x, kb_stop)
        if constexpr (S == 0) {
            switch (jc) {
                case 0: STRATA_TILE(128); break;
                case 1: STRATA_TILE(96); break;
                case 2: STRATA_TILE(64); break;
                case 3: STRATA_TILE(48); break;
                case 4: STRATA_TILE(32); break;
                default: STRATA_TILE(16); break;
            }
        } else {
            switch (jc) {
                case 0: STRATA_TILE(128); break;
                case 1: STRATA_TILE(112); break;
                case 2: STRATA_TILE(96); break;
                case 3: STRATA_TILE(80); break;
                case 4: STRATA_TILE(64); break;
                case 5: STRATA_TILE(48); break;
                case 6: STRATA_TILE(40); break;
                case 7: STRATA_TILE(32); break;
                case 8: STRATA_TILE(24); break;
                default: STRATA_TILE(16); break;
            }
        }
#undef STRATA_TILE
        __syncthreads();   // the next tile reuses the shared memory and s_g
    }
    // the last block out resets the counters for the next product (stream order makes it visible there)
    if (tid == 0 && atomicAdd(&ctl[1], 1) == (int) gridDim.x - 1) {
        atomicExch(&ctl[0], 0);
        atomicExch(&ctl[1], 0);
    }
#else
    NO_DEVICE_CODE;
#endif
}

struct TilesLaunch {
    const void* fn = nullptr;
    int smem = 0, grid = 0;
    bool ok = false;   // every width of the set has llama.cpp's 256-thread, 128-row tile on this GPU
};

template <ggml_type type, int S>
TilesLaunch tiles_launch() {
    static TilesLaunch L = [] {
        TilesLaunch l;
        l.fn = (const void*) tiles_kernel<type, S>;
        const int id = ggml_cuda_get_device();
        const int cc = ggml_cuda_info().devices[id].cc, nsm = ggml_cuda_info().devices[id].nsm;
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        l.ok = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
        for (int i = 0; i < jset_n(S) && l.ok; ++i) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, jset_J(S, i), false, cc);
            l.ok = c.type == type && c.nthreads == 256 && c.I == 128 && mmq_get_nbytes_shared(c, cc) <= smpbo;
            if (l.ok) l.smem = std::max(l.smem, (int) mmq_get_nbytes_shared(c, cc));
        }
        if (!l.ok) return l;
        ck(cudaFuncSetAttribute(tiles_kernel<type, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, l.smem), "tiles smem");
        int per_sm = 0;
        ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, tiles_kernel<type, S>, 256, l.smem), "tiles occupancy");
        l.grid = nsm * std::max(per_sm, 1);
        return l;
    }();
    return L;
}

int env_int(const char* name, int dflt) {
    const char* e = std::getenv(name);
    return e && *e ? std::atoi(e) : dflt;
}
int g_jset = -1, g_c0 = -1;   // the width set and the fixed tile cost (STRATA_MMQ_JSET / STRATA_MMQ_C0)
void knobs_init() {
    if (g_jset < 0) {
        g_jset = env_int("STRATA_MMQ_JSET", 1) == 0 ? 0 : 1;
        g_c0 = std::min(std::max(env_int("STRATA_MMQ_C0", 128), 0), 4096);
    }
}

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
    ck(cudaMalloc(&ctl_, 4 * sizeof(int32_t)), "tile counters");
    ck(cudaMemset(ctl_, 0, 4 * sizeof(int32_t)), "tile counters");
}
Context::~Context() {
    delete (ggml_backend_cuda_context*) ctx_;
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

void knob(const char* name, int value) {
    knobs_init();
    const std::string n = name;
    if (n == "jset") g_jset = value == 0 ? 0 : 1;
    else if (n == "c0") g_c0 = std::min(std::max(value, 0), 4096);
    else std::fprintf(stderr, "mmq: unknown knob %s\n", name);
}

void Context::run_tiles(const Product& p, void* stream) {
    if (p.n <= 0 || p.total_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const cudaStream_t s = (cudaStream_t) stream;
    knobs_init();
    const int64_t qk = ggml_blck_size(t);
    TilesLaunch L;
    if (t == GGML_TYPE_Q4_0) L = g_jset == 0 ? tiles_launch<GGML_TYPE_Q4_0, 0>() : tiles_launch<GGML_TYPE_Q4_0, 1>();
    else if (t == GGML_TYPE_Q8_0) L = g_jset == 0 ? tiles_launch<GGML_TYPE_Q8_0, 0>() : tiles_launch<GGML_TYPE_Q8_0, 1>();
    const bool covered = L.ok && p.w_rows % 128 == 0 && p.w_cols % qk == 0 &&
                         p.n <= kMaxExperts && p.expert_bytes % ggml_type_size(t) == 0 &&
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
    const char* x = (const char*) p.w;
    const int* y = (const int*) p.xq;
    const int32_t* ids = p.ids;
    const int32_t* bounds = p.bounds;
    int n_expert = p.n, c0 = g_c0;
    int32_t* ctl = ctl_;
    float* dst = p.dst;
    int nrows_x = (int) p.w_rows, stride_row_x = (int) (p.w_cols / qk), ncols_y = (int) p.total_rows,
        stride_col_dst = (int) p.ld_dst, stride_expert_x = (int) (p.expert_bytes / ggml_type_size(t)),
        kb_stop = (int) (p.w_cols / qk), nty = (int) (p.w_rows / 128);
    void* args[] = {&x, &y, &ids, &bounds, &n_expert, &c0, &ctl, &dst, &nrows_x, &stride_row_x, &ncols_y, &stride_col_dst,
                    &stride_expert_x, &kb_stop, &nty};
    ck(cudaLaunchKernel(L.fn, dim3(L.grid), dim3(32, 8, 1), args, (size_t) L.smem, s), "mmq tiles");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

}  // namespace strata::gemma::mmq
