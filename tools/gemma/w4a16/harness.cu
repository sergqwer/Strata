// tools/gemma/w4a16/harness.cu - the w4a16 expert GEMMs (src/gemma/moe_w4a16.cuh) against the engine's current path
// (q8_1 activations + llama.cpp's MMQ, src/gemma/mmq.cu) on one real layer: numerics against an fp64 reference, and
// speed in interleaved ~100 ms loops. See README.md for the build line and the data files.
//
//   harness num   <dir> <layer>                      real g + routing (mkref.py) vs fp64: hidden and ey errors
//   harness speed <dir> <layer> <route file> [ms]    synthetic routing (gen_routing.py): per-phase times, old vs new
// the ablation variants (ABL) skip parts of loops on purpose
#pragma nv_diag_suppress 128
#include "../../../src/gemma/moe_w4a16.cu"
#include "v1.cuh"
#include "strata/gemma/mmq.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace strata::gemma;
namespace dv = strata::gemma::w4a16::dev;

#define CK(x)                                                                                                    \
    do {                                                                                                         \
        cudaError_t e_ = (x);                                                                                    \
        if (e_ != cudaSuccess) {                                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));              \
            std::exit(1);                                                                                        \
        }                                                                                                        \
    } while (0)
#define CU(x)                                                                                                    \
    do {                                                                                                         \
        CUresult r_ = (x);                                                                                       \
        if (r_ != CUDA_SUCCESS) {                                                                                \
            const char* s_ = nullptr;                                                                            \
            cuGetErrorString(r_, &s_);                                                                           \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s_ ? s_ : "?");                       \
            std::exit(1);                                                                                        \
        }                                                                                                        \
    } while (0)

namespace {

constexpr int kQ4_0 = 2;
constexpr int D = 2816, FF = 704, E = 96, KU = 8;
constexpr size_t GU_ROW = D / 32 * 18, DN_ROW = FF / 32 * 18;
constexpr size_t GU_EXP = (size_t) 2 * FF * GU_ROW, DN_EXP = (size_t) D * DN_ROW;

// the candidates: <mode, KB, CPW, WM, WN, MT, NT, stages, scale on the fp32 partial sums, min blocks>
using GU_S = dv::v1::Cfg<0, 2, 4, 4, 2, 2, 4, 3, true, 2>;     // warps 4 x 2, each gate+up 16 units x 32 rows
using GU_F = dv::v1::Cfg<0, 2, 4, 4, 2, 2, 4, 3, false, 2>;    // the same, scale folded into fp16 weights
using GU_W8 = dv::v1::Cfg<0, 2, 4, 8, 1, 1, 8, 3, false, 2>;   // 8 warps x 16 weight rows x all 64 rows (gate | up warps)
using GU_W8K4 = dv::v1::Cfg<0, 4, 8, 8, 1, 1, 8, 3, false, 1>; // + 128-value k-tiles, 8-byte copies (one block an SM)
using GU_W8B32 = dv::v1::Cfg<0, 2, 4, 8, 1, 1, 4, 3, false, 2>; // 32 rows a block
using GU_FN8 = dv::v1::Cfg<0, 2, 4, 4, 1, 2, 8, 3, false, 2>;  // 4 warps, each gate+up 16 units x 64 rows
using GU_BIG = dv::v1::Cfg<0, 4, 8, 11, 1, 2, 8, 2, false, 1>;  // 11 warps: 176 units (gate + up) x 64 rows, 4 weight tiles
using GU_BIG2 = dv::v1::Cfg<0, 2, 4, 11, 1, 2, 8, 3, false, 1>; // the same with 64-value k-tiles, 3 stages
using DN_BIG = dv::v1::Cfg<1, 2, 4, 11, 1, 2, 8, 3, false, 1>;  // 352 output rows x 64 rows, 8 weight tiles
using GU_R = dv::v1::Cfg<0, 2, 16, 4, 2, 2, 4, 2, false, 2, 0, true>;   // GU_F with the weight sector ring
using GU_RW8 = dv::v1::Cfg<0, 2, 16, 8, 1, 1, 8, 2, false, 2, 0, true>;  // GU_W8 with the ring
using GU_RN8 = dv::v1::Cfg<0, 2, 16, 4, 1, 2, 8, 2, false, 2, 0, true>;  // GU_FN8 with the ring
template <int A> using GU_RA = dv::v1::Cfg<0, 2, 16, 4, 2, 2, 4, 2, false, 2, A, true>;
using GU2 = dv::Cfg<0, 11, 8, 2, false>;    // v2: 11 warps x 32 rows, B from global, line ring
using DN2 = dv::Cfg<1, 11, 8, 2, false>;
using GU2P2 = dv::Cfg<0, 11, 8, 2, false, 0, 2>;   // + B prefetched into the L1 2 / 4 k32 blocks ahead
using GU2P4 = dv::Cfg<0, 11, 8, 2, false, 0, 4>;
using DN2P2 = dv::Cfg<1, 11, 8, 2, false, 0, 2>;
using DN2P4 = dv::Cfg<1, 11, 8, 2, false, 0, 4>;
using GU2S4 = dv::Cfg<0, 11, 8, 2, false, 0, 2, 4>;   // + a block barrier every 4 / 8 k-tiles
using GU2S8 = dv::Cfg<0, 11, 8, 2, false, 0, 2, 8>;
using DN2S4 = dv::Cfg<1, 11, 8, 2, false, 0, 2, 4>;
using GU2RF = dv::Cfg<0, 11, 8, 2, false, 0, 2, 0, 1>;   // + each lane refills its own row
using GU2RFS = dv::Cfg<0, 11, 8, 2, false, 0, 2, 8, 1>;
using GU2W4 = dv::Cfg<0, 4, 8, 2, false, 0, 2>;   // 4 warps (128 weight rows), 2+ blocks an SM
using DN2W4 = dv::Cfg<1, 4, 8, 2, false, 0, 2>;
using DN2W8 = dv::Cfg<1, 8, 8, 2, false, 0, 2>;   // 8 warps (256 rows: 11 weight tiles)
template <int A> using GU2RA = dv::Cfg<0, 11, 8, 2, false, A, 2, 0, 1>;
template <int A> using GU2A = dv::Cfg<0, 11, 8, 2, false, A>;
template <int A> using GU_FA = dv::v1::Cfg<0, 2, 4, 4, 2, 2, 4, 3, false, 2, A>;   // ablations of GU_F
template <int A> using GU_BA = dv::v1::Cfg<0, 2, 4, 11, 1, 2, 8, 3, false, 1, A>;  // ablations of GU_BIG2
using DN_S = dv::v1::Cfg<1, 2, 4, 4, 2, 2, 4, 3, true, 2>;
using DN_F = dv::v1::Cfg<1, 2, 4, 4, 2, 2, 4, 3, false, 2>;
using DN_W8 = dv::v1::Cfg<1, 2, 4, 8, 1, 1, 8, 3, false, 2>;
using DN_W8B32 = dv::v1::Cfg<1, 2, 4, 8, 1, 1, 4, 3, false, 2>;
using DN_W4 = dv::v1::Cfg<1, 2, 4, 4, 1, 2, 8, 3, false, 2>;

std::vector<uint8_t> read_file(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", p.c_str());
        std::exit(1);
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (std::fread(v.data(), 1, n, f) != (size_t) n) std::exit(1);
    std::fclose(f);
    return v;
}
template <class T> std::vector<T> read_mat(const std::string& p, int64_t& rows, int64_t& cols) {
    auto b = read_file(p);
    std::memcpy(&rows, b.data(), 8);
    std::memcpy(&cols, b.data() + 8, 8);
    std::vector<T> v((size_t) rows * cols);
    std::memcpy(v.data(), b.data() + 16, v.size() * sizeof(T));
    return v;
}

size_t g_dev_bytes = 0;
template <class T> T* dalloc(size_t n) {
    void* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T) + 256));
    CK(cudaMemset(p, 0, n * sizeof(T) + 256));
    CK(cudaDeviceSynchronize());
    g_dev_bytes += n * sizeof(T);
    return (T*) p;
}
template <class T> void dfree(T*& p, size_t n) {
    if (p) CK(cudaFree(p));
    g_dev_bytes -= n * sizeof(T);
    p = nullptr;
}

// the weights of every expert, or (speed) a few experts' worth of physical memory mapped again and again over the
// virtual range: the DRAM traffic stays real (experts that share memory run far apart, the L2 is 4 MB) at a third of
// the VRAM
struct Weights {
    CUdeviceptr base = 0;
    size_t virt = 0, phys = 0;
    CUmemGenericAllocationHandle h = 0;
    void* plain = nullptr;
    const uint8_t* ptr() const { return plain ? (const uint8_t*) plain : (const uint8_t*) base; }
};
Weights load_weights(const std::vector<uint8_t>& host, size_t phys_limit) {
    Weights w;
    if (phys_limit == 0 || phys_limit >= host.size()) {
        CK(cudaMalloc(&w.plain, host.size()));
        CK(cudaMemcpy(w.plain, host.data(), host.size(), cudaMemcpyHostToDevice));
        w.phys = host.size();
        g_dev_bytes += w.phys;
        return w;
    }
    int dev = 0;
    CK(cudaGetDevice(&dev));
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    size_t gran = 0;
    CU(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    w.phys = (phys_limit + gran - 1) / gran * gran;
    w.virt = (host.size() + w.phys - 1) / w.phys * w.phys;
    CU(cuMemCreate(&w.h, w.phys, &prop, 0));
    CU(cuMemAddressReserve(&w.base, w.virt, 0, 0, 0));
    for (size_t off = 0; off < w.virt; off += w.phys) CU(cuMemMap(w.base + off, w.phys, 0, w.h, 0));
    CUmemAccessDesc acc = {};
    acc.location = prop.location;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU(cuMemSetAccess(w.base, w.virt, &acc, 1));
    CK(cudaMemcpy((void*) w.base, host.data(), w.phys, cudaMemcpyHostToDevice));
    g_dev_bytes += w.phys;
    return w;
}
void free_weights(Weights& w) {
    if (w.plain) {
        cudaFree(w.plain);
    } else if (w.base) {
        for (size_t off = 0; off < w.virt; off += w.phys) cuMemUnmap(w.base + off, w.phys);
        cuMemAddressFree(w.base, w.virt);
        cuMemRelease(w.h);
    }
    g_dev_bytes -= w.phys;
    w = Weights();
}

// k::moe_sort on the host: rows grouped by expert (pairs in order inside an expert)
struct Sorted {
    std::vector<int32_t> bounds, src, inv;
    int max_rows = 0;
};
Sorted sort_rows(const std::vector<int32_t>& ids, int T) {
    Sorted s;
    s.bounds.assign(E + 1, 0);
    std::vector<int> cnt(E, 0);
    for (int p = 0; p < T * KU; ++p) cnt[ids[p]]++;
    for (int e = 0; e < E; ++e) {
        s.bounds[e + 1] = s.bounds[e] + cnt[e];
        s.max_rows = std::max(s.max_rows, cnt[e]);
    }
    std::vector<int> cur(s.bounds.begin(), s.bounds.end() - 1);
    s.src.resize(T * KU);
    s.inv.resize(T * KU);
    for (int p = 0; p < T * KU; ++p) {
        const int r = cur[ids[p]]++;
        s.src[r] = p / KU;
        s.inv[p] = r;
    }
    return s;
}

__device__ __forceinline__ float gelu_tanh_ref(float x) {
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}
// = k::geglu (kernels.cu)
__global__ void geglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                             int64_t rows, int n_ff, int64_t ld) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, c = i % n_ff;
    h[i] = gelu_tanh_ref(gate[r * ld + c]) * up[r * ld + c];
}

// ------------------------------------------------------------------------------------------------ the two paths

struct Bufs {
    // shared
    const uint8_t* w_gu = nullptr;
    const uint8_t* w_dn = nullptr;
    float* g = nullptr;          // T x D
    int32_t *bounds = nullptr, *src = nullptr, *inv = nullptr, *iota = nullptr;
    int T = 0;
    int64_t R = 0;
    int max_rows = 0;
    // old
    void* xq = nullptr;          // q8_1 (input of either MMQ)
    float* gu = nullptr;         // R x 2FF
    float* eh = nullptr;         // R x FF
    // new
    __half* xs = nullptr;        // R x D
    __half* hid = nullptr;       // R x FF
    float* ey = nullptr;         // R x D
};

mmq::Context* g_mq = nullptr;
cudaStream_t g_s = nullptr;

void old_quant_in(const Bufs& b) { mmq::quantize(b.g, b.src, b.xq, kQ4_0, D, D, b.R, g_s); }
void old_gu(const Bufs& b) {
    mmq::Product p;
    p.w = b.w_gu;
    p.type = kQ4_0;
    p.w_rows = 2 * FF;
    p.w_cols = D;
    p.expert_bytes = GU_EXP;
    p.n = E;
    p.xq = b.xq;
    p.bounds = b.bounds;
    p.ids = b.iota;
    p.total_rows = b.R;
    p.max_rows = b.max_rows;
    p.dst = b.gu;
    p.ld_dst = 2 * FF;
    g_mq->run(p, g_s);
}
void old_geglu_quant(const Bufs& b) {
    geglu_kernel<<<(unsigned) ((b.R * FF + 255) / 256), 256, 0, g_s>>>(b.gu, b.gu + FF, b.eh, b.R, FF, 2 * FF);
    mmq::quantize(b.eh, nullptr, b.xq, kQ4_0, FF, FF, b.R, g_s);
}
void old_dn(const Bufs& b) {
    mmq::Product p;
    p.w = b.w_dn;
    p.type = kQ4_0;
    p.w_rows = D;
    p.w_cols = FF;
    p.expert_bytes = DN_EXP;
    p.n = E;
    p.xq = b.xq;
    p.bounds = b.bounds;
    p.ids = b.iota;
    p.total_rows = b.R;
    p.max_rows = b.max_rows;
    p.dst = b.ey;
    p.ld_dst = D;
    g_mq->run(p, g_s);
}

void new_scatter(const Bufs& b) {
    const int64_t n = (int64_t) b.T * (D / 16);
    dv::v1::scatter_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, g_s>>>(b.g, D, b.inv, b.T, KU, b.xs);
}
template <class C> void new_gemm(const Bufs& b) {
    static bool attr = false;
    if (!attr) {
        CK(cudaFuncSetAttribute(dv::v1::gemm_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        attr = true;
    }
    dv::Args a;
    a.bounds = b.bounds;
    a.n_expert = E;
    if constexpr (C::MODE == 0) {
        a.w = b.w_gu;
        a.row_bytes = GU_ROW;
        a.expert_bytes = GU_EXP;
        a.K = D;
        a.ff = FF;
        a.x = b.xs;
        a.hidden = b.hid;
        dv::v1::gemm_kernel<C><<<dv::v1::grid_of<C>(b.R, E, FF), C::THREADS, C::SMEM, g_s>>>(a);
    } else {
        a.w = b.w_dn;
        a.row_bytes = DN_ROW;
        a.expert_bytes = DN_EXP;
        a.K = FF;
        a.x = b.hid;
        a.ey = b.ey;
        a.ld_ey = D;
        dv::v1::gemm_kernel<C><<<dv::v1::grid_of<C>(b.R, E, D), C::THREADS, C::SMEM, g_s>>>(a);
    }
}

void new_scatter2(const Bufs& b) {
    const int64_t n = (int64_t) b.T * (D / 32);
    dv::scatter_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, g_s>>>(b.g, D, b.inv, b.T, KU, b.xs);
}
template <class C> void new_gemm2(const Bufs& b) {
    static bool attr = false;
    if (!attr) {
        CK(cudaFuncSetAttribute(dv::gemm_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        attr = true;
    }
    dv::Args a;
    a.bounds = b.bounds;
    a.n_expert = E;
    if constexpr (C::MODE == 0) {
        a.w = b.w_gu;
        a.row_bytes = GU_ROW;
        a.expert_bytes = GU_EXP;
        a.K = D;
        a.ff = FF;
        a.x = b.xs;
        a.hidden = b.hid;
        dv::gemm_kernel<C><<<dv::grid_of<C>(b.R, E, FF), C::THREADS, C::SMEM, g_s>>>(a);
    } else {
        a.w = b.w_dn;
        a.row_bytes = DN_ROW;
        a.expert_bytes = DN_EXP;
        a.K = FF;
        a.x = b.hid;
        a.ey = b.ey;
        a.ld_ey = D;
        dv::gemm_kernel<C><<<dv::grid_of<C>(b.R, E, D), C::THREADS, C::SMEM, g_s>>>(a);
    }
}
template <class C> void print_cfg2(const char* name) {
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, dv::gemm_kernel<C>));
    int nb = 0;
    CK(cudaFuncSetAttribute(dv::gemm_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, dv::gemm_kernel<C>, C::THREADS, C::SMEM));
    std::printf("  %-7s v2 BM %3d BN %3d LW %d %s: %3d regs, %5d B local, smem %6d, %d blocks/SM\n", name, C::BM, C::BN,
                C::LW, C::SACC ? "scale-acc " : "scale-fold", fa.numRegs, (int) fa.localSizeBytes, C::SMEM, nb);
}
template <class C> void print_cfg(const char* name) {
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, dv::v1::gemm_kernel<C>));
    int nb = 0;
    CK(cudaFuncSetAttribute(dv::v1::gemm_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, dv::v1::gemm_kernel<C>, C::THREADS, C::SMEM));
    std::printf("  %-7s BM %3d BN %3d KT %3d st %d %s: %3d regs, %5d B local, smem %6d, %d blocks/SM\n", name, C::BM,
                C::BN, C::KT, C::STAGES, C::SACC ? "scale-acc " : "scale-fold", fa.numRegs, (int) fa.localSizeBytes,
                C::SMEM, nb);
}

// ------------------------------------------------------------------------------------------------ numerics

struct Err {
    double rel_rms = 0, worst_row = 0, max_abs_rel = 0;
};
// got (sorted rows, unpermuted when `perm`), ref (pair order); compares got[inv[p]] with ref[p]
template <class T>
Err compare(const std::vector<T>& got, const std::vector<float>& ref, const std::vector<int32_t>& inv, int cols,
            int perm) {   // 0: plain, 1: perm16 (v1), 2: fpos (v2)
    Err e;
    double se = 0, sr = 0, maxabs = 0;
    const size_t P = inv.size();
    for (size_t p = 0; p < P; ++p) {
        const size_t r = inv[p];
        double re = 0, rr = 0;
        for (int c = 0; c < cols; ++c) {
            const int gc = perm == 1 ? (c & ~15) | dv::v1::perm16(c & 15) : perm == 2 ? (c & ~31) | dv::fpos(c & 31) : c;
            double gv;
            if constexpr (std::is_same_v<T, __half>) gv = (double) __half2float(got[r * cols + gc]);
            else gv = (double) got[r * cols + gc];
            const double rv = ref[p * cols + c], d = gv - rv;
            re += d * d;
            rr += rv * rv;
            maxabs = std::max(maxabs, std::fabs(d));
        }
        se += re;
        sr += rr;
        if (rr > 0) e.worst_row = std::max(e.worst_row, std::sqrt(re / rr));
    }
    e.rel_rms = std::sqrt(se / sr);
    e.max_abs_rel = maxabs / std::sqrt(sr / ((double) P * cols));
    return e;
}
void print_err(const char* what, const Err& e) {
    std::printf("    %-34s rel rms %.3e   worst row %.3e   max |err| / rms %.3e\n", what, e.rel_rms, e.worst_row,
                e.max_abs_rel);
}
template <class T> std::vector<T> to_host(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

int run_num(const std::string& dir, int il, int max_tokens) {
    const std::string L = dir + "/L" + std::to_string(il);
    int64_t T, c;
    auto g = read_mat<float>(L + ".g.f32", T, c);
    auto ids = read_mat<int32_t>(L + ".ids.i32", T, c);
    int64_t rr, cc;
    auto href = read_mat<float>(L + ".href.f32", rr, cc);
    auto yref = read_mat<float>(L + ".yref.f32", rr, cc);
    if (max_tokens > 0 && max_tokens < T) {   // the first max_tokens tokens only (small prompt chunks)
        T = max_tokens;
        g.resize((size_t) T * D);
        ids.resize((size_t) T * KU);
        href.resize((size_t) T * KU * FF);
        yref.resize((size_t) T * KU * D);
    }
    Sorted so = sort_rows(ids, (int) T);
    Bufs b;
    b.T = (int) T;
    b.R = T * KU;
    b.max_rows = so.max_rows;
    std::printf("layer %d: %lld tokens, %lld rows, max %d rows an expert\n", il, (long long) T, (long long) b.R, so.max_rows);
    b.g = dalloc<float>((size_t) T * D);
    CK(cudaMemcpy(b.g, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
    b.bounds = dalloc<int32_t>(E + 1);
    b.src = dalloc<int32_t>(b.R);
    b.inv = dalloc<int32_t>(b.R);
    b.iota = dalloc<int32_t>(b.R + 2);
    CK(cudaMemcpy(b.bounds, so.bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.src, so.src.data(), b.R * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.inv, so.inv.data(), b.R * 4, cudaMemcpyHostToDevice));
    mmq::iota(b.iota, b.R + 2, g_s);

    // ---- phase 1: gate_up (+ GeGLU)
    {
        auto wh = read_file(L + ".gate_up");
        Weights w = load_weights(wh, 0);
        b.w_gu = w.ptr();
        const size_t xq_bytes = mmq::q8_bytes(b.R, D);
        uint8_t* xq8 = dalloc<uint8_t>(xq_bytes);
        b.xq = xq8;
        b.gu = dalloc<float>((size_t) b.R * 2 * FF);   // also the fp16 xs (same bytes)
        b.xs = reinterpret_cast<__half*>(b.gu);
        b.eh = dalloc<float>((size_t) b.R * FF);
        b.hid = dalloc<__half>((size_t) b.R * FF);
        std::printf("  phase gate_up: %.0f MiB on the device\n", g_dev_bytes / 1048576.0);
        old_quant_in(b);
        old_gu(b);
        old_geglu_quant(b);
        CK(cudaStreamSynchronize(g_s));
        auto eh_old = to_host(b.eh, (size_t) b.R * FF);
        print_err("hidden, old (q8_1 + MMQ)", compare(eh_old, href, so.inv, FF, 0));
        new_scatter(b);
        new_gemm<GU_S>(b);
        CK(cudaStreamSynchronize(g_s));
        auto hid_s = to_host(b.hid, (size_t) b.R * FF);
        print_err("hidden, w4a16 fp16, scale on acc", compare(hid_s, href, so.inv, FF, 1));
        new_gemm<GU_F>(b);
        CK(cudaStreamSynchronize(g_s));
        auto hid_f = to_host(b.hid, (size_t) b.R * FF);
        print_err("hidden, w4a16 fp16, scale folded", compare(hid_f, href, so.inv, FF, 1));
        // every layout of the same arithmetic gives the same bits
        for (auto& [name, fn] : std::vector<std::pair<const char*, std::function<void()>>>{
                 {"GU_W8", [&] { new_gemm<GU_W8>(b); }}, {"GU_W8K4", [&] { new_gemm<GU_W8K4>(b); }},
                 {"GU_W8B32", [&] { new_gemm<GU_W8B32>(b); }}, {"GU_FN8", [&] { new_gemm<GU_FN8>(b); }},
                 {"GU_BIG", [&] { new_gemm<GU_BIG>(b); }}, {"GU_BIG2", [&] { new_gemm<GU_BIG2>(b); }},
                 {"GU_R", [&] { new_gemm<GU_R>(b); }}, {"GU_RW8", [&] { new_gemm<GU_RW8>(b); }},
                 {"GU_RN8", [&] { new_gemm<GU_RN8>(b); }}}) {
            CK(cudaMemsetAsync(b.hid, 0, (size_t) b.R * FF * 2, g_s));
            fn();
            CK(cudaStreamSynchronize(g_s));
            auto hv = to_host(b.hid, (size_t) b.R * FF);
            std::printf("    %-8s == GU_F: %s\n", name, std::memcmp(hv.data(), hid_f.data(), hv.size() * 2) == 0 ? "yes" : "NO");
        }
        // v2: the fragment-contiguous layout, its own scatter
        new_scatter2(b);
        CK(cudaMemsetAsync(b.hid, 0, (size_t) b.R * FF * 2, g_s));
        new_gemm2<GU2>(b);
        CK(cudaStreamSynchronize(g_s));
        auto hid_v2 = to_host(b.hid, (size_t) b.R * FF);
        print_err("hidden, v2 (fp16, scale folded)", compare(hid_v2, href, so.inv, FF, 2));
        new_scatter(b);   // back to v1's layout for the rest
        // the production entry points (moe_w4a16.cu) give the same bits as their configuration (v2)
        CK(cudaMemsetAsync(b.hid, 0, (size_t) b.R * FF * 2, g_s));
        w4a16::scatter(b.g, D, b.inv, b.T, KU, b.xs, g_s);
        w4a16::gate_up(b.w_gu, GU_ROW, GU_EXP, D, FF, b.xs, b.bounds, E, b.R, b.hid, g_s);
        CK(cudaStreamSynchronize(g_s));
        auto hid_p = to_host(b.hid, (size_t) b.R * FF);
        std::printf("    production gate_up == v2: %s\n",
                    std::memcmp(hid_p.data(), hid_v2.data(), hid_p.size() * 2) == 0 ? "yes" : "NO");
        new_scatter(b);
        dfree(xq8, xq_bytes);
        b.xq = nullptr;
        dfree(b.gu, (size_t) b.R * 2 * FF);
        b.xs = nullptr;
        free_weights(w);

        // ---- phase 2: down, each path on its own hidden, and both on the exact hidden
        auto wd = read_file(L + ".down");
        Weights w2 = load_weights(wd, 0);
        b.w_dn = w2.ptr();
        const size_t xq2 = mmq::q8_bytes(b.R, FF);
        xq8 = dalloc<uint8_t>(xq2);
        b.xq = xq8;
        b.ey = dalloc<float>((size_t) b.R * D);
        std::printf("  phase down: %.0f MiB on the device\n", g_dev_bytes / 1048576.0);
        // old: eh_old -> q8_1 -> MMQ
        CK(cudaMemcpy(b.eh, eh_old.data(), eh_old.size() * 4, cudaMemcpyHostToDevice));
        mmq::quantize(b.eh, nullptr, b.xq, kQ4_0, FF, FF, b.R, g_s);
        old_dn(b);
        CK(cudaStreamSynchronize(g_s));
        print_err("ey, old (end to end)", compare(to_host(b.ey, (size_t) b.R * D), yref, so.inv, D, 0));
        CK(cudaMemcpy(b.hid, hid_s.data(), hid_s.size() * 2, cudaMemcpyHostToDevice));
        new_gemm<DN_S>(b);
        CK(cudaStreamSynchronize(g_s));
        auto ey_s = to_host(b.ey, (size_t) b.R * D);
        print_err("ey, w4a16 scale on acc (end to end)", compare(ey_s, yref, so.inv, D, 0));

        CK(cudaMemcpy(b.hid, hid_f.data(), hid_f.size() * 2, cudaMemcpyHostToDevice));
        new_gemm<DN_F>(b);
        CK(cudaStreamSynchronize(g_s));
        auto ey_f = to_host(b.ey, (size_t) b.R * D);
        print_err("ey, w4a16 scale folded (end to end)", compare(ey_f, yref, so.inv, D, 0));
        for (auto& [name, fn] : std::vector<std::pair<const char*, std::function<void()>>>{
                 {"DN_W8", [&] { new_gemm<DN_W8>(b); }}, {"DN_W8B32", [&] { new_gemm<DN_W8B32>(b); }},
                 {"DN_W4", [&] { new_gemm<DN_W4>(b); }}, {"DN_BIG", [&] { new_gemm<DN_BIG>(b); }}}) {
            CK(cudaMemsetAsync(b.ey, 0, (size_t) b.R * D * 4, g_s));
            fn();
            CK(cudaStreamSynchronize(g_s));
            auto ev = to_host(b.ey, (size_t) b.R * D);
            size_t nd = 0, first = 0;
            double md = 0;
            for (size_t i = 0; i < ev.size(); ++i)
                if (ev[i] != ey_f[i]) {
                    if (!nd) first = i;
                    ++nd;
                    md = std::max(md, (double) std::fabs(ev[i] - ey_f[i]));
                }
            std::printf("    %-8s == DN_F: %s (%zu differ, max %.3g, first row %zu col %zu: %g vs %g)\n", name,
                        nd == 0 ? "yes" : "NO", nd, md, first / D, first % D, ev[first], ey_f[first]);
        }
        CK(cudaMemcpy(b.hid, hid_v2.data(), hid_v2.size() * 2, cudaMemcpyHostToDevice));
        new_gemm2<DN2>(b);
        CK(cudaStreamSynchronize(g_s));
        auto ey_v2 = to_host(b.ey, (size_t) b.R * D);
        print_err("ey, v2 (end to end)", compare(ey_v2, yref, so.inv, D, 0));
        CK(cudaMemsetAsync(b.ey, 0, (size_t) b.R * D * 4, g_s));
        w4a16::down(b.w_dn, DN_ROW, DN_EXP, FF, D, b.hid, b.bounds, E, b.R, b.ey, g_s);
        CK(cudaStreamSynchronize(g_s));
        auto ey_p = to_host(b.ey, (size_t) b.R * D);
        std::printf("    production down == v2: %s\n",
                    std::memcmp(ey_p.data(), ey_v2.data(), ey_p.size() * 4) == 0 ? "yes" : "NO");
        // the exact hidden (fp64 reference) through each down GEMM alone
        std::vector<float> h_sorted((size_t) b.R * FF);
        std::vector<__half> h16((size_t) b.R * FF);
        for (size_t p = 0; p < so.inv.size(); ++p)
            for (int c2 = 0; c2 < FF; ++c2) {
                const float v = href[p * FF + c2];
                h_sorted[(size_t) so.inv[p] * FF + c2] = v;
                h16[(size_t) so.inv[p] * FF + ((c2 & ~15) | dv::v1::perm16(c2 & 15))] = __float2half_rn(v);
            }
        CK(cudaMemcpy(b.eh, h_sorted.data(), h_sorted.size() * 4, cudaMemcpyHostToDevice));
        mmq::quantize(b.eh, nullptr, b.xq, kQ4_0, FF, FF, b.R, g_s);
        old_dn(b);
        CK(cudaStreamSynchronize(g_s));
        print_err("ey, old down alone (exact hidden)", compare(to_host(b.ey, (size_t) b.R * D), yref, so.inv, D, 0));
        CK(cudaMemcpy(b.hid, h16.data(), h16.size() * 2, cudaMemcpyHostToDevice));
        new_gemm<DN_S>(b);
        CK(cudaStreamSynchronize(g_s));
        print_err("ey, w4a16 down alone, scale on acc", compare(to_host(b.ey, (size_t) b.R * D), yref, so.inv, D, 0));
        new_gemm<DN_F>(b);
        CK(cudaStreamSynchronize(g_s));
        print_err("ey, w4a16 down alone, scale folded", compare(to_host(b.ey, (size_t) b.R * D), yref, so.inv, D, 0));
        dfree(xq8, xq2);
        b.xq = nullptr;
        dfree(b.ey, (size_t) b.R * D);
        free_weights(w2);
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ speed

struct Cand {
    const char* name;
    std::function<void()> f;
    int iters = 1;
    std::vector<double> ms;
};
double time_once(const std::function<void()>& f, int iters) {
    cudaEvent_t a, z;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&z));
    CK(cudaEventRecord(a, g_s));
    for (int i = 0; i < iters; ++i) f();
    CK(cudaEventRecord(z, g_s));
    CK(cudaEventSynchronize(z));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, z));
    cudaEventDestroy(a);
    cudaEventDestroy(z);
    return ms / iters;
}
// interleaved rounds of ~loop_ms loops; min and median per call
void bench(std::vector<Cand>& cs, double loop_ms, int rounds) {
    for (auto& c : cs) {
        c.f();
        const double t = time_once(c.f, 3);
        c.iters = std::max(1, (int) (loop_ms / std::max(t, 1e-3)));
    }
    for (int r = 0; r < rounds; ++r)
        for (auto& c : cs) c.ms.push_back(time_once(c.f, c.iters));
}
double vmin(std::vector<double> v) { return *std::min_element(v.begin(), v.end()); }
double vq(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[(size_t) (q * (v.size() - 1) + 0.5)];
}
double vmed(std::vector<double> v) { return vq(v, 0.5); }

// every GEMM candidate by name (gate_up: GU_*, down: DN_*)
struct Variant {
    const char* name;
    int mode;
    std::function<void(const Bufs&)> run;
    std::function<void()> info;
};
template <class C> Variant V2(const char* name) {
    return {name, C::MODE, [](const Bufs& b) { new_gemm2<C>(b); }, [name] { print_cfg2<C>(name); }};
}
template <class C> Variant V(const char* name) {
    return {name, C::MODE, [](const Bufs& b) { new_gemm<C>(b); }, [name] { print_cfg<C>(name); }};
}
std::vector<Variant> variants() {
    return {V2<GU2>("GU2"), V2<DN2>("DN2"), V2<GU2P2>("GU2P2"), V2<GU2P4>("GU2P4"), V2<DN2P2>("DN2P2"),
            V2<DN2P4>("DN2P4"), V2<GU2S4>("GU2S4"), V2<GU2S8>("GU2S8"), V2<DN2S4>("DN2S4"), V2<GU2RF>("GU2RF"), V2<GU2RFS>("GU2RFS"), V2<GU2W4>("GU2W4"), V2<DN2W4>("DN2W4"), V2<DN2W8>("DN2W8"), V2<GU2RA<7>>("W7_nocp"),
            V2<GU2RA<8>>("W8_copyonly"), V2<GU2A<6>>("V6_nomma"), V2<GU2A<7>>("V7_nocp"), V2<GU2A<8>>("V8_copyonly"),
            V<GU_S>("GU_S"), V<GU_F>("GU_F"), V<GU_W8>("GU_W8"), V<GU_W8K4>("GU_W8K4"), V<GU_W8B32>("GU_W8B32"),
            V<GU_FN8>("GU_FN8"), V<GU_BIG>("GU_BIG"), V<GU_R>("GU_R"), V<GU_RW8>("GU_RW8"), V<GU_RN8>("GU_RN8"), V<GU_RA<6>>("R6_nomma"),
            V<GU_RA<4>>("R4_nowcp"), V<GU_RA<8>>("R8_copyonly"), V<GU_RA<9>>("R9_wonly"), V<GU_RA<10>>("R10_aonly"), V<GU_FA<8>>("A8_copyonly"), V<GU_FA<1>>("A1_nodq"), V<GU_FA<2>>("A2_noA"), V<GU_FA<3>>("A3_noB"),
            V<GU_FA<4>>("A4_nowcp"), V<GU_FA<5>>("A5_noacp"), V<GU_FA<6>>("A6_nomma"), V<GU_FA<7>>("A7_nocp"), V<GU_BA<6>>("B6_nomma"), V<GU_BA<7>>("B7_nocp"), V<GU_BIG2>("GU_BIG2"), V<DN_BIG>("DN_BIG"), V<DN_S>("DN_S"), V<DN_F>("DN_F"), V<DN_W8>("DN_W8"), V<DN_W8B32>("DN_W8B32"),
            V<DN_W4>("DN_W4")};
}
bool wanted(const std::string& filter, const char* name) {
    if (filter.empty()) return true;
    const std::string f = "," + filter + ",", n = std::string(",") + name + ",";
    return f.find(n) != std::string::npos;
}

// what: "speed" (interleaved loops) or "prof" (one launch each, for ncu)
int run_speed(const std::string& dir, int il, const std::string& route, double loop_ms, int rounds,
              const std::string& filter, bool prof) {
    int64_t T, c;
    auto ids = read_mat<int32_t>(route, T, c);
    Sorted so = sort_rows(ids, (int) T);
    Bufs b;
    b.T = (int) T;
    b.R = T * KU;
    b.max_rows = so.max_rows;
    int tiles64 = 0;
    for (int e = 0; e < E; ++e) tiles64 += (so.bounds[e + 1] - so.bounds[e] + 63) / 64;
    std::printf("%s: %lld tokens, %lld rows, rows an expert max %d mean %.1f; %d tiles of 64 rows (%.0f%% filled)\n",
                route.c_str(), (long long) T, (long long) b.R, so.max_rows, b.R / (double) E, tiles64,
                100.0 * b.R / (64.0 * tiles64));
    std::vector<float> g((size_t) T * D);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 0.4f);
    for (auto& v : g) v = nd(rng);
    b.g = dalloc<float>(g.size());
    CK(cudaMemcpy(b.g, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
    b.bounds = dalloc<int32_t>(E + 1);
    b.src = dalloc<int32_t>(b.R);
    b.inv = dalloc<int32_t>(b.R);
    b.iota = dalloc<int32_t>(b.R + 2);
    CK(cudaMemcpy(b.bounds, so.bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.src, so.src.data(), b.R * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.inv, so.inv.data(), b.R * 4, cudaMemcpyHostToDevice));
    mmq::iota(b.iota, b.R + 2, g_s);
    const std::string L = dir + "/L" + std::to_string(il);
    const double flop[2] = {2.0 * b.R * D * 2 * FF, 2.0 * b.R * FF * D};
    const auto vs = variants();

    double old_tot = 0, new_tot = 0;
    for (int mode = 0; mode < 2; ++mode) {
        // weights: 32 experts' worth of physical memory mapped three times over the virtual range
        auto wh = read_file(L + (mode == 0 ? ".gate_up" : ".down"));
        Weights w = load_weights(wh, (size_t) (mode == 0 ? 72 : 36) << 20);
        size_t A, B, Cq = 0;
        uint8_t *bufA, *bufB, *bufC = nullptr;
        if (mode == 0) {   // A: old gu (f32) / new xs (fp16); B: old q8_1 input / new hidden
            b.w_gu = w.ptr();
            A = std::max((size_t) b.R * 2 * FF * 4, (size_t) b.R * D * 2);
            B = std::max(mmq::q8_bytes(b.R, D), (size_t) b.R * FF * 2);
            bufA = dalloc<uint8_t>(A);
            bufB = dalloc<uint8_t>(B);
            b.gu = (float*) bufA;
            b.xs = (__half*) bufA;
            b.xq = bufB;
            b.hid = (__half*) bufB;
        } else {   // A: ey (the old geglu's gu input aliases it); B: old eh / new hidden; C: old q8_1
            b.w_dn = w.ptr();
            A = std::max((size_t) b.R * D * 4, (size_t) b.R * 2 * FF * 4);
            B = (size_t) b.R * FF * 4;
            Cq = mmq::q8_bytes(b.R, FF);
            bufA = dalloc<uint8_t>(A);
            bufB = dalloc<uint8_t>(B);
            bufC = dalloc<uint8_t>(Cq);
            b.ey = (float*) bufA;
            b.gu = (float*) bufA;
            b.eh = (float*) bufB;
            b.hid = (__half*) bufB;
            b.xq = bufC;
        }
        std::printf("  %s phase: %.0f MiB on the device\n", mode == 0 ? "gate_up" : "down", g_dev_bytes / 1048576.0);
        std::vector<Cand> cs;
        int n_old = 0;
        if (mode == 0) {
            cs.push_back({"old q8_1 gather", [&] { old_quant_in(b); }});
            cs.push_back({"old MMQ gate_up", [&] { old_gu(b); }});
            cs.push_back({"new scatter fp16", [&] { new_scatter(b); }});
            n_old = 2;
        } else {
            cs.push_back({"old geglu+q8_1", [&] { old_geglu_quant(b); }});
            cs.push_back({"old MMQ down", [&] { old_dn(b); }});
            n_old = 2;
        }
        const size_t first_new = cs.size();
        for (const auto& v : vs)
            if (v.mode == mode && wanted(filter, v.name)) cs.push_back({v.name, [&b, &v] { v.run(b); }});
        if (prof) {
            for (auto& cnd : cs) cnd.f();
            CK(cudaStreamSynchronize(g_s));
        } else {
            bench(cs, loop_ms, rounds);
            double best_new = 1e30;
            for (size_t i = 0; i < cs.size(); ++i) {
                const auto& cnd = cs[i];
                const bool gemm = i == 1 || i >= first_new;
                std::printf("    %-18s min %7.3f  p10 %7.3f  med %7.3f ms   x30 %6.1f ms", cnd.name, vmin(cnd.ms),
                            vq(cnd.ms, 0.1), vmed(cnd.ms), 30 * vmin(cnd.ms));
                if (gemm) std::printf("  %5.1f TFLOPS", flop[mode] / (vmin(cnd.ms) * 1e9));
                std::printf("\n");
                if (i >= first_new) best_new = std::min(best_new, vmin(cnd.ms));
            }
            for (int i = 0; i < n_old; ++i) old_tot += vmin(cs[i].ms);
            if (mode == 0) new_tot += vmin(cs[2].ms);
            if (best_new < 1e30) new_tot += best_new;
        }
        dfree(bufA, A);
        dfree(bufB, B);
        if (bufC) dfree(bufC, Cq);
        free_weights(w);
    }
    if (!prof)
        std::printf("  MoE expert work a layer: old %.3f ms (x30 %.1f)  new (best) %.3f ms (x30 %.1f)  = %.2fx\n",
                    old_tot, 30 * old_tot, new_tot, 30 * new_tot, old_tot / new_tot);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s num <dir> <layer> [tokens] | speed|prof <dir> <layer> <route.i32> [loop_ms] [rounds] "
                             "[variants,...]\n", argv[0]);
        return 1;
    }
    CK(cudaFree(0));
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    g_mq = new mmq::Context();
    const std::string mode = argv[1];
    const std::string filter = argc > 7 ? argv[7] : "";
    std::printf("kernels:\n");
    for (const auto& v : variants())
        if (wanted(filter, v.name)) v.info();
    int rc = 1;
    if (mode == "num") rc = run_num(argv[2], std::atoi(argv[3]), argc > 4 ? std::atoi(argv[4]) : 0);
    else if ((mode == "speed" || mode == "prof") && argc >= 5)
        rc = run_speed(argv[2], std::atoi(argv[3]), argv[4], argc > 5 ? std::atof(argv[5]) : 4.0,
                       argc > 6 ? std::atoi(argv[6]) : 60, filter, mode == "prof");
    CK(cudaDeviceSynchronize());
    return rc;
}
