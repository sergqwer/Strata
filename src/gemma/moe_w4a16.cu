// src/gemma/moe_w4a16.cu - see include/strata/gemma/moe_w4a16.hpp; the kernels are in moe_w4a16.cuh.
#include "strata/gemma/moe_w4a16.hpp"

#include "moe_w4a16.cuh"

#include <stdexcept>
#include <string>

namespace strata::gemma::w4a16 {
namespace {

// v2 (moe_w4a16.cuh): 11 warps x 32 weight rows a block, 64 activation rows, weights through a per-row 128-byte line
// ring two k-tiles ahead, activations straight into the B fragments (their L1 lines prefetched two k32 blocks ahead),
// block scales folded into the fp16 weights (as accurate as scaling the fp32 partial sums: rel. error 2.448e-4 vs
// 2.449e-4 on real layers, and faster)
using GateUp = dev::Cfg<0, 11, 8, 2, false, 0, 2>;
using Down = dev::Cfg<1, 11, 8, 2, false, 0, 2>;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("w4a16 ") + what + ": " + cudaGetErrorString(e));
}

template <class C> void launch(const dev::Args& a, int64_t rows, int n_out_rows, cudaStream_t s) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(dev::gemm_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        attr = true;
    }
    dev::gemm_kernel<C><<<dev::grid_of<C>(rows, a.n_expert, n_out_rows), C::THREADS, C::SMEM, s>>>(a);
}

constexpr int kQ4_0 = 2;   // ggml type id

}  // namespace

bool supported(int type_gate_up, int type_down, int d, int ff, size_t gu_row_bytes, size_t down_row_bytes,
               const void* gu, const void* down, size_t gu_expert_bytes, size_t down_expert_bytes) {
    if (type_gate_up != kQ4_0 || type_down != kQ4_0) return false;
    // 64-value k-tiles; gate_up blocks of BM/2 hidden units, down blocks of BM output rows; 32-value fpos blocks
    if (d % 64 || ff % 64 || ff % (GateUp::BM / 2) || d % Down::BM) return false;
    if (gu_row_bytes != (size_t) d / 32 * 18 || down_row_bytes != (size_t) ff / 32 * 18) return false;
    auto aligned = [](size_t v, size_t a) { return v % a == 0; };
    // gate_up rows are copied in 16-byte units, down rows in 4-byte ones
    return aligned(gu_row_bytes, 16) && aligned(gu_expert_bytes, 16) && aligned((size_t) gu, 16) &&
           aligned(down_row_bytes, 4) && aligned(down_expert_bytes, 4) && aligned((size_t) down, 4);
}

void scatter(const float* g, int d, const int32_t* inv, int tokens, int k, __half* xs, cudaStream_t s) {
    if (tokens <= 0) return;
    const int64_t n = (int64_t) tokens * (d / 32);
    dev::scatter_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(g, d, inv, tokens, k, xs);
    check("scatter");
}

void gate_up(const void* w, size_t row_bytes, size_t expert_bytes, int d, int ff, const __half* xs,
             const int32_t* bounds, int n_expert, int64_t rows, __half* hidden, cudaStream_t s) {
    if (rows <= 0) return;
    dev::Args a;
    a.w = static_cast<const uint8_t*>(w);
    a.row_bytes = row_bytes;
    a.expert_bytes = expert_bytes;
    a.K = d;
    a.ff = ff;
    a.x = xs;
    a.bounds = bounds;
    a.n_expert = n_expert;
    a.hidden = hidden;
    launch<GateUp>(a, rows, ff, s);
    check("gate_up");
}

void down(const void* w, size_t row_bytes, size_t expert_bytes, int ff, int d, const __half* hidden,
          const int32_t* bounds, int n_expert, int64_t rows, float* ey, cudaStream_t s) {
    if (rows <= 0) return;
    dev::Args a;
    a.w = static_cast<const uint8_t*>(w);
    a.row_bytes = row_bytes;
    a.expert_bytes = expert_bytes;
    a.K = ff;
    a.x = hidden;
    a.bounds = bounds;
    a.n_expert = n_expert;
    a.ey = ey;
    a.ld_ey = d;
    launch<Down>(a, rows, d, s);
    check("down");
}

}  // namespace strata::gemma::w4a16
