// src/gemma/vision.cu - see include/strata/gemma/vision.hpp.
#include "strata/gemma/vision.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::gemma {
namespace {

using bf16 = __nv_bfloat16;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("vision ") + what + ": " + cudaGetErrorString(e));
}
void cb(cublasStatus_t e, const char* what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("vision cuBLAS ") + what + ": " + std::to_string((int) e));
}
void kcheck(const char* what) { ck(cudaGetLastError(), what); }
unsigned nblk(int64_t n, int bs = 256) { return (unsigned) ((n + bs - 1) / bs); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[32];
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = l < (int) (blockDim.x / 32) ? sh[l] : 0.f;
    return warp_sum(v);
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float shm[32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) shm[w] = v;
    __syncthreads();
    v = l < (int) (blockDim.x / 32) ? shm[l] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// ---------------------------------------------------------------------------------------------------- kernels

// patches [n][3 * P * P] f16, column (c, ky, kx) = the conv kernel's ggml layout; value = px / 255 * 2 - 1
// a batch of same-size images back to back: block g is patch g % n of image g / n
__global__ void patchify_kernel(const uint8_t* __restrict__ img, int nx, int P, int ncols, __half* __restrict__ out,
                                int n) {
    const int g = blockIdx.x, p = g % n;
    img += (size_t) (g / n) * n * P * P * 3;
    const int px = p % ncols, py = p / ncols;
    for (int j = threadIdx.x; j < 3 * P * P; j += blockDim.x) {
        const int c = j / (P * P), ky = (j / P) % P, kx = j % P;
        const int x = px * P + kx, y = py * P + ky;
        const float v = (float) img[((size_t) y * nx + x) * 3 + c] / 255.0f;
        out[(size_t) g * 3 * P * P + j] = __float2half(v * 2.0f + -1.0f);
    }
}

__global__ void add_pos_kernel(float* __restrict__ x, const float* __restrict__ tbl, int pos_size, int D, int ncols,
                               int n) {
    const int g = blockIdx.x, p = g % n;   // row g: patch p of image g / n
    const int px = p % ncols, py = p / ncols;
    const float* tx = tbl + (size_t) px * D;
    const float* ty = tbl + ((size_t) pos_size + py) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x[(size_t) g * D + i] + tx[i];
        x[(size_t) g * D + i] = v + ty[i];
    }
}

// y_bf16 = rms(x) * w
__global__ void rms_bf16_kernel(const float* __restrict__ x, const float* __restrict__ w, bf16* __restrict__ y, int D,
                                float eps) {
    const size_t r = blockIdx.x;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += x[r * D + i] * x[r * D + i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x[r * D + i] * sc;
        if (w) v *= w[i];
        y[r * D + i] = __float2bfloat16(v);
    }
}

// W8A8: rms(x) * w rounded to bf16 (what the bf16 path feeds its GEMM), then int8 with the row's scale (absmax / 127)
__global__ void rms_i8_kernel(const float* __restrict__ x, const float* __restrict__ w, int8_t* __restrict__ q,
                              float* __restrict__ s, int D, float eps) {
    extern __shared__ float t[];
    const size_t r = blockIdx.x;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += x[r * D + i] * x[r * D + i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / D + eps);
    float mx = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x[r * D + i] * sc;
        if (w) v *= w[i];
        v = __bfloat162float(__float2bfloat16(v));
        t[i] = v;
        mx = fmaxf(mx, fabsf(v));
    }
    mx = block_max(mx);
    const float scale = fmaxf(mx, 1e-12f) / 127.f, inv = 1.f / scale;
    for (int i = threadIdx.x; i < D; i += blockDim.x) q[r * D + i] = (int8_t) __float2int_rn(t[i] * inv);
    if (threadIdx.x == 0) s[r] = scale;
}

// x += rms(y) * w; y either the bf16 GEMM output or (W8A8) the int32 GEMM output times the row and column scales
__global__ void add_rms_kernel(float* __restrict__ x, const bf16* __restrict__ y, const float* __restrict__ w, int D,
                               float eps, const int32_t* __restrict__ yi, const float* __restrict__ sx,
                               const float* __restrict__ sw) {
    const size_t r = blockIdx.x;
    auto yv = [&](int i) { return yi ? (float) yi[r * D + i] * sx[r] * sw[i] : __bfloat162float(y[r * D + i]); };
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { const float v = yv(i); ss += v * v; }
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) x[r * D + i] = yv(i) * sc * w[i] + x[r * D + i];
}

// per patch (one block), a warp per (q|k|v, head) slot of the bf16 GEMM output (W8A8: int32 * scales): rms * norm ->
// 2-D rope for q / k, rms for v; written f16 [head][n][HDP], zero-padded. hd <= 96: a lane holds elements lane,
// lane + 32, lane + 64.
__global__ void vit_qkv_kernel(const bf16* __restrict__ qkv, const float* __restrict__ qn, const float* __restrict__ kn,
                               __half* __restrict__ Q, __half* __restrict__ K, __half* __restrict__ V, int n, int H,
                               int hd, int HDP, int ncols, float eps, float theta_scale,
                               const int32_t* __restrict__ qi, const float* __restrict__ sx, const float* __restrict__ sw) {
    __shared__ float buf[8][96];
    const int g = blockIdx.x, p = g % n, warp = threadIdx.x / 32, lane = threadIdx.x % 32;   // row g: patch p, image g / n
    const size_t img_off = (size_t) (g / n) * H * n * HDP;
    const float pos_x = (float) (p % ncols), pos_y = (float) (p / ncols);
    const bool swap = theta_scale < 0.f;   // test hook: x / y swapped
    const float ts = fabsf(theta_scale);
    for (int slot = warp; slot < 3 * H; slot += 8) {
        const int which = slot / H, h = slot % H;
        const size_t col0 = (size_t) which * H * hd + (size_t) h * hd;   // W8A8: int32 * row scale * column scale
        const bf16* src = qkv + (size_t) g * 3 * H * hd + col0;
        const int32_t* srci = qi ? qi + (size_t) g * 3 * H * hd + col0 : nullptr;
        const float rs = qi ? sx[g] : 0.f;
        float v[3];
        float ss = 0.f;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const int i = lane + 32 * c;
            v[c] = i < hd ? (srci ? (float) srci[i] * rs * sw[col0 + i] : __bfloat162float(src[i])) : 0.f;
            ss += v[c] * v[c];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float sc = rsqrtf(ss / hd + eps);
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const int i = lane + 32 * c;
            float y = v[c] * sc;
            if (which < 2 && i < hd) y *= (which == 0 ? qn : kn)[i];
            if (i < 96) buf[warp][i] = y;
        }
        __syncwarp();
        __half* dst = (which == 0 ? Q : which == 1 ? K : V) + img_off + ((size_t) h * n + p) * HDP;
        const int half = hd / 2, quarter = hd / 4;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const int i = lane + 32 * c;
            if (i >= HDP) continue;
            float y = i < hd ? buf[warp][i] : 0.f;
            if (which < 2 && i < hd) {
                const int base = i < half ? 0 : half;
                const int j = (i - base) % quarter;
                const bool lo = (i - base) < quarter;
                const float a = buf[warp][base + j], b = buf[warp][base + j + quarter];
                const float pos = (i < half) != swap ? pos_x : pos_y;
                const float theta = pos * powf(ts, (float) j);
                const float cs = cosf(theta), sn = sinf(theta);
                y = lo ? a * cs - b * sn : a * sn + b * cs;
            }
            dst[i] = __float2half(y);
        }
        __syncwarp();
    }
}

// W8A8: one row: gate / up from the int32 GEMM output, gelu-quick(g) * u rounded to bf16, then int8 with its scale
__global__ void geglu_i8_kernel(const int32_t* __restrict__ gu, const float* __restrict__ sx,
                                const float* __restrict__ sw, int8_t* __restrict__ q, float* __restrict__ s, int F) {
    extern __shared__ float t[];
    const size_t r = blockIdx.x;
    const float rs = sx[r];
    float mx = 0.f;
    for (int c = threadIdx.x; c < F; c += blockDim.x) {
        const float g = (float) gu[r * 2 * F + c] * rs * sw[c], u = (float) gu[r * 2 * F + F + c] * rs * sw[F + c];
        const float h = __bfloat162float(__float2bfloat16(g * (1.0f / (1.0f + expf(-1.702f * g))) * u));
        t[c] = h;
        mx = fmaxf(mx, fabsf(h));
    }
    mx = block_max(mx);
    const float scale = fmaxf(mx, 1e-12f) / 127.f, inv = 1.f / scale;
    for (int c = threadIdx.x; c < F; c += blockDim.x) q[r * F + c] = (int8_t) __float2int_rn(t[c] * inv);
    if (threadIdx.x == 0) s[r] = scale;
}

__global__ void geglu_quick_kernel(const bf16* __restrict__ gu, bf16* __restrict__ h, int64_t n, int F) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * F) return;
    const int64_t r = i / F, c = i % F;
    const float g = __bfloat162float(gu[r * 2 * F + c]), u = __bfloat162float(gu[r * 2 * F + F + c]);
    h[i] = __float2bfloat16(g * (1.0f / (1.0f + expf(-1.702f * g))) * u);
}

// 3x3 (k x k) average pool over the patch grid, then * scale, (x - bias) * std_scale, rms -> bf16
__global__ void pool_kernel(const float* __restrict__ x, int ncols, int k, int D, float scale,
                            const float* __restrict__ sb, const float* __restrict__ ssc, bf16* __restrict__ out,
                            float eps, int n, int n_tok) {
    extern __shared__ float t[];
    const int tok = blockIdx.x % n_tok, ox_n = ncols / k;
    x += (size_t) (blockIdx.x / n_tok) * n * D;   // image blockIdx.x / n_tok of the batch
    const int ox = tok % ox_n, oy = tok / ox_n;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.f;
        for (int dy = 0; dy < k; ++dy)
            for (int dx = 0; dx < k; ++dx) acc += x[((size_t) (oy * k + dy) * ncols + ox * k + dx) * D + i] / (float) (k * k);
        float v = acc * scale;
        if (sb) v = (v - sb[i]) * ssc[i];
        t[i] = v;
    }
    __syncthreads();
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += t[i] * t[i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[(size_t) blockIdx.x * D + i] = __float2bfloat16(t[i] * sc);
}

__global__ void f32_to_f16_kernel(const float* __restrict__ a, __half* __restrict__ b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __float2half(a[i]);
}
__global__ void bf16_rows_kernel(const uint16_t* __restrict__ a, bf16* __restrict__ b, int64_t n) {   // raw copy
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __ushort_as_bfloat16(a[i]);
}

// ------------------------------------------------------------------- flash attention (non-causal), WMMA f16
//
// Q, K, V: [head][n][HDP] f16 (head_dim padded with zeros to HDP); O: [n][H * hd] bf16. A block owns 64 query rows of
// one head (4 warps x 16 rows) and streams the keys in tiles of 64: S = Q K^T on tensor cores into shared memory,
// the online softmax per row in registers (two lanes per row), O += P V on tensor cores with O kept in shared memory.
namespace wm = nvcuda::wmma;
constexpr int FA_BQ = 64, FA_BK = 64, FA_THREADS = 128;

template <int HDP>
struct FaSmem {
    // padded row strides keep the WMMA loads and the per-row softmax off shared-memory bank conflicts
    static constexpr int LH = HDP + 8;    // halves per Q / K / V row
    static constexpr int LS = FA_BK + 4;  // floats per S row
    static constexpr int LP = FA_BK + 8;  // halves per P row
    static constexpr int LO = HDP + 4;    // floats per O row
    static constexpr size_t q = 0, k = q + FA_BQ * LH * 2, v = k + FA_BK * LH * 2, s = v + FA_BK * LH * 2,
                            p = s + FA_BQ * LS * 4, o = p + FA_BQ * LP * 2, total = o + FA_BQ * LO * 4;
};

template <int HDP>
__global__ void __launch_bounds__(FA_THREADS) vit_fa_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                                            const __half* __restrict__ V, bf16* __restrict__ O, int n,
                                                            int H, int hd) {
    extern __shared__ __align__(128) uint8_t smem[];
    using L = FaSmem<HDP>;
    constexpr int LH = L::LH, LS = L::LS, LP = L::LP, LO = L::LO;
    __half* Qs = reinterpret_cast<__half*>(smem + L::q);
    __half* Ks = reinterpret_cast<__half*>(smem + L::k);
    __half* Vs = reinterpret_cast<__half*>(smem + L::v);
    float* Ss = reinterpret_cast<float*>(smem + L::s);
    __half* Ps = reinterpret_cast<__half*>(smem + L::p);
    float* Os = reinterpret_cast<float*>(smem + L::o);
    const int h = blockIdx.y, q0 = blockIdx.x * FA_BQ;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    constexpr int VEC = HDP / 8;   // uint4 (8 halves) per row

    const __half* Qh = Q + (size_t) h * n * HDP;
    const __half* Kh = K + (size_t) h * n * HDP;
    const __half* Vh = V + (size_t) h * n * HDP;
    for (int i = threadIdx.x; i < FA_BQ * VEC; i += FA_THREADS) {
        const int r = i / VEC, c = i % VEC;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (q0 + r < n) val = reinterpret_cast<const uint4*>(Qh + (size_t) (q0 + r) * HDP)[c];
        *reinterpret_cast<uint4*>(Qs + r * LH + c * 8) = val;
    }
    for (int i = threadIdx.x; i < FA_BQ * LO; i += FA_THREADS) Os[i] = 0.f;
    // the warp's 16 rows: their running max and sum, every lane holds all 16
    float m[16], l[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.f;
    }

    for (int k0 = 0; k0 < n; k0 += FA_BK) {
        __syncthreads();
        for (int i = threadIdx.x; i < FA_BK * VEC; i += FA_THREADS) {
            const int r = i / VEC, c = i % VEC;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (k0 + r < n) {
                kv = reinterpret_cast<const uint4*>(Kh + (size_t) (k0 + r) * HDP)[c];
                vv = reinterpret_cast<const uint4*>(Vh + (size_t) (k0 + r) * HDP)[c];
            }
            *reinterpret_cast<uint4*>(Ks + r * LH + c * 8) = kv;
            *reinterpret_cast<uint4*>(Vs + r * LH + c * 8) = vv;
        }
        __syncthreads();
        // S rows [16 warp, +16) x 64 keys
#pragma unroll
        for (int ct = 0; ct < FA_BK / 16; ++ct) {
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < HDP / 16; ++kk) {
                wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> a;
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> b;
                wm::load_matrix_sync(a, Qs + 16 * warp * LH + kk * 16, LH);
                wm::load_matrix_sync(b, Ks + ct * 16 * LH + kk * 16, LH);
                wm::mma_sync(acc, a, b, acc);
            }
            wm::store_matrix_sync(Ss + 16 * warp * LS + ct * 16, acc, LS, wm::mem_row_major);
        }
        __syncwarp();
        // online softmax, one row at a time: lane owns columns lane and lane + 32
        const bool v0 = k0 + lane < n, v1 = k0 + lane + 32 < n;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int row = 16 * warp + r;
            const float s0 = v0 ? Ss[row * LS + lane] : -INFINITY;
            const float s1 = v1 ? Ss[row * LS + lane + 32] : -INFINITY;
            float mx = fmaxf(s0, s1);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_new = fmaxf(m[r], mx);
            const float corr = expf(m[r] - m_new);
            const float p0 = v0 ? expf(s0 - m_new) : 0.f, p1 = v1 ? expf(s1 - m_new) : 0.f;
            Ps[row * LP + lane] = __float2half(p0);
            Ps[row * LP + lane + 32] = __float2half(p1);
            float sum = p0 + p1;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            l[r] = l[r] * corr + sum;
            m[r] = m_new;
            for (int c = lane; c < HDP; c += 32) Os[row * LO + c] *= corr;
        }
        __syncwarp();
        // O rows [16 warp, +16) += P V
#pragma unroll
        for (int ct = 0; ct < HDP / 16; ++ct) {
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::load_matrix_sync(acc, Os + 16 * warp * LO + ct * 16, LO, wm::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < FA_BK / 16; ++kk) {
                wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> a;
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::row_major> b;
                wm::load_matrix_sync(a, Ps + 16 * warp * LP + kk * 16, LP);
                wm::load_matrix_sync(b, Vs + kk * 16 * LH + ct * 16, LH);
                wm::mma_sync(acc, a, b, acc);
            }
            wm::store_matrix_sync(Os + 16 * warp * LO + ct * 16, acc, LO, wm::mem_row_major);
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = 16 * warp + r;
        if (q0 + row >= n) break;
        const float inv = 1.f / l[r];
        bf16* out = O + (size_t) (q0 + row) * H * hd + (size_t) h * hd;
        for (int c = lane; c < hd; c += 32) out[c] = __float2bfloat16(Os[row * LO + c] * inv);
    }
}

// ---- the same attention with the scores, probabilities and output in registers (FlashAttention-2 layout):
// mma.sync m16n8k16 (f16 in, f32 accumulate). A warp owns 16 query rows; S = Q K^T lands in accumulators whose rows
// are lane / 4 and lane / 4 + 8, so the row max / sum are quad shuffles, and the f32 accumulators of P re-pack as the
// f16 A fragments of the P V product without touching shared memory. K is staged row-major, V transposed.
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t*>(&h);
}

template <int HDP>
__global__ void __launch_bounds__(128) vit_fa2_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                                      const __half* __restrict__ V, bf16* __restrict__ O, int n, int H,
                                                      int hd) {
    constexpr int BK = 64, LH = HDP + 8, LV = BK + 8, VEC = HDP / 8, NT = BK / 8, DT = HDP / 8, KK = HDP / 16;
    __shared__ __align__(16) __half Ks[BK * LH];
    __shared__ __align__(16) __half Vt[HDP * LV];
    const int h = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int r0 = blockIdx.x * 64 + warp * 16;
    {   // image blockIdx.z of a batch: its own Q / K / V blocks and output rows (each image attends only to itself)
        const size_t z = blockIdx.z;
        Q += z * H * n * HDP;
        K += z * H * n * HDP;
        V += z * H * n * HDP;
        O += z * n * H * hd;
    }
    const __half* Qh = Q + (size_t) h * n * HDP;
    const __half* Kh = K + (size_t) h * n * HDP;
    const __half* Vh = V + (size_t) h * n * HDP;
    uint32_t qa[KK][4];
#pragma unroll
    for (int kk = 0; kk < KK; ++kk)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r0 + g + (i & 1) * 8, col = kk * 16 + 2 * t + (i >> 1) * 8;
            qa[kk][i] = row < n ? *reinterpret_cast<const uint32_t*>(Qh + (size_t) row * HDP + col) : 0u;
        }
    float o[DT][4];
#pragma unroll
    for (int d = 0; d < DT; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    for (int k0 = 0; k0 < n; k0 += BK) {
        __syncthreads();
        for (int i = threadIdx.x; i < BK * VEC; i += 128) {
            const int r = i / VEC, c = i % VEC;
            uint4 kv = make_uint4(0, 0, 0, 0);
            if (k0 + r < n) kv = reinterpret_cast<const uint4*>(Kh + (size_t) (k0 + r) * HDP)[c];
            *reinterpret_cast<uint4*>(Ks + r * LH + c * 8) = kv;
        }
        for (int i = threadIdx.x; i < BK * VEC; i += 128) {
            const int r = i % BK, c = i / BK;
            uint4 vv = make_uint4(0, 0, 0, 0);
            if (k0 + r < n) vv = reinterpret_cast<const uint4*>(Vh + (size_t) (k0 + r) * HDP)[c];
            const __half* e = reinterpret_cast<const __half*>(&vv);
#pragma unroll
            for (int j = 0; j < 8; ++j) Vt[(c * 8 + j) * LV + r] = e[j];
        }
        __syncthreads();
        float s[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < KK; ++kk) {
                const __half* kp = Ks + (nt * 8 + g) * LH + kk * 16 + 2 * t;
                mma16816(s[nt], qa[kk], *reinterpret_cast<const uint32_t*>(kp), *reinterpret_cast<const uint32_t*>(kp + 8));
            }
            const int col = k0 + nt * 8 + 2 * t;
            if (col >= n) s[nt][0] = s[nt][2] = -INFINITY;
            if (col + 1 >= n) s[nt][1] = s[nt][3] = -INFINITY;
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float c0 = expf(m0 - mn0), c1 = expf(m1 - mn1);
        float sum0 = 0.f, sum1 = 0.f;
        uint32_t pa[BK / 16][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const float p00 = expf(s[nt][0] - mn0), p01 = expf(s[nt][1] - mn0);
            const float p10 = expf(s[nt][2] - mn1), p11 = expf(s[nt][3] - mn1);
            sum0 += p00 + p01;
            sum1 += p10 + p11;
            pa[nt / 2][(nt & 1) * 2 + 0] = pack_h2(p00, p01);
            pa[nt / 2][(nt & 1) * 2 + 1] = pack_h2(p10, p11);
        }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l0 = l0 * c0 + sum0;
        l1 = l1 * c1 + sum1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int d = 0; d < DT; ++d) {
            o[d][0] *= c0;
            o[d][1] *= c0;
            o[d][2] *= c1;
            o[d][3] *= c1;
        }
#pragma unroll
        for (int j = 0; j < BK / 16; ++j)
#pragma unroll
            for (int d = 0; d < DT; ++d) {
                const __half* vp = Vt + (d * 8 + g) * LV + j * 16 + 2 * t;
                mma16816(o[d], pa[j], *reinterpret_cast<const uint32_t*>(vp), *reinterpret_cast<const uint32_t*>(vp + 8));
            }
    }
    const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
    for (int d = 0; d < DT; ++d) {
        const int col = d * 8 + 2 * t;
        if (col >= hd) continue;
        const int ra = r0 + g, rb = r0 + g + 8;
        if (ra < n) {
            bf16* out = O + (size_t) ra * H * hd + (size_t) h * hd + col;
            out[0] = __float2bfloat16(o[d][0] * i0);
            out[1] = __float2bfloat16(o[d][1] * i0);
        }
        if (rb < n) {
            bf16* out = O + (size_t) rb * H * hd + (size_t) h * hd + col;
            out[0] = __float2bfloat16(o[d][2] * i1);
            out[1] = __float2bfloat16(o[d][3] * i1);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- the model

struct Vision::Impl {
    int D = 0, n_layer = 0, H = 0, hd = 0, F = 0, P = 0, merge = 3, n_out = 0, pos_size = 0;
    float eps = 1e-6f, theta = 100.f;
    static constexpr int HDP = 80;
    struct Layer {
        const float *ln1, *ln2, *post_attn, *post_ffn, *qn, *kn;
        const bf16 *wqkv, *wo, *wgu, *wdown;
        // int8: per-row int8 matrices and their row scales ([3D], [2F], [D])
        const int8_t *q8qkv = nullptr, *q8gu = nullptr, *q8down = nullptr;
        const float *sqkv = nullptr, *sgu = nullptr, *sdown = nullptr;
    };
    bool i8 = false;
    std::vector<Layer> layers;
    const __half* patch_w = nullptr;   // [D][3 P P]
    const float* pos = nullptr;        // [2][pos_size][D]
    const float *std_bias = nullptr, *std_scale = nullptr;
    const bf16* proj = nullptr;        // [n_out][D]
    void* arena = nullptr;
    size_t arena_bytes = 0;
    // work buffers
    int max_n = 0;
    uint8_t* img = nullptr;
    __half* patches = nullptr;
    float *x = nullptr, *outv = nullptr;
    bf16 *y = nullptr, *qkv = nullptr, *gu = nullptr;   // GEMM outputs (bf16; int32 in the int8 path's GEMMs)
    bf16 *hb = nullptr, *ab = nullptr, *fb = nullptr;
    int8_t* hq = nullptr;                    // int8: a GEMM input [N][<= F]
    float *sx1 = nullptr, *sx2 = nullptr;    // int8: its row scales (gate / up input, down input)
    __half *Qh = nullptr, *Kh = nullptr, *Vh = nullptr;
    void* work = nullptr;
    cublasHandle_t blas = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    // set_profile: per-layer phase boundaries (kVisPhases + 1 events a layer), milliseconds summed per phase
    bool prof = false;
    std::vector<cudaEvent_t> pev;
    std::vector<double> ptot;
    void mark(int il, int k, cudaStream_t s);
};

Vision::Vision(const std::string& path, int tokens, int max_patches, bool int8) : p_(new Impl), tokens_(tokens) {
    Impl& m = *p_;
    m.i8 = int8;
    GgufFile g(path);
    auto u = [&](const char* k) -> uint64_t {
        const MetaValue* v = g.get(k);
        if (!v) throw std::runtime_error(std::string("mmproj: missing ") + k);
        return v->u;
    };
    const MetaValue* pt = g.get("clip.vision.projector_type");
    if (!pt || pt->s != "gemma4v") throw std::runtime_error("mmproj: projector type is not gemma4v");
    m.D = (int) u("clip.vision.embedding_length");
    m.n_layer = (int) u("clip.vision.block_count");
    m.H = (int) u("clip.vision.attention.head_count");
    m.F = (int) u("clip.vision.feed_forward_length");
    m.P = (int) u("clip.vision.patch_size");
    m.n_out = (int) u("clip.vision.projection_dim");
    if (const MetaValue* e = g.get("clip.vision.attention.layer_norm_epsilon")) m.eps = (float) e->num();
    if (const MetaValue* sf = g.get("clip.vision.projector.scale_factor")) m.merge = (int) sf->u;
    m.hd = m.D / m.H;
    if (m.hd > Impl::HDP || m.hd % 4 || m.hd > 96) throw std::runtime_error("mmproj: head_dim " + std::to_string(m.hd) + " unsupported");
    n_out_ = m.n_out;

    // ---- weights: matrices as bf16 (they are BF16 in the file), the conv kernel as f16, the rest f32
    struct Item { const TensorInfo* ti; size_t off; int kind; size_t soff = 0; };   // kind 0 copy, 1 f32 -> f16,
                                                                                    // 2 bf16 -> int8 rows (+ scales)
    std::vector<Item> items;
    size_t total = 0;
    auto add = [&](const std::string& name, int kind, bool required = true) -> size_t {
        const TensorInfo* ti = g.find(name);
        if (!ti) {
            if (required) throw std::runtime_error("mmproj: missing " + name);
            return SIZE_MAX;
        }
        total = (total + 255) / 256 * 256;
        const size_t off = total;
        const size_t elems = ti->elements();
        const size_t bytes = kind == 2 ? elems : kind == 1 ? elems * 2 : ti->type == 0 ? elems * 4 : elems * 2;
        if (kind == 2 && ti->type != 30) throw std::runtime_error("mmproj: " + name + " is not BF16 (int8 needs BF16)");
        if (kind == 0 && ti->type != 0 && ti->type != 30) throw std::runtime_error("mmproj: " + name + " is " + ti->type_name());
        total += bytes;
        items.push_back({ti, off, kind});
        return off;
    };
    struct LOff { size_t ln1, ln2, pa, pf, qn, kn, q, k, v, o, g, up, d, sqkv = 0, sgu = 0, sd = 0; };
    auto reserve = [&](size_t bytes) { total = (total + 255) / 256 * 256; const size_t o = total; total += bytes; return o; };
    const int k8 = m.i8 ? 2 : 0;
    std::vector<LOff> lo(m.n_layer);
    const size_t o_patch = add("v.patch_embd.weight", 1);
    const size_t o_pos = add("v.position_embd.weight", 0);
    const size_t o_sb = add("v.std_bias", 0, false), o_ss = add("v.std_scale", 0, false);
    const size_t o_proj = add("mm.input_projection.weight", 0);
    for (int il = 0; il < m.n_layer; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        LOff& L = lo[il];
        L.ln1 = add(p + "ln1.weight", 0);
        L.ln2 = add(p + "ln2.weight", 0);
        L.pa = add(p + "attn_post_norm.weight", 0);
        L.pf = add(p + "ffn_post_norm.weight", 0);
        L.qn = add(p + "attn_q_norm.weight", 0);
        L.kn = add(p + "attn_k_norm.weight", 0);
        // q, k, v back to back = one [3D][D] matrix; gate, up = one [2F][D]
        const size_t iq = items.size();
        L.q = add(p + "attn_q.weight", k8);
        L.k = add(p + "attn_k.weight", k8);
        L.v = add(p + "attn_v.weight", k8);
        L.o = add(p + "attn_out.weight", 0);
        const size_t ig = items.size();
        L.g = add(p + "ffn_gate.weight", k8);
        L.up = add(p + "ffn_up.weight", k8);
        const size_t id = items.size();
        L.d = add(p + "ffn_down.weight", k8);
        const size_t es = m.i8 ? 1 : 2;
        if (L.k != L.q + (size_t) m.D * m.D * es || L.v != L.k + (size_t) m.D * m.D * es || L.up != L.g + (size_t) m.F * m.D * es)
            throw std::runtime_error("mmproj: unexpected matrix sizes (q/k/v must be D x D, gate/up F x D)");
        if (m.i8) {   // the row scales of the fused matrices back to back: [q | k | v], [gate | up], [down]
            L.sqkv = reserve((size_t) 3 * m.D * 4);
            L.sgu = reserve((size_t) 2 * m.F * 4);
            L.sd = reserve((size_t) m.D * 4);
            for (int j = 0; j < 3; ++j) items[iq + j].soff = L.sqkv + (size_t) j * m.D * 4;
            items[ig].soff = L.sgu;
            items[ig + 1].soff = L.sgu + (size_t) m.F * 4;
            items[id].soff = L.sd;
        }
    }
    ck(cudaMalloc(&m.arena, total), "malloc weights");
    m.arena_bytes = total;
    weight_bytes_ = total;
    ck(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), "stream");
    {
        std::vector<uint8_t> tmp;
        for (const Item& it : items) {
            const uint8_t* src = g.tensor_data(*it.ti);
            const size_t elems = it.ti->elements();
            uint8_t* dst = static_cast<uint8_t*>(m.arena) + it.off;
            if (it.kind == 2) {   // per output row: int8 = round(w / s), s = absmax / 127 (rows are the matrix's ne1)
                const int64_t in = it.ti->shape[0], rows = (int64_t) (elems / in);
                const uint16_t* b = reinterpret_cast<const uint16_t*>(src);
                std::vector<int8_t> q(elems);
                std::vector<float> sc(rows);
                for (int64_t r = 0; r < rows; ++r) {
                    float mx = 0.f;
                    for (int64_t i = 0; i < in; ++i) {
                        uint32_t u = (uint32_t) b[r * in + i] << 16;
                        float f;
                        std::memcpy(&f, &u, 4);
                        mx = std::max(mx, std::fabs(f));
                    }
                    const float s = std::max(mx, 1e-12f) / 127.f, inv = 1.f / s;
                    sc[r] = s;
                    for (int64_t i = 0; i < in; ++i) {
                        uint32_t u = (uint32_t) b[r * in + i] << 16;
                        float f;
                        std::memcpy(&f, &u, 4);
                        q[r * in + i] = (int8_t) std::lrint(std::max(-127.f, std::min(127.f, f * inv)));
                    }
                }
                ck(cudaMemcpy(dst, q.data(), elems, cudaMemcpyHostToDevice), "upload int8");
                ck(cudaMemcpy(static_cast<uint8_t*>(m.arena) + it.soff, sc.data(), rows * 4, cudaMemcpyHostToDevice),
                   "upload scales");
            } else if (it.kind == 1) {
                std::vector<__half> h(elems);
                const float* f = reinterpret_cast<const float*>(src);
                for (size_t i = 0; i < elems; ++i) h[i] = __float2half(f[i]);
                ck(cudaMemcpy(dst, h.data(), elems * 2, cudaMemcpyHostToDevice), "upload");
            } else {
                ck(cudaMemcpy(dst, src, elems * (it.ti->type == 0 ? 4 : 2), cudaMemcpyHostToDevice), "upload");
            }
        }
    }
    auto at = [&](size_t off) { return static_cast<uint8_t*>(m.arena) + off; };
    m.patch_w = reinterpret_cast<const __half*>(at(o_patch));
    m.pos = reinterpret_cast<const float*>(at(o_pos));
    m.pos_size = (int) g.find("v.position_embd.weight")->shape[1];
    m.std_bias = o_sb == SIZE_MAX ? nullptr : reinterpret_cast<const float*>(at(o_sb));
    m.std_scale = o_ss == SIZE_MAX ? nullptr : reinterpret_cast<const float*>(at(o_ss));
    m.proj = reinterpret_cast<const bf16*>(at(o_proj));
    for (const LOff& L : lo)
        m.layers.push_back({reinterpret_cast<const float*>(at(L.ln1)), reinterpret_cast<const float*>(at(L.ln2)),
                            reinterpret_cast<const float*>(at(L.pa)), reinterpret_cast<const float*>(at(L.pf)),
                            reinterpret_cast<const float*>(at(L.qn)), reinterpret_cast<const float*>(at(L.kn)),
                            reinterpret_cast<const bf16*>(at(L.q)), reinterpret_cast<const bf16*>(at(L.o)),
                            reinterpret_cast<const bf16*>(at(L.g)), reinterpret_cast<const bf16*>(at(L.d))});
    if (m.i8)
        for (size_t il = 0; il < lo.size(); ++il) {
            Impl::Layer& Ly = m.layers[il];
            const LOff& L = lo[il];
            Ly.wqkv = Ly.wgu = Ly.wdown = nullptr;   // those matrices exist only as int8
            Ly.q8qkv = reinterpret_cast<const int8_t*>(at(L.q));
            Ly.q8gu = reinterpret_cast<const int8_t*>(at(L.g));
            Ly.q8down = reinterpret_cast<const int8_t*>(at(L.d));
            Ly.sqkv = reinterpret_cast<const float*>(at(L.sqkv));
            Ly.sgu = reinterpret_cast<const float*>(at(L.sgu));
            Ly.sdown = reinterpret_cast<const float*>(at(L.sd));
        }

    // ---- work buffers
    const int N = max_patches;
    m.max_n = N;
    const size_t D = m.D, F = m.F;
    size_t off = 0;
    auto carve = [&](size_t bytes) { off = (off + 255) / 256 * 256; const size_t o = off; off += bytes; return o; };
    const size_t yb = m.i8 ? 4 : 2;   // GEMM output element: bf16, or int32 where the int8 path's GEMMs write
    const size_t b_img = carve((size_t) N * m.P * m.P * 3), b_patch = carve((size_t) N * 3 * m.P * m.P * 2),
                 b_x = carve(N * D * 4), b_y = carve(N * D * yb), b_qkv = carve(N * 3 * D * yb), b_gu = carve(N * 2 * F * yb),
                 b_out = carve((size_t) N * m.n_out * 4), b_hb = carve(N * D * 2), b_ab = carve(N * D * 2),
                 b_fb = carve(N * F * 2), b_q = carve((size_t) m.H * N * Impl::HDP * 2),
                 b_k = carve((size_t) m.H * N * Impl::HDP * 2), b_v = carve((size_t) m.H * N * Impl::HDP * 2);
    const size_t b_hq = m.i8 ? carve((size_t) N * F) : 0, b_s1 = m.i8 ? carve((size_t) N * 4) : 0,
                 b_s2 = m.i8 ? carve((size_t) N * 4) : 0;
    ck(cudaMalloc(&m.work, off), "malloc work");
    auto w8 = [&](size_t o) { return static_cast<uint8_t*>(m.work) + o; };
    m.img = w8(b_img);
    m.patches = reinterpret_cast<__half*>(w8(b_patch));
    m.x = reinterpret_cast<float*>(w8(b_x));
    m.y = reinterpret_cast<bf16*>(w8(b_y));
    m.qkv = reinterpret_cast<bf16*>(w8(b_qkv));
    m.gu = reinterpret_cast<bf16*>(w8(b_gu));
    m.outv = reinterpret_cast<float*>(w8(b_out));
    m.hb = reinterpret_cast<bf16*>(w8(b_hb));
    m.ab = reinterpret_cast<bf16*>(w8(b_ab));
    m.fb = reinterpret_cast<bf16*>(w8(b_fb));
    m.Qh = reinterpret_cast<__half*>(w8(b_q));
    m.Kh = reinterpret_cast<__half*>(w8(b_k));
    m.Vh = reinterpret_cast<__half*>(w8(b_v));
    if (m.i8) {
        m.hq = reinterpret_cast<int8_t*>(w8(b_hq));
        m.sx1 = reinterpret_cast<float*>(w8(b_s1));
        m.sx2 = reinterpret_cast<float*>(w8(b_s2));
    }
    weight_bytes_ += off;
    cb(cublasCreate(&m.blas), "create");
    cb(cublasSetStream(m.blas, s_), "stream");
    ck(cudaFuncSetAttribute(vit_fa_kernel<Impl::HDP>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int) FaSmem<Impl::HDP>::total), "fa smem");
    ck(cudaEventCreate(&m.e0), "event");
    ck(cudaEventCreate(&m.e1), "event");
}

Vision::~Vision() {
    if (!p_) return;
    if (p_->blas) cublasDestroy(p_->blas);
    if (p_->arena) cudaFree(p_->arena);
    if (p_->work) cudaFree(p_->work);
    if (p_->e0) cudaEventDestroy(p_->e0);
    if (p_->e1) cudaEventDestroy(p_->e1);
    if (s_) cudaStreamDestroy(s_);
}

ImageU8 Vision::preprocess(const ImageU8& raw, int tokens) const {
    if (tokens > 0) return preprocess_gemma4_hf(raw, p_->P, p_->merge, tokens);
    return preprocess_gemma4(raw, p_->P, p_->merge, tokens_, tokens_);
}

static constexpr int kVisPhases = 12;
static const char* kVisPhaseNames[kVisPhases] = {"rms1", "qkv_gemm", "qk_norm_rope", "attention", "attn_out_gemm",
                                                 "add_rms_rms2", "gate_up_gemm", "geglu", "down_gemm", "add_rms_post",
                                                 "patch_embed", "pool_proj"};

void Vision::Impl::mark(int il, int k, cudaStream_t s) {
    if (!prof) return;
    const size_t nb = kVisPhases + 1, need = (layers.size() + 1) * nb;
    if (pev.size() < need) {
        for (size_t i = pev.size(); i < need; ++i) {
            cudaEvent_t e;
            ck(cudaEventCreate(&e), "event");
            pev.push_back(e);
        }
        ptot.assign(kVisPhases, 0.0);
    }
    ck(cudaEventRecord(pev[(size_t) il * nb + k], s), "event record");
}

void Vision::set_profile(bool on) { p_->prof = on; }

std::vector<std::pair<const char*, double>> Vision::profile_take() {
    std::vector<std::pair<const char*, double>> out;
    if (p_->ptot.empty()) return out;
    for (int k = 0; k < kVisPhases; ++k) out.emplace_back(kVisPhaseNames[k], p_->ptot[k]);
    p_->ptot.assign(kVisPhases, 0.0);
    return out;
}

int Vision::encode(const ImageU8& img, std::vector<float>& out) {
    return encode_batch({&img}, out);
}

int Vision::max_batch(const ImageU8& img) const {
    const int n = (img.nx / p_->P) * (img.ny / p_->P);
    return n > 0 ? std::max(1, p_->max_n / n) : 1;
}

// B images of one size in one pass: every GEMM over all B * n patch rows (one 70-token video frame alone is 540 rows,
// too few to fill the GPU), attention per image (blockIdx.z), the pooled tokens image after image
int Vision::encode_batch(const std::vector<const ImageU8*>& imgs, std::vector<float>& out) {
    Impl& m = *p_;
    if (imgs.empty()) return 0;
    const ImageU8& img = *imgs[0];
    const int B = (int) imgs.size();
    const int P = m.P, ncols = img.nx / P, nrows = img.ny / P, n = ncols * nrows, N = B * n;
    if (img.nx % (P * m.merge) || img.ny % (P * m.merge)) throw std::runtime_error("vision: image size not aligned");
    for (const ImageU8* o : imgs)
        if (o->nx != img.nx || o->ny != img.ny) throw std::runtime_error("vision: a batch takes images of one size");
    if (N > m.max_n) throw std::runtime_error("vision: " + std::to_string(N) + " patches > max " + std::to_string(m.max_n));
    const int D = m.D, F = m.F, H = m.H, hd = m.hd;
    const int n_tok = (ncols / m.merge) * (nrows / m.merge), T = B * n_tok;
    cudaStream_t s = s_;
    ck(cudaEventRecord(m.e0, s), "event");
    for (int b = 0; b < B; ++b)
        ck(cudaMemcpyAsync(m.img + (size_t) b * img.rgb.size(), imgs[b]->rgb.data(), img.rgb.size(),
                           cudaMemcpyHostToDevice, s), "image upload");
    patchify_kernel<<<N, 256, 0, s>>>(m.img, img.nx, P, ncols, m.patches, n);
    kcheck("patchify");
    const float one = 1.f, zero = 0.f;
    // Y [n][out] = X [n][in] . W [out][in]^T, accumulated in f32; Y bf16 in the layers (as transformers runs the
    // encoder; the residual stream stays f32), f32 for the patch embedding and the projection
    auto gemm = [&](const void* W, cudaDataType wt, const void* X, cudaDataType xt, void* Y, cudaDataType yt, int rows,
                    int out_d, int in_d, const char* what) {
        cb(cublasGemmEx(m.blas, CUBLAS_OP_T, CUBLAS_OP_N, out_d, rows, in_d, &one, W, wt, in_d, X, xt, in_d, &zero, Y,
                        yt, out_d, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), what);
    };
    // W8A8: Y [n][out] int32 = Xq [n][in] . Wq [out][in]^T (scales applied by the consumer kernels)
    auto gemm8 = [&](const int8_t* W, const int8_t* X, int32_t* Y, int rows, int out_d, int in_d, const char* what) {
        const int32_t i1 = 1, i0 = 0;
        cb(cublasGemmEx(m.blas, CUBLAS_OP_T, CUBLAS_OP_N, out_d, rows, in_d, &i1, W, CUDA_R_8I, in_d, X, CUDA_R_8I, in_d,
                        &i0, Y, CUDA_R_32I, out_d, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT), what);
    };
    gemm(m.patch_w, CUDA_R_16F, m.patches, CUDA_R_16F, m.x, CUDA_R_32F, N, D, 3 * P * P, "patch");
    add_pos_kernel<<<N, 256, 0, s>>>(m.x, m.pos, m.pos_size, D, ncols, n);
    kcheck("add_pos");
    float theta_scale = powf(m.theta, -2.0f / (float) (hd / 2));
    if (std::getenv("STRATA_VIT_SWAPXY")) theta_scale = -theta_scale;
    const int nl = (int) m.layers.size();
    m.mark(nl, 0, s);   // the patch embedding: from here to layer 0's first mark
    for (int il = 0; il < nl; ++il) {
        const Impl::Layer& L = m.layers[il];
        m.mark(il, 0, s);
        if (m.i8)
            rms_i8_kernel<<<N, 256, D * sizeof(float), s>>>(m.x, L.ln1, m.hq, m.sx1, D, m.eps);
        else
            rms_bf16_kernel<<<N, 256, 0, s>>>(m.x, L.ln1, m.hb, D, m.eps);
        m.mark(il, 1, s);
        if (m.i8)
            gemm8(L.q8qkv, m.hq, reinterpret_cast<int32_t*>(m.qkv), N, 3 * D, D, "qkv8");
        else
            gemm(L.wqkv, CUDA_R_16BF, m.hb, CUDA_R_16BF, m.qkv, CUDA_R_16BF, N, 3 * D, D, "qkv");
        m.mark(il, 2, s);
        vit_qkv_kernel<<<N, 256, 0, s>>>(m.qkv, L.qn, L.kn, m.Qh, m.Kh, m.Vh, n, H, hd, Impl::HDP, ncols, m.eps,
                                          theta_scale, m.i8 ? reinterpret_cast<const int32_t*>(m.qkv) : nullptr,
                                          m.sx1, L.sqkv);
        kcheck("vit_qkv");
        m.mark(il, 3, s);
        vit_fa2_kernel<Impl::HDP><<<dim3((n + 63) / 64, H, B), 128, 0, s>>>(m.Qh, m.Kh, m.Vh, m.ab, n, H, hd);
        kcheck("vit_fa");
        m.mark(il, 4, s);
        gemm(L.wo, CUDA_R_16BF, m.ab, CUDA_R_16BF, m.y, CUDA_R_16BF, N, D, D, "attn_out");
        m.mark(il, 5, s);
        add_rms_kernel<<<N, 256, 0, s>>>(m.x, m.y, L.post_attn, D, m.eps, nullptr, nullptr, nullptr);
        if (m.i8) {
            rms_i8_kernel<<<N, 256, D * sizeof(float), s>>>(m.x, L.ln2, m.hq, m.sx1, D, m.eps);
            m.mark(il, 6, s);
            gemm8(L.q8gu, m.hq, reinterpret_cast<int32_t*>(m.gu), N, 2 * F, D, "gate_up8");
            m.mark(il, 7, s);
            geglu_i8_kernel<<<N, 256, F * sizeof(float), s>>>(reinterpret_cast<const int32_t*>(m.gu), m.sx1, L.sgu,
                                                             m.hq, m.sx2, F);
            m.mark(il, 8, s);
            gemm8(L.q8down, m.hq, reinterpret_cast<int32_t*>(m.y), N, D, F, "down8");
            m.mark(il, 9, s);
            add_rms_kernel<<<N, 256, 0, s>>>(m.x, m.y, L.post_ffn, D, m.eps, reinterpret_cast<const int32_t*>(m.y),
                                             m.sx2, L.sdown);
        } else {
            rms_bf16_kernel<<<N, 256, 0, s>>>(m.x, L.ln2, m.hb, D, m.eps);
            m.mark(il, 6, s);
            gemm(L.wgu, CUDA_R_16BF, m.hb, CUDA_R_16BF, m.gu, CUDA_R_16BF, N, 2 * F, D, "gate_up");
            m.mark(il, 7, s);
            geglu_quick_kernel<<<nblk((int64_t) N * F), 256, 0, s>>>(m.gu, m.fb, N, F);
            m.mark(il, 8, s);
            gemm(L.wdown, CUDA_R_16BF, m.fb, CUDA_R_16BF, m.y, CUDA_R_16BF, N, D, F, "down");
            m.mark(il, 9, s);
            add_rms_kernel<<<N, 256, 0, s>>>(m.x, m.y, L.post_ffn, D, m.eps, nullptr, nullptr, nullptr);
        }
        kcheck("layer");
        m.mark(il, 10, s);
    }
    m.mark(nl, 1, s);   // the pooling + projection: from here to the end
    pool_kernel<<<T, 256, D * sizeof(float), s>>>(m.x, ncols, m.merge, D, sqrtf((float) D), m.std_bias, m.std_scale,
                                                 m.hb, m.eps, n, n_tok);
    kcheck("pool");
    gemm(m.proj, CUDA_R_16BF, m.hb, CUDA_R_16BF, m.outv, CUDA_R_32F, T, m.n_out, D, "projection");
    const size_t base = out.size();
    out.resize(base + (size_t) T * m.n_out);
    ck(cudaMemcpyAsync(out.data() + base, m.outv, (size_t) T * m.n_out * sizeof(float), cudaMemcpyDeviceToHost, s), "download");
    ck(cudaEventRecord(m.e1, s), "event");
    m.mark(nl, 2, s);
    ck(cudaStreamSynchronize(s), "encode");
    cudaEventElapsedTime(&last_ms_, m.e0, m.e1);
    if (m.prof) {
        const size_t nb = kVisPhases + 1;
        float ms = 0;
        for (int il = 0; il < nl; ++il)
            for (int k = 0; k < 10; ++k) {
                cudaEventElapsedTime(&ms, m.pev[(size_t) il * nb + k], m.pev[(size_t) il * nb + k + 1]);
                m.ptot[k] += ms;
            }
        cudaEventElapsedTime(&ms, m.pev[(size_t) nl * nb], m.pev[0]);
        m.ptot[10] += ms;
        cudaEventElapsedTime(&ms, m.pev[(size_t) nl * nb + 1], m.pev[(size_t) nl * nb + 2]);
        m.ptot[11] += ms;
    }
    return n_tok;
}

}  // namespace strata::gemma
