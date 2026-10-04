// src/gemma/kernels.cu - see include/strata/gemma/kernels.hpp. The arithmetic follows llama.cpp's CUDA backend
// (ggml-cuda norm.cu, rope.cu, unary.cu, getrows) wherever the order of operations decides the last bits.
#include "strata/gemma/kernels.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::gemma::k {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("gemma kernel ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// sum over the block (blockDim.x a multiple of 32, <= 1024); every thread gets the result
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[32];
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    const int nw = blockDim.x / 32;
    v = l < nw ? sh[l] : 0.f;
    return warp_sum(v);
}
__device__ __forceinline__ float block_max(float v) {
    __shared__ float sh[32];
    v = warp_max(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    const int nw = blockDim.x / 32;
    v = l < nw ? sh[l] : -INFINITY;
    return warp_max(v);
}

__device__ __forceinline__ float gelu_tanh(float x) {   // ggml_cuda_op_gelu_single
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

// ------------------------------------------------------------------------------------------------ norms

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n,
                                float eps, float scale) {
    const size_t r = blockIdx.x;
    x += r * n;
    y += r * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i] * sc;
        if (w) v *= w[i];
        y[i] = v * scale;
    }
}

__global__ void add_rms_kernel(const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ w,
                               float* __restrict__ out, int n, float eps) {
    const size_t r = blockIdx.x;
    x += r * n;
    y += r * n;
    out += r * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += y[i] * y[i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = y[i] * sc * w[i] + x[i];
}

__global__ void ffn_norms_kernel(const float* __restrict__ x, const float* __restrict__ wf, const float* __restrict__ wg,
                                 const float* __restrict__ wr, float* __restrict__ f, float* __restrict__ g,
                                 float* __restrict__ rr, int n, float eps, float rscale) {
    const size_t r = blockIdx.x;
    x += r * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i] * sc;
        f[r * n + i] = v * wf[i];
        if (g) g[r * n + i] = v * wg[i];
        if (rr) rr[r * n + i] = v * rscale * wr[i];
    }
}

// t = rms(mlp) * pn1 + rms(moe) * pn2 (or t = mlp), out = (x1 + rms(t) * pfn) * out_scale; t lives in shared memory
__global__ void ffn_post_kernel(const float* __restrict__ x1, const float* __restrict__ mlp, const float* __restrict__ moe,
                                const float* __restrict__ pn1, const float* __restrict__ pn2,
                                const float* __restrict__ pfn, float out_scale, float* __restrict__ out, int n,
                                float eps) {
    extern __shared__ float t[];
    const size_t r = blockIdx.x;
    mlp += r * n;
    if (moe) {
        moe += r * n;
        float s1 = 0.f, s2 = 0.f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            s1 += mlp[i] * mlp[i];
            s2 += moe[i] * moe[i];
        }
        s1 = block_sum(s1);
        s2 = block_sum(s2);
        const float c1 = rsqrtf(s1 / n + eps), c2 = rsqrtf(s2 / n + eps);
        for (int i = threadIdx.x; i < n; i += blockDim.x) t[i] = mlp[i] * c1 * pn1[i] + moe[i] * c2 * pn2[i];
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) t[i] = mlp[i];
    }
    __syncthreads();
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += t[i] * t[i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[r * n + i] = (t[i] * sc * pfn[i] + x1[r * n + i]) * out_scale;
}

// ------------------------------------------------------------------------------------------------ embeddings

__device__ __forceinline__ float dequant_one(const uint8_t* __restrict__ row, int type, int i) {
    switch (type) {
        case 0: return reinterpret_cast<const float*>(row)[i];
        case 1: return __half2float(reinterpret_cast<const __half*>(row)[i]);
        case 2: {   // Q4_0: {f16 d; u8 qs[16]}, value j and j + 16 share byte j
            const uint8_t* b = row + (i / 32) * 18;
            const float d = __half2float(*reinterpret_cast<const __half*>(b));
            const int j = i % 32;
            const int q = j < 16 ? (b[2 + j] & 0xF) : (b[2 + j - 16] >> 4);
            return d * (q - 8);
        }
        case 8: {   // Q8_0: {f16 d; i8 qs[32]}
            const uint8_t* b = row + (i / 32) * 34;
            return __half2float(*reinterpret_cast<const __half*>(b)) * (float) reinterpret_cast<const int8_t*>(b + 2)[i % 32];
        }
        case 14: {   // Q6_K: {u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d}, dequantize_row_q6_K's index map
            const uint8_t* b = row + (i / 256) * 210;
            const uint8_t* ql = b;
            const uint8_t* qh = b + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
            const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
            const int j = i % 256;
            const int half = j / 128, jj = j % 128, quarter = jj / 32, l = jj % 32;
            ql += 64 * half;
            qh += 32 * half;
            sc += 8 * half;
            const int is = l / 16;
            int q;
            switch (quarter) {
                case 0: q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4); break;
                case 1: q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); break;
                case 2: q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); break;
                default: q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); break;
            }
            return d * sc[is + 2 * quarter] * (float) (q - 32);
        }
        default: return 0.f;
    }
}

__global__ void embed_kernel(const uint8_t* __restrict__ table, size_t row_bytes, int type, int n,
                             const int32_t* __restrict__ tokens, float* __restrict__ out, float scale) {
    const size_t r = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* row = table + (size_t) tokens[r] * row_bytes;
    out[r * n + i] = dequant_one(row, type, i) * scale;
}

// ------------------------------------------------------------------------------------------------ attention inputs

// grid (rows, n_head + 2 n_kv), block hd / 2: thread i owns the NEOX pair (i, i + hd / 2)
__global__ void qkv_prep_kernel(QkvArgs a, float theta_scale) {
    const int t = blockIdx.x, slot = blockIdx.y, i = threadIdx.x, hd = a.hd, h2 = hd / 2;
    const float* src;
    const float* w;
    bool rope;
    if (slot < a.n_head) {
        src = a.q + ((size_t) t * a.n_head + slot) * hd;
        w = a.q_norm;
        rope = true;
    } else if (slot < a.n_head + a.n_kv) {
        src = a.k + ((size_t) t * a.n_kv + (slot - a.n_head)) * hd;
        w = a.k_norm;
        rope = true;
    } else {
        const float* vb = a.v ? a.v : a.k;
        src = vb + ((size_t) t * a.n_kv + (slot - a.n_head - a.n_kv)) * hd;
        w = nullptr;
        rope = false;
    }
    float x0 = src[i], x1 = src[i + h2];
    const float ss = block_sum(x0 * x0 + x1 * x1);
    const float sc = rsqrtf(ss / hd + a.eps);
    x0 *= sc;
    x1 *= sc;
    if (w) {
        x0 *= w[i];
        x1 *= w[i + h2];
    }
    if (rope) {
        const float ff = a.freq_factors ? a.freq_factors[i] : 1.0f;
        const float theta = (float) a.pos[t] * powf(theta_scale, (float) i) / ff;
        const float c = cosf(theta), s = sinf(theta);
        const float y0 = x0 * c - x1 * s, y1 = x0 * s + x1 * c;
        x0 = y0;
        x1 = y1;
    }
    if (slot < a.n_head) {
        float* q = a.q + ((size_t) t * a.n_head + slot) * hd;
        q[i] = x0;
        q[i + h2] = x1;
        if (a.q16) {
            __half* q16 = a.q16 + ((size_t) t * a.n_head + slot) * hd;
            q16[i] = __float2half(x0);
            q16[i + h2] = __float2half(x1);
        }
    } else {
        const int g = slot < a.n_head + a.n_kv ? slot - a.n_head : slot - a.n_head - a.n_kv;
        __half* c = (slot < a.n_head + a.n_kv ? a.kc : a.vc) + ((size_t) a.pos[t] * a.n_kv + g) * hd;
        c[i] = __float2half(x0);
        c[i + h2] = __float2half(x1);
    }
}

// ------------------------------------------------------------------------------------------------ decode attention

constexpr int ATT_WARPS = 4;
constexpr int ATT_CHUNK = 128;   // keys per block

template <int HD>
__global__ void attn_decode_kernel(const float* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ vc,
                                   const int32_t* __restrict__ lo_a, const int32_t* __restrict__ hi_a,
                                   float* __restrict__ part, int n_head, int n_kv, int n_split) {
    constexpr int E = HD / 32;   // elements per lane: 8 or 16 (one or two 16-byte loads)
    const int h = blockIdx.x, t = blockIdx.y, sp = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = h / (n_head / n_kv);
    const int lo = max(lo_a[t], sp * ATT_CHUNK), hi = min(hi_a[t], sp * ATT_CHUNK + ATT_CHUNK - 1);
    float qv[E], acc[E];
    const float* qp = q + ((size_t) t * n_head + h) * HD + lane * E;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        qv[e] = qp[e];
        acc[e] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (int j = lo + warp; j <= hi; j += ATT_WARPS) {
        const __half* kp = kc + ((size_t) j * n_kv + g) * HD + lane * E;
        float s = 0.f;
#pragma unroll
        for (int e0 = 0; e0 < E; e0 += 8) {
            const uint4 raw = *reinterpret_cast<const uint4*>(kp + e0);
            const __half2* k2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 kf = __half22float2(k2[u]);
                s += qv[e0 + 2 * u] * kf.x + qv[e0 + 2 * u + 1] * kf.y;
            }
        }
        s = warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new), p = expf(s - m_new);
        l = l * corr + p;
        const __half* vp = vc + ((size_t) j * n_kv + g) * HD + lane * E;
#pragma unroll
        for (int e0 = 0; e0 < E; e0 += 8) {
            const uint4 raw = *reinterpret_cast<const uint4*>(vp + e0);
            const __half2* v2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 vf = __half22float2(v2[u]);
                acc[e0 + 2 * u] = acc[e0 + 2 * u] * corr + p * vf.x;
                acc[e0 + 2 * u + 1] = acc[e0 + 2 * u + 1] * corr + p * vf.y;
            }
        }
        m = m_new;
    }
    // combine the warps through shared memory
    __shared__ float sm[ATT_WARPS], sl[ATT_WARPS];
    __shared__ float sacc[ATT_WARPS][HD];
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
#pragma unroll
    for (int e = 0; e < E; ++e) sacc[warp][lane * E + e] = acc[e];
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < ATT_WARPS; ++w)
        if (sl[w] > 0.f) M = fmaxf(M, sm[w]);
    float L = 0.f;
    float f[ATT_WARPS];
#pragma unroll
    for (int w = 0; w < ATT_WARPS; ++w) {
        f[w] = sl[w] > 0.f ? expf(sm[w] - M) : 0.f;
        L += sl[w] * f[w];
    }
    float* out = part + (((size_t) t * n_head + h) * n_split + sp) * (HD + 2);
    for (int e = threadIdx.x; e < HD; e += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < ATT_WARPS; ++w) v += sacc[w][e] * f[w];
        out[2 + e] = v;
    }
    if (threadIdx.x == 0) {
        out[0] = M;
        out[1] = L;
    }
}

__global__ void attn_combine_kernel(const float* __restrict__ part, float* __restrict__ out, int n_head, int n_split,
                                    int hd) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float* p = part + ((size_t) t * n_head + h) * n_split * (hd + 2);
    float M = -INFINITY;
    for (int s = 0; s < n_split; ++s)
        if (p[s * (hd + 2) + 1] > 0.f) M = fmaxf(M, p[s * (hd + 2)]);
    float L = 0.f;
    for (int s = 0; s < n_split; ++s) {
        const float l = p[s * (hd + 2) + 1];
        if (l > 0.f) L += l * expf(p[s * (hd + 2)] - M);
    }
    const float inv = L > 0.f ? 1.f / L : 0.f;
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < n_split; ++s) {
            const float l = p[s * (hd + 2) + 1];
            if (l > 0.f) v += p[s * (hd + 2) + 2 + e] * expf(p[s * (hd + 2)] - M);
        }
        out[((size_t) t * n_head + h) * hd + e] = v * inv;
    }
}

// ------------------------------------------------------------------------------------------------ prompt attention

__global__ void mask_softmax_kernel(const float* __restrict__ S, __half* __restrict__ P, const int32_t* __restrict__ lo_a,
                                    const int32_t* __restrict__ hi_a, int rows, int t, int ld) {
    const int i = blockIdx.x, h = blockIdx.y;
    const size_t off = ((size_t) h * rows + i) * ld;
    const float* s = S + off;
    __half* p = P + off;
    const int lo = lo_a[i], hi = hi_a[i];
    float m = -INFINITY;
    for (int j = threadIdx.x; j < t; j += blockDim.x)
        if (j >= lo && j <= hi) m = fmaxf(m, s[j]);
    m = block_max(m);
    float sum = 0.f;
    for (int j = threadIdx.x; j < t; j += blockDim.x)
        if (j >= lo && j <= hi) sum += expf(s[j] - m);
    sum = block_sum(sum);
    const float inv = 1.f / sum;
    for (int j = threadIdx.x; j < ld; j += blockDim.x)
        p[j] = __float2half(j >= lo && j <= hi && j < t ? expf(s[j] - m) * inv : 0.f);
}

__global__ void key_ranges_kernel(const int32_t* __restrict__ pos, int rows, int n_swa, bool swa,
                                  const int32_t* __restrict__ spans, int n_spans, int32_t* __restrict__ lo,
                                  int32_t* __restrict__ hi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const int p = pos[i];
    int l = 0, h = p;
    if (swa) {
        l = max(0, p - n_swa + 1);
        for (int s = 0; s < n_spans; ++s)
            if (p >= spans[2 * s] && p < spans[2 * s + 1]) h = spans[2 * s + 1] - 1;
    }
    lo[i] = l;
    hi[i] = h;
}

// ------------------------------------------------------------------------------------------------ FFN / MoE

// one warp per row; n_expert <= 256
__global__ void router_topk_kernel(const float* __restrict__ logits, int n_expert, int k, int rows,
                                   int32_t* __restrict__ ids, float* __restrict__ w) {
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (r >= rows) return;
    constexpr int MAXPER = 8;
    const float* lg = logits + (size_t) r * n_expert;
    float v[MAXPER];
    float m = -INFINITY;
#pragma unroll
    for (int c = 0; c < MAXPER; ++c) {
        const int e = c * 32 + lane;
        v[c] = e < n_expert ? lg[e] : -INFINITY;
        m = fmaxf(m, v[c]);
    }
    m = warp_max(m);
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < MAXPER; ++c) {
        const int e = c * 32 + lane;
        v[c] = e < n_expert ? expf(v[c] - m) : 0.f;
        sum += v[c];
    }
    sum = warp_sum(sum);
#pragma unroll
    for (int c = 0; c < MAXPER; ++c) v[c] = (c * 32 + lane) < n_expert ? v[c] / sum : -INFINITY;
    float picked[16];
    float wsum = 0.f;
    for (int it = 0; it < k; ++it) {
        float best = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int c = 0; c < MAXPER; ++c) {
            const int e = c * 32 + lane;
            if (v[c] > best || (v[c] == best && e < bi)) {
                best = v[c];
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) {
                best = ob;
                bi = oi;
            }
        }
        if ((bi % 32) == lane) v[bi / 32] = -INFINITY;
        picked[it] = best;
        wsum += best;
        if (lane == 0) ids[(size_t) r * k + it] = bi;
    }
    wsum = fmaxf(wsum, 6.103515625e-5f);
    if (lane == 0)
        for (int it = 0; it < k; ++it) w[(size_t) r * k + it] = picked[it] / wsum;
}

__global__ void geglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                             int64_t rows, int n_ff, int64_t ld) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, c = i % n_ff;
    h[i] = gelu_tanh(gate[r * ld + c]) * up[r * ld + c];
}

__global__ void moe_combine_kernel(const float* __restrict__ y, const int32_t* __restrict__ inv,
                                   const int32_t* __restrict__ ids, const float* __restrict__ w,
                                   const float* __restrict__ scale, float* __restrict__ out, int n, int k) {
    const size_t t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int j = 0; j < k; ++j) {
        const size_t p = t * k + j;
        const size_t row = inv ? (size_t) inv[p] : p;
        float v = y[row * n + i];
        if (scale) v *= scale[ids[p]];
        v *= w[p];
        acc = j == 0 ? v : acc + v;
    }
    out[t * n + i] = acc;
}

__global__ void softcap_kernel(float* x, int64_t n, float cap) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] * (1.0f / cap)) * cap;
}

__global__ void moe_count_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&counts[ids[i]], 1);
}
__global__ void moe_scan_kernel(int32_t* __restrict__ counts, int32_t* __restrict__ bounds, int n_expert) {
    if (threadIdx.x != 0) return;
    int acc = 0;
    for (int e = 0; e < n_expert; ++e) {
        bounds[e] = acc;
        const int c = counts[e];
        counts[e] = acc;   // becomes the cursor
        acc += c;
    }
    bounds[n_expert] = acc;
}
__global__ void moe_scatter_kernel(const int32_t* __restrict__ ids, int n, int k, int32_t* __restrict__ cursor,
                                   int32_t* __restrict__ src, int32_t* __restrict__ inv) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const int r = atomicAdd(&cursor[ids[p]], 1);
    src[r] = p / k;
    inv[p] = r;
}

__global__ void iota_kernel(int32_t* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = i;
}

unsigned nblk(int64_t n, int bs = 256) { return (unsigned) ((n + bs - 1) / bs); }

}  // namespace

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, float scale, cudaStream_t s) {
    if (rows <= 0) return;
    rms_norm_kernel<<<rows, 256, 0, s>>>(x, w, y, n, eps, scale);
    check("rms_norm");
}

void embed_rows(const void* table, int type, int n, const int32_t* tokens, float* out, int rows, float scale,
                cudaStream_t s) {
    if (rows <= 0) return;
    size_t rb;
    switch (type) {
        case 0: rb = (size_t) n * 4; break;
        case 1: rb = (size_t) n * 2; break;
        case 2: rb = (size_t) n / 32 * 18; break;
        case 8: rb = (size_t) n / 32 * 34; break;
        case 14: rb = (size_t) n / 256 * 210; break;
        default: throw std::runtime_error("embed_rows: unsupported type " + std::to_string(type));
    }
    embed_kernel<<<dim3(nblk(n), rows), 256, 0, s>>>(static_cast<const uint8_t*>(table), rb, type, n, tokens, out, scale);
    check("embed_rows");
}

void qkv_prep(const QkvArgs& a, cudaStream_t s) {
    if (a.rows <= 0) return;
    if (a.n_rot != a.hd) throw std::runtime_error("qkv_prep: partial rotary (n_rot != head_dim) is not implemented");
    if (a.hd % 64 || a.hd > 2048) throw std::runtime_error("qkv_prep: head_dim must be a multiple of 64");
    const float theta_scale = powf(a.base, -2.0f / (float) a.n_rot);
    qkv_prep_kernel<<<dim3(a.rows, a.n_head + 2 * a.n_kv), a.hd / 2, 0, s>>>(a, theta_scale);
    check("qkv_prep");
}

size_t attn_decode_scratch_bytes(int rows, int n_head, int hd, int t_max) {
    const int n_split = (t_max + ATT_CHUNK - 1) / ATT_CHUNK;
    return (size_t) rows * n_head * n_split * (hd + 2) * sizeof(float);
}

void attn_decode(const float* q, const __half* kc, const __half* vc, const int32_t* lo, const int32_t* hi, float* out,
                 int rows, int n_head, int n_kv, int hd, int t_max, void* scratch, cudaStream_t s) {
    if (rows <= 0) return;
    const int n_split = (t_max + ATT_CHUNK - 1) / ATT_CHUNK;
    float* part = static_cast<float*>(scratch);
    const dim3 grid(n_head, rows, n_split);
    switch (hd) {
        case 256: attn_decode_kernel<256><<<grid, ATT_WARPS * 32, 0, s>>>(q, kc, vc, lo, hi, part, n_head, n_kv, n_split); break;
        case 512: attn_decode_kernel<512><<<grid, ATT_WARPS * 32, 0, s>>>(q, kc, vc, lo, hi, part, n_head, n_kv, n_split); break;
        default: throw std::runtime_error("attn_decode: head_dim " + std::to_string(hd) + " is not instantiated");
    }
    check("attn_decode");
    attn_combine_kernel<<<dim3(n_head, rows), 256, 0, s>>>(part, out, n_head, n_split, hd);
    check("attn_combine");
}

void mask_softmax(const float* S, __half* P, const int32_t* lo, const int32_t* hi, int n_head, int rows, int t,
                  int ld, cudaStream_t s) {
    if (rows <= 0) return;
    mask_softmax_kernel<<<dim3(rows, n_head), 256, 0, s>>>(S, P, lo, hi, rows, t, ld);
    check("mask_softmax");
}

void add_rms(const float* x, const float* y, const float* w, float* out, int n, int rows, float eps, cudaStream_t s) {
    if (rows <= 0) return;
    add_rms_kernel<<<rows, 256, 0, s>>>(x, y, w, out, n, eps);
    check("add_rms");
}

void ffn_norms(const float* x, const float* w_f, const float* w_g, const float* w_r, float* f, float* g, float* r,
               int n, int rows, float eps, cudaStream_t s) {
    if (rows <= 0) return;
    ffn_norms_kernel<<<rows, 256, 0, s>>>(x, w_f, w_g, w_r, f, g, r, n, eps, 1.0f / sqrtf((float) n));
    check("ffn_norms");
}

void router_topk(const float* logits, int n_expert, int k, int rows, int32_t* ids, float* w, cudaStream_t s) {
    if (rows <= 0) return;
    if (n_expert > 256 || k > 16) throw std::runtime_error("router_topk: n_expert <= 256 and k <= 16");
    router_topk_kernel<<<nblk(rows, 4), 128, 0, s>>>(logits, n_expert, k, rows, ids, w);
    check("router_topk");
}

void geglu(const float* gate, const float* up, float* h, int64_t rows, int n_ff, int64_t ld, cudaStream_t s) {
    if (rows <= 0) return;
    geglu_kernel<<<nblk(rows * n_ff), 256, 0, s>>>(gate, up, h, rows, n_ff, ld);
    check("geglu");
}

void moe_combine(const float* y, const int32_t* inv, const int32_t* ids, const float* w, const float* scale,
                 float* out, int n, int rows, int k, cudaStream_t s) {
    if (rows <= 0) return;
    moe_combine_kernel<<<dim3(nblk(n), rows), 256, 0, s>>>(y, inv, ids, w, scale, out, n, k);
    check("moe_combine");
}

void ffn_post(const float* x1, const float* mlp, const float* moe, const float* pn1, const float* pn2, const float* pfn,
              float out_scale, float* x_out, int n, int rows, float eps, cudaStream_t s) {
    if (rows <= 0) return;
    ffn_post_kernel<<<rows, 256, (size_t) n * sizeof(float), s>>>(x1, mlp, moe, pn1, pn2, pfn, out_scale, x_out, n, eps);
    check("ffn_post");
}

void softcap(float* logits, int64_t n, float cap, cudaStream_t s) {
    if (n <= 0 || cap <= 0.f) return;
    softcap_kernel<<<nblk(n), 256, 0, s>>>(logits, n, cap);
    check("softcap");
}

void moe_sort(const int32_t* ids, int rows, int k, int n_expert, int32_t* bounds, int32_t* src, int32_t* inv,
              int32_t* counts, cudaStream_t s) {
    const int n = rows * k;
    if (n <= 0) return;
    cudaMemsetAsync(counts, 0, sizeof(int32_t) * n_expert, s);
    moe_count_kernel<<<nblk(n), 256, 0, s>>>(ids, n, counts);
    moe_scan_kernel<<<1, 32, 0, s>>>(counts, bounds, n_expert);
    moe_scatter_kernel<<<nblk(n), 256, 0, s>>>(ids, n, k, counts, src, inv);
    check("moe_sort");
}

void key_ranges(const int32_t* pos, int rows, int n_swa, bool swa, const int32_t* spans, int n_spans, int32_t* lo,
                int32_t* hi, cudaStream_t s) {
    if (rows <= 0) return;
    key_ranges_kernel<<<nblk(rows), 256, 0, s>>>(pos, rows, n_swa, swa, spans, n_spans, lo, hi);
    check("key_ranges");
}

void iota(int32_t* dst, int n, cudaStream_t s) {
    if (n <= 0) return;
    iota_kernel<<<nblk(n), 256, 0, s>>>(dst, n);
    check("iota");
}

}  // namespace strata::gemma::k

// ================================================================================================ decode fusions
namespace strata::gemma::k {
namespace {
using strata::kernels::q8_1_ds;
using strata::kernels::q8_1_finite;
using strata::kernels::q8_1_quant;

struct Q81 {
    half2 ds;
    int8_t qs[32];
};

// sum over a 1024-thread block, every thread gets the result
__device__ __forceinline__ float bsum(float v, float* sh) {
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = l < (int) (blockDim.x / 32) ? sh[l] : 0.f;
    return warp_sum(v);
}

// q8_1 of row values held in shared memory: warp w quantizes blocks w, w + 32, ...
__device__ __forceinline__ void quant_row(const float* row, Q81* out, int n) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
    for (int b = warp; b < n / 32; b += nw) {
        const float xi = row[b * 32 + lane];
        const float amax = warp_max(fabsf(xi));
        const float sum = warp_sum(xi);
        const float d = q8_1_finite(amax / 127.0f);
        out[b].qs[lane] = q8_1_quant(xi, d, amax);
        if (lane == 0) out[b].ds = q8_1_ds(d, sum);
    }
}

constexpr int FT = 1024;   // fused kernels: threads per row

__global__ void __launch_bounds__(FT) norm_quant_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                         Q81* __restrict__ xq, int n, float eps) {
    extern __shared__ float row[];
    __shared__ float sh[32];
    const size_t r = blockIdx.x;
    x += r * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += FT) {
        row[i] = x[i];
        ss += row[i] * row[i];
    }
    ss = bsum(ss, sh);
    const float sc = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += FT) row[i] = row[i] * sc * w[i];
    __syncthreads();
    quant_row(row, xq + r * (n / 32), n);
}

__global__ void __launch_bounds__(FT) post_attn_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                                        const float* __restrict__ w_pa, float* __restrict__ x1,
                                                        const float* __restrict__ w_f, Q81* __restrict__ xqf,
                                                        const float* __restrict__ w_g, Q81* __restrict__ xqg,
                                                        const float* __restrict__ w_r, float* __restrict__ rr, int n,
                                                        float eps, float rscale) {
    extern __shared__ float sm[];
    float* a = sm;        // x1
    float* b = sm + n;    // scratch row for quantization
    __shared__ float sh[32];
    const size_t r = blockIdx.x;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += FT) {
        const float v = y[r * n + i];
        b[i] = v;
        ss += v * v;
    }
    ss = bsum(ss, sh);
    float sc = rsqrtf(ss / n + eps);
    float s2 = 0.f;
    for (int i = threadIdx.x; i < n; i += FT) {
        const float v = b[i] * sc * w_pa[i] + x[r * n + i];
        a[i] = v;
        x1[r * n + i] = v;
        s2 += v * v;
    }
    s2 = bsum(s2, sh);
    sc = rsqrtf(s2 / n + eps);
    for (int i = threadIdx.x; i < n; i += FT) b[i] = a[i] * sc * w_f[i];
    __syncthreads();
    quant_row(b, xqf + r * (n / 32), n);
    if (xqg) {
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += FT) {
            const float v = a[i] * sc;
            b[i] = v * w_g[i];
            rr[r * n + i] = v * rscale * w_r[i];
        }
        __syncthreads();
        quant_row(b, xqg + r * (n / 32), n);
    }
}

__global__ void geglu_quant_kernel(const float* __restrict__ gate, const float* __restrict__ up, int64_t ld,
                                   Q81* __restrict__ xq, int n_ff) {
    const size_t r = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // n_ff is a multiple of 32: whole warps return
    if (i >= n_ff) return;
    const float g = gate[r * ld + i];
    const float xi = gelu_tanh(g) * up[r * ld + i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);
    Q81* blk = xq + r * (n_ff / 32) + i / 32;
    blk->qs[i % 32] = q8_1_quant(xi, d, amax);
    if (i % 32 == 0) blk->ds = q8_1_ds(d, sum);
}

__global__ void __launch_bounds__(FT) moe_post_kernel(const float* __restrict__ ey, const int32_t* __restrict__ ids,
                                                       const float* __restrict__ ew, const float* __restrict__ scale, int k,
                                                       const float* __restrict__ mlp, const float* __restrict__ x1,
                                                       const float* __restrict__ pn1, const float* __restrict__ pn2,
                                                       const float* __restrict__ pfn, float out_scale,
                                                       float* __restrict__ x_out, const float* __restrict__ w_next,
                                                       Q81* __restrict__ xq, int n, float eps, float* __restrict__ h_out) {
    extern __shared__ float sm[];
    float* t = sm;   // the FFN sum, then the next layer's normed input
    __shared__ float sh[32];
    const size_t r = blockIdx.x;
    float s1 = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < n; i += FT) {
        const float m = mlp[r * n + i];
        s1 += m * m;
        if (ey) {
            float acc = 0.f;
            for (int j = 0; j < k; ++j) {
                const size_t p = r * k + j;
                float v = ey[p * n + i];
                if (scale) v *= scale[ids[p]];
                v *= ew[p];
                acc = j == 0 ? v : acc + v;
            }
            t[i] = acc;
            s2 += acc * acc;
        }
    }
    s1 = bsum(s1, sh);
    if (ey) {
        s2 = bsum(s2, sh);
        const float c1 = rsqrtf(s1 / n + eps), c2 = rsqrtf(s2 / n + eps);
        for (int i = threadIdx.x; i < n; i += FT) t[i] = mlp[r * n + i] * c1 * pn1[i] + t[i] * c2 * pn2[i];
    } else {
        for (int i = threadIdx.x; i < n; i += FT) t[i] = mlp[r * n + i];
    }
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += FT) ss += t[i] * t[i];
    ss = bsum(ss, sh);
    float sc = rsqrtf(ss / n + eps);
    float s3 = 0.f;
    for (int i = threadIdx.x; i < n; i += FT) {
        const float v = (t[i] * sc * pfn[i] + x1[r * n + i]) * out_scale;
        x_out[r * n + i] = v;
        t[i] = v;
        s3 += v * v;
    }
    if (!w_next) return;
    s3 = bsum(s3, sh);
    sc = rsqrtf(s3 / n + eps);
    for (int i = threadIdx.x; i < n; i += FT) {
        t[i] = t[i] * sc * w_next[i];
        if (h_out) h_out[r * n + i] = t[i];
    }
    __syncthreads();
    quant_row(t, xq + r * (n / 32), n);
}

__global__ void mtp_ranges_kernel(const int32_t* pos, int n_swa, int32_t* lo_s, int32_t* hi_s, int32_t* lo_g,
                                  int32_t* hi_g) {
    const int p = pos[0];
    lo_s[0] = max(0, p - n_swa + 1);
    hi_s[0] = p - 1;
    lo_g[0] = 0;
    hi_g[0] = p - 1;
}

__global__ void embed_concat_kernel(const uint8_t* __restrict__ table, size_t row_bytes, int type, int n1,
                                    const int32_t* __restrict__ token, float scale, const float* __restrict__ h, int n2,
                                    float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n1) out[i] = dequant_one(table + (size_t) token[0] * row_bytes, type, i) * scale;
    else if (i < n1 + n2) out[i] = h[i - n1];
}

// grid (ceil(n_expert / 8), rows), 8 warps: warp = one expert row
__global__ void router_gemv_kernel(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ out,
                                   int n_expert, int n) {
    const int e = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    const size_t r = blockIdx.y;
    if (e >= n_expert) return;
    const float4* w4 = reinterpret_cast<const float4*>(W + (size_t) e * n);
    const float4* x4 = reinterpret_cast<const float4*>(x + r * n);
    float acc = 0.f;
    for (int i = lane; i < n / 4; i += 32) {
        const float4 a = w4[i], b = x4[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) out[r * n_expert + e] = acc;
}

__global__ void attn_combine_quant_kernel(const float* __restrict__ part, float* __restrict__ out, Q81* __restrict__ xq,
                                          int n_head, int n_split, int hd) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float* p = part + ((size_t) t * n_head + h) * n_split * (hd + 2);
    __shared__ float w[64];
    __shared__ float Ls;
    if (threadIdx.x == 0) {
        float M = -INFINITY;
        for (int s = 0; s < n_split; ++s)
            if (p[s * (hd + 2) + 1] > 0.f) M = fmaxf(M, p[s * (hd + 2)]);
        float L = 0.f;
        for (int s = 0; s < n_split; ++s) {
            const float l = p[s * (hd + 2) + 1];
            w[s] = l > 0.f ? expf(p[s * (hd + 2)] - M) : 0.f;
            L += l * w[s];
        }
        Ls = L > 0.f ? 1.f / L : 0.f;
    }
    __syncthreads();
    const int e = threadIdx.x;   // blockDim.x == hd
    float v = 0.f;
    for (int s = 0; s < n_split; ++s)
        if (w[s] != 0.f) v += p[s * (hd + 2) + 2 + e] * w[s];
    v *= Ls;
    const size_t base = (size_t) t * n_head * hd + (size_t) h * hd;
    out[base + e] = v;
    if (xq) {
        const float amax = warp_max(fabsf(v));
        const float sum = warp_sum(v);
        const float d = q8_1_finite(amax / 127.0f);
        Q81* blk = xq + (base + e) / 32;
        blk->qs[e % 32] = q8_1_quant(v, d, amax);
        if (e % 32 == 0) blk->ds = q8_1_ds(d, sum);
    }
}

__global__ void argmax_kernel(const float* __restrict__ x, int64_t n, int32_t* __restrict__ ids) {
    const size_t r = blockIdx.x;
    x += r * n;
    float best = -INFINITY;
    int bi = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > best) {
            best = x[i];
            bi = (int) i;
        }
    __shared__ float sv[32];
    __shared__ int si[32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    if (l == 0) {
        sv[w] = best;
        si[w] = bi;
    }
    __syncthreads();
    if (w == 0) {
        best = l < (int) (blockDim.x / 32) ? sv[l] : -INFINITY;
        bi = l < (int) (blockDim.x / 32) ? si[l] : 0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) {
                best = ov;
                bi = oi;
            }
        }
        if (l == 0) ids[r] = bi;
    }
}

}  // namespace

void norm_quant(const float* x, const float* w, void* xq, int n, int rows, float eps, cudaStream_t s) {
    if (rows <= 0) return;
    norm_quant_kernel<<<rows, FT, n * sizeof(float), s>>>(x, w, static_cast<Q81*>(xq), n, eps);
    check("norm_quant");
}

void post_attn_fused(const float* x, const float* y, const float* w_pa, float* x1, const float* w_f, void* xqf,
                     const float* w_g, void* xqg, const float* w_r, float* r, int n, int rows, float eps,
                     cudaStream_t s) {
    if (rows <= 0) return;
    post_attn_kernel<<<rows, FT, 2 * n * sizeof(float), s>>>(x, y, w_pa, x1, w_f, static_cast<Q81*>(xqf), w_g,
                                                            static_cast<Q81*>(xqg), w_r, r, n, eps, 1.0f / sqrtf((float) n));
    check("post_attn_fused");
}

void geglu_quant(const float* gate, const float* up, int64_t ld, void* xq, int n_ff, int rows, cudaStream_t s) {
    if (rows <= 0) return;
    if (n_ff % 32) throw std::runtime_error("geglu_quant: n_ff must be a multiple of 32");
    geglu_quant_kernel<<<dim3(nblk(n_ff), rows), 256, 0, s>>>(gate, up, ld, static_cast<Q81*>(xq), n_ff);
    check("geglu_quant");
}

void moe_post_fused(const float* ey, const int32_t* ids, const float* ew, const float* scale, int k, const float* mlp,
                    const float* x1, const float* pn1, const float* pn2, const float* pfn, float out_scale, float* x_out,
                    const float* w_next, void* xq, int n, int rows, float eps, cudaStream_t s, float* h_out) {
    if (rows <= 0) return;
    moe_post_kernel<<<rows, FT, n * sizeof(float), s>>>(ey, ids, ew, scale, k, mlp, x1, pn1, pn2, pfn, out_scale, x_out,
                                                        w_next, static_cast<Q81*>(xq), n, eps, h_out);
    check("moe_post_fused");
}

void mtp_ranges(const int32_t* pos, int n_swa, int32_t* lo_swa, int32_t* hi_swa, int32_t* lo_g, int32_t* hi_g,
                cudaStream_t s) {
    mtp_ranges_kernel<<<1, 1, 0, s>>>(pos, n_swa, lo_swa, hi_swa, lo_g, hi_g);
    check("mtp_ranges");
}

void embed_concat(const void* table, int type, int n1, const int32_t* token, float scale, const float* h, int n2,
                  float* out, cudaStream_t s) {
    size_t rb;
    switch (type) {
        case 0: rb = (size_t) n1 * 4; break;
        case 1: rb = (size_t) n1 * 2; break;
        case 2: rb = (size_t) n1 / 32 * 18; break;
        case 8: rb = (size_t) n1 / 32 * 34; break;
        case 14: rb = (size_t) n1 / 256 * 210; break;
        default: throw std::runtime_error("embed_concat: unsupported type");
    }
    embed_concat_kernel<<<nblk(n1 + n2), 256, 0, s>>>(static_cast<const uint8_t*>(table), rb, type, n1, token, scale, h, n2, out);
    check("embed_concat");
}

void router_gemv(const float* W, const float* r, float* logits, int n_expert, int n, int rows, cudaStream_t s) {
    if (rows <= 0) return;
    if (n % 4) throw std::runtime_error("router_gemv: n must be a multiple of 4");
    router_gemv_kernel<<<dim3((n_expert + 7) / 8, rows), 256, 0, s>>>(W, r, logits, n_expert, n);
    check("router_gemv");
}

void attn_partials(const float* q, const __half* kc, const __half* vc, const int32_t* lo, const int32_t* hi,
                   void* scratch, int rows, int n_head, int n_kv, int hd, int n_split, cudaStream_t s) {
    if (rows <= 0) return;
    const dim3 grid(n_head, rows, n_split);
    float* part = static_cast<float*>(scratch);
    switch (hd) {
        case 256: attn_decode_kernel<256><<<grid, ATT_WARPS * 32, 0, s>>>(q, kc, vc, lo, hi, part, n_head, n_kv, n_split); break;
        case 512: attn_decode_kernel<512><<<grid, ATT_WARPS * 32, 0, s>>>(q, kc, vc, lo, hi, part, n_head, n_kv, n_split); break;
        default: throw std::runtime_error("attn_partials: head_dim " + std::to_string(hd) + " is not instantiated");
    }
    check("attn_partials");
}

void attn_combine_quant(const void* part, float* out, void* xq, int rows, int n_head, int hd, int n_split,
                        cudaStream_t s) {
    if (rows <= 0) return;
    if (n_split > 64 || hd > 1024 || hd % 32) throw std::runtime_error("attn_combine_quant: n_split <= 64, hd % 32 == 0");
    attn_combine_quant_kernel<<<dim3(n_head, rows), hd, 0, s>>>(static_cast<const float*>(part), out,
                                                               static_cast<Q81*>(xq), n_head, n_split, hd);
    check("attn_combine_quant");
}

void argmax_rows(const float* x, int64_t n, int rows, int32_t* ids, cudaStream_t s) {
    if (rows <= 0) return;
    argmax_kernel<<<rows, 1024, 0, s>>>(x, n, ids);
    check("argmax_rows");
}

}  // namespace strata::gemma::k
