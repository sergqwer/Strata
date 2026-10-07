// the current production kernel (vision.cu vit_fa2_kernel), copied verbatim except: output type templated
// (bf16 = production; float = to see the kernel's own error without the final bf16 rounding).
#pragma once
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ void st_out(bf16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void st_out(float* p, float v) { *p = v; }

template <int HDP, typename OT>
__global__ void __launch_bounds__(128) cur_fa2_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                                      const __half* __restrict__ V, OT* __restrict__ O, int n, int H,
                                                      int hd) {
    constexpr int BK = 64, LH = HDP + 8, LV = BK + 8, VEC = HDP / 8, NT = BK / 8, DT = HDP / 8, KK = HDP / 16;
    __shared__ __align__(16) __half Ks[BK * LH];
    __shared__ __align__(16) __half Vt[HDP * LV];
    const int h = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int r0 = blockIdx.x * 64 + warp * 16;
    {
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
            OT* out = O + (size_t) ra * H * hd + (size_t) h * hd + col;
            st_out(out, o[d][0] * i0);
            st_out(out + 1, o[d][1] * i0);
        }
        if (rb < n) {
            OT* out = O + (size_t) rb * H * hd + (size_t) h * hd + col;
            st_out(out, o[d][2] * i1);
            st_out(out + 1, o[d][3] * i1);
        }
    }
}
