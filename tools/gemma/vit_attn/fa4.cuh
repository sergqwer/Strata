// fa4: fa3 + software pipelining inside a warp: the Q K^T of tile t + 1 is issued in the same basic block as the
// softmax of tile t, so the tensor pipe has work while the warp's ALU / MUFU part runs. The partial last tile is
// peeled (masking compiled only into its step).
#pragma once
#include <type_traits>
namespace xp {

template <int NW, int MT, int BK, int ST, int MINB, typename OT>
__global__ void __launch_bounds__(NW * 32, MINB) fa4_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                                            const __half* __restrict__ V, OT* __restrict__ O, int n,
                                                            int H) {
    static_assert(ST >= 3, "a tile's K is read one step before its V");
    constexpr int HD = 72, HDP = 80, LS = HD, CH = HD / 8, NT = BK / 8, KJ = BK / 16, QB = NW * 16 * MT;
    constexpr int TH = NW * 32, TILE = BK * LS;
    constexpr float L2E = 1.4426950408889634f;
    extern __shared__ __align__(128) __half smem[];   // [ST][K | V][BK][LS]
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
    {
        const size_t z = blockIdx.z;
        Q += z * H * n * HDP;
        K += z * H * n * HDP;
        V += z * H * n * HDP;
        O += z * n * H * HD;
    }
    const __half* Qh = Q + (size_t) h * n * HDP;
    const __half* Kh = K + (size_t) h * n * HDP;
    const __half* Vh = V + (size_t) h * n * HDP;
    const int q0 = blockIdx.x * QB + warp * 16 * MT;

    auto load_tile = [&](int tile) {
        __half* ks = smem + (tile % ST) * 2 * TILE;
        __half* vs = ks + TILE;
        const int k0 = tile * BK;
#pragma unroll
        for (int i0 = 0; i0 < BK * CH; i0 += TH) {
            const int i = i0 + tid;
            if (BK * CH % TH == 0 || i < BK * CH) {
                const int r = i / CH, c = i % CH;
                const bool ok = k0 + r < n;
                const size_t off = (size_t) (ok ? k0 + r : 0) * HDP + c * 8;
                cp_async16(ks + i * 8, Kh + off, ok ? 16 : 0);
                cp_async16(vs + i * 8, Vh + off, ok ? 16 : 0);
            }
        }
    };
    const int ntiles = (n + BK - 1) / BK;
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < ntiles) load_tile(s);
        cp_async_commit();
    }

    uint32_t qa[MT][4][4], qb[MT][2];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int ra = q0 + m * 16 + g, rb = ra + 8;
        const __half* pa = Qh + (size_t) ra * HDP + 2 * t;
        const __half* pb = Qh + (size_t) rb * HDP + 2 * t;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            qa[m][kk][0] = ra < n ? *reinterpret_cast<const uint32_t*>(pa + kk * 16) : 0u;
            qa[m][kk][1] = rb < n ? *reinterpret_cast<const uint32_t*>(pb + kk * 16) : 0u;
            qa[m][kk][2] = ra < n ? *reinterpret_cast<const uint32_t*>(pa + kk * 16 + 8) : 0u;
            qa[m][kk][3] = rb < n ? *reinterpret_cast<const uint32_t*>(pb + kk * 16 + 8) : 0u;
        }
        qb[m][0] = ra < n ? *reinterpret_cast<const uint32_t*>(pa + 64) : 0u;
        qb[m][1] = rb < n ? *reinterpret_cast<const uint32_t*>(pb + 64) : 0u;
    }
    float o[MT][9][4];
    float mrow[MT][2], lrow[MT][2];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int d = 0; d < 9; ++d) o[m][d][0] = o[m][d][1] = o[m][d][2] = o[m][d][3] = 0.f;
        mrow[m][0] = mrow[m][1] = -INFINITY;
        lrow[m][0] = lrow[m][1] = 0.f;
    }
    const int lr = lane & 7, lj = lane >> 3;

    // S = Q K^T of one tile (masked past n when MASK)
    auto qk = [&](float (&s)[MT][NT][4], int tile, auto mask_c) {
        const __half* ks = smem + (tile % ST) * 2 * TILE;
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) s[m][nt][0] = s[m][nt][1] = s[m][nt][2] = s[m][nt][3] = 0.f;
#pragma unroll
        for (int n4 = 0; n4 < NT / 4; ++n4) {
            uint32_t c8[4];
            ldsm_x4(c8[0], c8[1], c8[2], c8[3], ks + ((n4 * 4 + lj) * 8 + lr) * LS + 64);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int nt = n4 * 4 + j;
                uint32_t b[8];
                ldsm_x4(b[0], b[1], b[2], b[3], ks + (nt * 8 + lr) * LS + lj * 8);
                ldsm_x4(b[4], b[5], b[6], b[7], ks + (nt * 8 + lr) * LS + 32 + lj * 8);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    mma16816b(s[m][nt], qa[m][0], b[0], b[1]);
                    mma16816b(s[m][nt], qa[m][1], b[2], b[3]);
                    mma16816b(s[m][nt], qa[m][2], b[4], b[5]);
                    mma16816b(s[m][nt], qa[m][3], b[6], b[7]);
                    mma1688(s[m][nt], qb[m], c8[j]);
                }
            }
        }
        if constexpr (decltype(mask_c)::value) {
            const int k0 = tile * BK;
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int col = k0 + nt * 8 + 2 * t;
                    if (col >= n) s[m][nt][0] = s[m][nt][2] = -INFINITY;
                    if (col + 1 >= n) s[m][nt][1] = s[m][nt][3] = -INFINITY;
                }
        }
    };
    // online softmax of S (tile `tile`) and O += P V
    auto softmax_pv = [&](float (&s)[MT][NT][4], int tile) {
        const __half* vs = smem + (tile % ST) * 2 * TILE + TILE;
        uint32_t pa[MT][KJ][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float mx0 = mrow[m][0], mx1 = mrow[m][1];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                mx0 = fmaxf(mx0, fmaxf(s[m][nt][0], s[m][nt][1]));
                mx1 = fmaxf(mx1, fmaxf(s[m][nt][2], s[m][nt][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float c0 = ex2((mrow[m][0] - mx0) * L2E), c1 = ex2((mrow[m][1] - mx1) * L2E);
            mrow[m][0] = mx0;
            mrow[m][1] = mx1;
            float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const float p00 = ex2((s[m][nt][0] - mx0) * L2E), p01 = ex2((s[m][nt][1] - mx0) * L2E);
                const float p10 = ex2((s[m][nt][2] - mx1) * L2E), p11 = ex2((s[m][nt][3] - mx1) * L2E);
                sum0 += p00 + p01;
                sum1 += p10 + p11;
                pa[m][nt / 2][(nt & 1) * 2 + 0] = pack_h2b(p00, p01);
                pa[m][nt / 2][(nt & 1) * 2 + 1] = pack_h2b(p10, p11);
            }
            lrow[m][0] = lrow[m][0] * c0 + sum0;
            lrow[m][1] = lrow[m][1] * c1 + sum1;
#pragma unroll
            for (int d = 0; d < 9; ++d) {
                o[m][d][0] *= c0;
                o[m][d][1] *= c0;
                o[m][d][2] *= c1;
                o[m][d][3] *= c1;
            }
        }
#pragma unroll
        for (int j = 0; j < KJ; ++j) {
            const __half* vrow = vs + (j * 16 + lr + (lj & 1) * 8) * LS;
#pragma unroll
            for (int dp = 0; dp < 4; ++dp) {
                uint32_t v0, v1, v2, v3;
                ldsm_x4_t(v0, v1, v2, v3, vrow + (dp * 2 + (lj >> 1)) * 8);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    mma16816b(o[m][2 * dp], pa[m][j], v0, v1);
                    mma16816b(o[m][2 * dp + 1], pa[m][j], v2, v3);
                }
            }
            uint32_t v0, v1;
            ldsm_x2_t(v0, v1, vrow + 64);
#pragma unroll
            for (int m = 0; m < MT; ++m) mma16816b(o[m][8], pa[m][j], v0, v1);
        }
    };

    using T_ = std::true_type;
    using F_ = std::false_type;
    float sa[MT][NT][4], sb[MT][NT][4];
    cp_async_wait<ST - 2>();
    __syncthreads();
    if (ntiles == 1) qk(sa, 0, T_{});
    else qk(sa, 0, F_{});
    // step it: tile it + 1's K is complete; tile it + ST - 1 is issued; S(it + 1) and softmax / PV of it
    auto step = [&](float (&scur)[MT][NT][4], float (&snext)[MT][NT][4], int it, auto mask_next, auto has_next) {
        cp_async_wait<ST - 3>();
        __syncthreads();
        if (it + ST - 1 < ntiles) load_tile(it + ST - 1);
        cp_async_commit();
        if constexpr (decltype(has_next)::value) qk(snext, it + 1, mask_next);
        softmax_pv(scur, it);
    };
    int it = 0;
    for (; it + 2 < ntiles; it += 2) {   // two steps an iteration so sa / sb swap roles statically
        step(sa, sb, it, F_{}, T_{});
        if (it + 3 < ntiles) step(sb, sa, it + 1, F_{}, T_{});
        else { step(sb, sa, it + 1, T_{}, T_{}); it += 2; goto tail_a; }
    }
    // here it + 2 >= ntiles: tile it is in sa
    if (it + 1 < ntiles) {
        step(sa, sb, it, T_{}, T_{});
        step(sb, sa, it + 1, F_{}, F_{});
    } else {
        step(sa, sb, it, F_{}, F_{});
    }
    goto done;
tail_a:   // tile it (the last) is in sa
    step(sa, sb, it, F_{}, F_{});
done:
    cp_async_wait<0>();
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        float l0 = lrow[m][0], l1 = lrow[m][1];
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        const float i0 = 1.f / l0, i1 = 1.f / l1;
        const int ra = q0 + m * 16 + g, rb = ra + 8;
#pragma unroll
        for (int d = 0; d < 9; ++d) {
            const int col = d * 8 + 2 * t;
            if (ra < n) st_out2(O + (size_t) ra * H * HD + (size_t) h * HD + col, o[m][d][0] * i0, o[m][d][1] * i0);
            if (rb < n) st_out2(O + (size_t) rb * H * HD + (size_t) h * HD + col, o[m][d][2] * i1, o[m][d][3] * i1);
        }
    }
}

}  // namespace xp
