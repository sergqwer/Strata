// src/gemma/mmq_magic.cuh - llama.cpp's MMQ tile (mmq.cuh: mul_mat_q_process_tile + vec_dot_q8_0_q8_1_mma, Ampere
// path; ggml, MIT) with one change: the int32 dot product of each 32-value block reaches the float epilogue without a
// conversion instruction.  Included by mmq.cu only.
//
// llama.cpp computes  sum += float(c) * dA * dB  per output value and block, float(c) by I2FP.  On the A4000 the
// tensor cores and the float epilogue share the issue / register bandwidth, and the conversion is the expensive op
// (measured, IMMA m16n8k32 + epilogue, registers only: I2FP + FMUL + FFMA 51 TOPS, FADD + FMUL + FFMA 75 TOPS,
// IMMA alone 115).  Here the accumulator of the IMMA starts at 0x4B400000 instead of 0: the float with the bits
// c + 0x4B400000 is exactly 1.5 * 2^23 + c (|c| <= 32 * 128 * 128 < 2^22), so one FADD of -1.5 * 2^23 gives float(c)
// exactly - the same value I2FP gives - and the FMUL / FFMA that follow are llama.cpp's, in the same order.  Every
// tile, load and k step is llama.cpp's own code, so the results are bit-identical; SEG adds llama.cpp's stream-k split
// (two partial sums of one tile meeting as its fixup kernel adds them) for the dense products that take it.
#pragma once

namespace strata::gemma::mmq::magic {

using namespace ggml_cuda_mma;

__device__ __forceinline__ float c_float(int d) { return __fsub_rn(__int_as_float(d), 12582912.0f); }

// vec_dot_q8_0_q8_1_mma, the NVIDIA (non-AMD) branch, letter for letter but for C's start and c_float
template <ggml_type type, int J, bool fallback, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot(const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum,
                                               const int k00) {
    typedef tile<16, 8, int> tile_A;
    typedef tile<8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx = rows_per_warp / tile_C::I;   // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int* x_qs = (const int*) x;
    const float* x_df = (const float*) x_qs + 2 * MMQ_TILE_NE_K;
    const int* y_qs = (const int*) y + 4;
    const float* y_df = (const float*) y;
    const half2* y_ds = (const half2*) y;

    tile_A A[ntx][MMQ_TILE_NE_K / QI8_0];
    float dA[ntx][tile_C::ne / 2][MMQ_TILE_NE_K / QI8_0];

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;
            load_ldmatrix(A[n][k01 / QI8_0], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
        }
#pragma unroll
        for (int l = 0; l < tile_C::ne / 2; ++l) {
            const int i = i0 + n * tile_A::I + tile_C::get_i(2 * l);
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;
                dA[n][l][k01 / QI8_0] = x_df[i * sram_stride + k0 / QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne / 2];

            load_generic(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);   // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne / 2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
                }
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) C.x[l] = 0x4B400000;
                mma(C, A[n][k01 / QI8_0], B);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    float& s = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
                    s = __fmaf_rn(__fmul_rn(c_float(C.x[l]), dA[n][l / 2][k01 / QI8_0]), dB[l % 2], s);
                }
            }
        }
    }
}

template <ggml_type type> __host__ __device__ constexpr mmq_q8_1_ds_layout ds_layout_of() {
    return type == GGML_TYPE_Q4_0 ? MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4;
}

// mul_mat_q_process_tile (fixup = false) with the vec_dot above; SEG: at k block `seg` (a multiple of the
// 256-value iteration) the partial so far is set aside and added back as llama.cpp's stream-k fixup adds it:
// dst = partial[seg, end) + (0 + partial[0, seg))
template <ggml_type type, int J, bool fallback, bool SEG>
static __device__ __forceinline__ void process_tile(const char* __restrict__ x, const int offset_x, const int* __restrict__ y,
                                                    const int* __restrict__ ids_dst, float* __restrict__ dst,
                                                    const int stride_row_x, const int ncols_y, const int stride_col_dst,
                                                    const int tile_x_max_i, const int tile_y_max_j, const int kb0_stop,
                                                    const int seg) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, fallback>();

    extern __shared__ int data_mul_mat_q[];
    int* tile_y = data_mul_mat_q + J;
    int* tile_x = tile_y + GGML_PAD(J * MMQ_TILE_Y_K, nwarps * warp_size);

    constexpr int ne_block = QK8_1_MMQ;
    constexpr int ITER_K = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[J * I / (nwarps * warp_size)] = {0.0f};
    float keep[SEG ? J * I / (nwarps * warp_size) : 1];

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = 0; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        if constexpr (SEG) {
            if (kb0 == seg) {
#pragma unroll
                for (int l = 0; l < J * I / (nwarps * warp_size); ++l) {
                    keep[l] = sum[l];
                    sum[l] = 0.0f;
                }
            }
        }
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int* by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot<type, J, fallback, ds_layout_of<type>()>(tile_x, tile_y, sum, 0);
        __syncthreads();
        {
            const int* by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        vec_dot<type, J, fallback, ds_layout_of<type>()>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }
    if constexpr (SEG) {
        if (seg > 0) {
#pragma unroll
            for (int l = 0; l < J * I / (nwarps * warp_size); ++l) sum[l] = __fadd_rn(sum[l], __fadd_rn(0.0f, keep[l]));
        }
    }
    write_back(sum, ids_dst, dst, nullptr, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

}  // namespace strata::gemma::mmq::magic
