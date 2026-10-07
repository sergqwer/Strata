// include/strata/gemma/moe_w4a16.hpp - the prompt path's expert GEMMs on the fp16 tensor cores, straight from the
// Q4_0 expert tensors (w4a16; src/gemma/moe_w4a16.cu).
//
// The old path rounds every expert input to q8_1 and runs llama.cpp's MMQ (int8 MMA) twice, with the GeGLU and a
// second q8_1 rounding in between, and copies the expert bounds to the host every layer (MMQ's grid needs the largest
// expert). Here the inputs stay fp16 (11-bit significand instead of 8 bits per 32-value block), the Q4_0 weights are
// expanded in registers to fp16 (n - 8 exactly, times the block scale: one rounding), gate and up come out of one
// kernel with the GeGLU applied in its epilogue, and the hidden goes to the down kernel as fp16. The weights are read
// in place (the decode path reads the same tensors): no copy, no repack. Nothing comes back to the host.
//
// Measured (tools/gemma/w4a16, one real layer, skewed routing as in production): 20-40x less error than the old path
// against an fp64 reference, gate_up + down 1.8-1.9x faster at 692-917 prompt tokens and 1.5x at 1787.
//
// Rows are the (token, slot) pairs sorted by expert (k::moe_sort): expert e owns rows [bounds[e], bounds[e+1]); the
// bounds are read on the device. Activations (xs, hidden) are stored "fragment-contiguous" within every 32 values
// (moe_w4a16.cuh: fpos), so one 16-byte load gives an MMA lane its B registers.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::gemma::w4a16 {

/// The shapes this path covers: Q4_0 experts, gate_up [2*ff rows x d] and down [d rows x ff], d and ff multiples of 64,
/// ff a multiple of 176 and d of 352 (the block tiles), 16-byte aligned gate_up rows, 4-byte aligned down rows.
bool supported(int type_gate_up, int type_down, int d, int ff, size_t gu_row_bytes, size_t down_row_bytes,
               const void* gu, const void* down, size_t gu_expert_bytes, size_t down_expert_bytes);

/// Bytes of the fp16 expert inputs (rows x d) and of the fp16 hidden (rows x ff).
inline size_t xs_bytes(int64_t rows, int d) { return (size_t) rows * d * sizeof(__half); }
inline size_t hidden_bytes(int64_t rows, int ff) { return (size_t) rows * ff * sizeof(__half); }

/// xs[inv[t*k + j]] = fp16(g[t]) (32-value blocks reordered) for every token t < tokens and slot j < k: the expert
/// inputs, one copy per chosen expert, in the sorted row order. Values beyond the fp16 range are clamped.
void scatter(const float* g, int d, const int32_t* inv, int tokens, int k, __half* xs, cudaStream_t s);

/// hidden[r] = gelu_tanh(W_gate[e] xs[r]) * (W_up[e] xs[r]) for every sorted row r of expert e (rows = tokens * k,
/// the host knows it without a sync). w: Q4_0, per expert 2*ff rows (gate rows, then up rows) of d values,
/// expert_bytes apart. The hidden is fp16, reordered like xs (values beyond the fp16 range clamped).
void gate_up(const void* w, size_t row_bytes, size_t expert_bytes, int d, int ff, const __half* xs,
             const int32_t* bounds, int n_expert, int64_t rows, __half* hidden, cudaStream_t s);

/// ey[r] = W_down[e] hidden[r] (f32, d values a row) for every sorted row r: k::moe_combine's input.
void down(const void* w, size_t row_bytes, size_t expert_bytes, int ff, int d, const __half* hidden,
          const int32_t* bounds, int n_expert, int64_t rows, float* ey, cudaStream_t s);

}  // namespace strata::gemma::w4a16
