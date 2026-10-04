// include/strata/gemma/mmq.hpp - the prompt path's matrix products through llama.cpp's MMQ kernels (ggml-cuda
// mmq.cuh, MIT), adapted from Strata's include/strata/prefill/moe_mmq.hpp: the weights stay quantized, the
// activations are rounded to q8_1 and the products run on int8 tensor cores. A dense projection is a Product with
// one "expert"; the MoE experts are one Product over the stacked expert tensor with the (token, slot) rows sorted by
// expert (k::moe_sort).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::gemma::mmq {

/// This build has the MMQ path (the ggml sources were available to the build).
bool built();
/// MMQ covers this ggml type: Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K.
bool supported(int ggml_type);
/// #420: `supported`, and on every visible GPU llama.cpp's MMQ has a tile for this type and a weight matrix of
/// `w_rows` rows that fits the card's shared memory - the same test its tile choice makes, which aborts the process
/// ("J_best=0") when nothing fits.  false (said once per type) keeps that product on the non-MMQ path.
bool fits(int ggml_type, int64_t w_rows);
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream);

/// One launch over n experts whose weights lie `expert_bytes` apart from `w`: for expert e, the activation rows
/// [bounds[e], bounds[e+1]) of `xq` (bounds on the device, n+1 entries) times its [w_rows, w_cols] matrix into
/// dst rows of the same indices (`ld_dst` floats apart, via `ids`: dst row = ids[row], an identity table works).
/// `total_rows`: the rows of xq; `max_rows`: the most rows one expert has (the launch grid).
struct Product {
    const void* w = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    int64_t total_rows = 0, max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
};

/// The launch context (llama.cpp's MMQ keeps a small scratch pool for its stream-k fixup).  One per prompt path.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);

private:
    void* ctx_ = nullptr;
};

/// dst[i] = i on the device (an identity row table for dense products).
void iota(int32_t* dst, int64_t n, void* stream);

}  // namespace strata::gemma::mmq
