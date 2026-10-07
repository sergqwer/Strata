// include/strata/gemma/decode.hpp - one layer of a decode step (Engine::layer_small), as a free function so the
// harness (tools/gemma/glue) runs exactly the engine's code on its own buffers.
//
// For one row (a decode step; the speculative verify windows have more) the layer runs the moe-glue paths, each
// bitwise equal to the old kernel sequence (tools/gemma/glue checks it on the model's weights):
//   - q / k / v and the dense gate / up as one launch each (native_mmvq_multi_w);
//   - the expert products and the K = 2112 down projection in the warp-per-row layout (native_mmvq_vt);
//   - post_attn_fused2, router_gemv2 + router_topk2, moe_post_fused2;
//   - the dense MLP on a second stream while the router and the experts run on the first (they read the same
//     post-attention rows; the layer's end waits for both), so the latency-bound router kernels overlap the
//     bandwidth-bound dense products.
// STRATA_OLD_GLUE=1 at startup: the old kernel sequence, one stream.
#pragma once

#include "strata/gemma/model.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::gemma {

/// The decode path's buffers (Engine's arena; the harness has its own). Rows are n_embd (x, x1, y, mlp, r),
/// q / att: n_head * head_dim, k / v: n_kv * head_dim, gate / up: n_ff, rlog: n_expert, ew / eid: k, gu: k * 2 n_ff_exp,
/// ey: k * n_embd; the xq* buffers hold q8_1 rows of the matching widths.
struct DecodeBufs {
    float *x = nullptr, *x1 = nullptr, *y = nullptr, *mlp = nullptr, *q = nullptr, *att = nullptr, *k = nullptr,
          *v = nullptr, *gate = nullptr, *up = nullptr, *r = nullptr, *rlog = nullptr, *ew = nullptr, *gu = nullptr,
          *ey = nullptr;
    int32_t* eid = nullptr;
    void *xq = nullptr, *xqf = nullptr, *xqg = nullptr, *xqh = nullptr, *xqe = nullptr, *att_scratch = nullptr,
         *groups = nullptr;
    const int32_t *pos = nullptr, *lo = nullptr, *hi = nullptr;   // lo / hi: [0, max_batch) sliding, then global
    int max_batch = 0, n_split = 0;
};

/// The FFN fork of a one-row layer: the dense MLP on `side`. Events created with cudaEventDisableTiming; they are
/// re-recorded every layer (in a stream capture each record / wait pair is its own dependency).
struct DecodeFork {
    cudaStream_t side = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
};

/// One layer for n <= 8 rows. kc / vc: the layer's caches; freq: rope frequency factors (global layers) or null;
/// w_next: the next layer's attn_norm (output_norm after the last layer); h_out: the last layer's normed rows (null
/// otherwise). fork: the second stream, or null (one stream).
void decode_layer(const Config& c, const Layer& L, const float* freq, __half* kc, __half* vc, const float* w_next,
                  float* h_out, const DecodeBufs& b, int n, cudaStream_t s, const DecodeFork* fork);

}  // namespace strata::gemma
