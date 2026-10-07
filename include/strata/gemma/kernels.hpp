// include/strata/gemma/kernels.hpp - the Gemma 4 layer's own kernels (src/gemma/kernels.cu). The matrix products are
// elsewhere: native_mmvq (decode), llama.cpp's MMQ (prompt, src/gemma/mmq.cu) and cuBLAS (attention, router).
//
// Every activation is f32, row-major, one row per token; `rows` is the number of tokens.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::gemma::k {

/// y = x / sqrt(mean(x^2) + eps) * (w ? w : 1) * scale, per row of n.
void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, float scale, cudaStream_t s);

/// out[i] = row tokens[i] of an embedding table of `type` (Q6_K, Q8_0, Q4_0, F16, F32), times scale.
void embed_rows(const void* table, int type, int n, const int32_t* tokens, float* out, int rows, float scale,
                cudaStream_t s);

/// The attention inputs of one layer. Per token: q (n_head heads) -> rms * q_norm -> rope, in place, plus an f16 copy
/// in q16 when q16 is not null; k (n_kv heads) -> rms * k_norm -> rope -> K cache row pos; v -> rms (no weight) ->
/// V cache row pos. v may be null: then v is the raw k (Gemma 4's global layers have no V projection).
/// Caches are f16 [T][n_kv][hd]. RoPE is NEOX over the first n_rot dims, theta = pos * base^(-2i/n_rot) / ff[i].
struct QkvArgs {
    float* q = nullptr;
    const float* k = nullptr;
    const float* v = nullptr;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    const float* freq_factors = nullptr;
    const int32_t* pos = nullptr;
    __half* q16 = nullptr;
    __half* kc = nullptr;
    __half* vc = nullptr;
    int rows = 0, n_head = 0, n_kv = 0, hd = 0, n_rot = 0;
    float base = 10000.f, eps = 1e-6f;
};
void qkv_prep(const QkvArgs& a, cudaStream_t s);

/// Attention for a few query tokens (decode, verify windows): out[t][h] = softmax_j(q[t][h] . K[j]) V[j] over the
/// keys j in [lo[t], hi[t]] (inclusive), head h reading kv head h / (n_head / n_kv). Scale 1.0 (Gemma 4).
/// `scratch` needs attn_decode_scratch_bytes(rows, n_head, hd, t_max) bytes.
size_t attn_decode_scratch_bytes(int rows, int n_head, int hd, int t_max);
void attn_decode(const float* q, const __half* kc, const __half* vc, const int32_t* lo, const int32_t* hi, float* out,
                 int rows, int n_head, int n_kv, int hd, int t_max, void* scratch, cudaStream_t s);

/// The prompt's attention probabilities: S [n_head][rows][t] f32 scores -> P f16, softmax over j in [lo[i], hi[i]]
/// of row i, zero elsewhere.
void mask_softmax(const float* S, __half* P, const int32_t* lo, const int32_t* hi, int n_head, int rows, int t,
                  int ld, cudaStream_t s);

/// out = x + rms(y) * w   (the post-attention residual).
void add_rms(const float* x, const float* y, const float* w, float* out, int n, int rows, float eps, cudaStream_t s);

/// The three normalizations of the attention output that feed the FFN, in one pass:
/// f = rms(x) * w_f (dense MLP), g = rms(x) * w_g (experts), r = rms(x) / sqrt(n) * w_r (router). g, r may be null.
void ffn_norms(const float* x, const float* w_f, const float* w_g, const float* w_r, float* f, float* g, float* r,
               int n, int rows, float eps, cudaStream_t s);

/// Router: per row, softmax over n_expert logits, the top k, their probabilities renormalized to sum 1.
void router_topk(const float* logits, int n_expert, int k, int rows, int32_t* ids, float* w, cudaStream_t s);

/// h[r][i] = gelu_tanh(gate[r][i]) * up[r][i], i < n_ff; gate row r at gate + r * ld, up row r at up + r * ld.
void geglu(const float* gate, const float* up, float* h, int64_t rows, int n_ff, int64_t ld, cudaStream_t s);

/// MoE combine: out[t] = sum_k w[t][k] * scale[ids[t][k]] * y[row(t, k)], row = inv ? inv[t * k + k'] : t * k + k'.
/// scale may be null (1).
void moe_combine(const float* y, const int32_t* inv, const int32_t* ids, const float* w, const float* scale,
                 float* out, int n, int rows, int k, cudaStream_t s);

/// The FFN's end and the layer output: t = rms(mlp) * pn1 + rms(moe) * pn2 (dense layer: t = mlp, moe null),
/// x_out = (x1 + rms(t) * pfn) * out_scale.
void ffn_post(const float* x1, const float* mlp, const float* moe, const float* pn1, const float* pn2, const float* pfn,
              float out_scale, float* x_out, int n, int rows, float eps, cudaStream_t s);

/// logits = cap * tanh(logits / cap) in place (cap > 0).
void softcap(float* logits, int64_t n, float cap, cudaStream_t s);

/// MoE prompt path: sort the (token, slot) pairs by expert. From ids [rows][k]: bounds [n_expert + 1] (expert e's
/// pairs are sorted rows [bounds[e], bounds[e+1])), src [rows * k] (the token of sorted row r) and inv [rows * k]
/// (the sorted row of pair t * k + j). `counts` is n_expert ints of scratch.
void moe_sort(const int32_t* ids, int rows, int k, int n_expert, int32_t* bounds, int32_t* src, int32_t* inv,
              int32_t* counts, cudaStream_t s);

/// Per query row i of a prompt chunk: the key range its attention reads. Causal: [lo, pos]; sliding layers
/// lo = pos - n_swa + 1; a token inside an image span [b, e) on a sliding layer also sees the whole span (hi = e - 1),
/// llama.cpp's non-causal image batch (LLAMA_NON_CAUSAL_TYPE_SWA_ONLY). spans: n_spans pairs of positions.
void key_ranges(const int32_t* pos, int rows, int n_swa, bool swa, const int32_t* spans, int n_spans, int32_t* lo,
                int32_t* hi, cudaStream_t s);

/// iota: dst[i] = i.
void iota(int32_t* dst, int n, cudaStream_t s);

/// moe-glue (prompt path), each bitwise what the old calls give (tools/gemma/glue); STRATA_OLD_GLUE=1: the old calls.
/// moe_sort with the expert scan in parallel (the old one was one thread walking the experts).
void moe_sort2(const int32_t* ids, int rows, int k, int n_expert, int32_t* bounds, int32_t* src, int32_t* inv,
               int32_t* counts, cudaStream_t s);
/// moe_combine then ffn_post in one pass for k = 8 (moe: the combined rows' buffer, used only by the fallback).
void moe_combine_post(const float* y, const int32_t* inv, const int32_t* ids, const float* w, const float* scale,
                      float* moe, const float* x1, const float* mlp, const float* pn1, const float* pn2, const float* pfn,
                      float out_scale, float* x_out, int n, int rows, int k, float eps, cudaStream_t s);
/// add_rms (x1 = x + rms(y) * w_pa) then ffn_norms(x1) in one pass.
void add_rms_ffn_norms(const float* x, const float* y, const float* w_pa, float* x1, const float* w_f, const float* w_g,
                       const float* w_r, float* f, float* g, float* r, int n, int rows, float eps, cudaStream_t s);

}  // namespace strata::gemma::k

// ---- the decode path's fused kernels (rows <= a few; one block of 1024 threads per row). q8_1 outputs are
// native_mmvq's activation format: per 32 values {half2 (d, sum), int8 q[32]}, row r at xq + r * n / 32 blocks.
namespace strata::gemma::k {

/// xq = q8_1(rms(x) * w)
void norm_quant(const float* x, const float* w, void* xq, int n, int rows, float eps, cudaStream_t s);

/// x1 = x + rms(y) * w_pa; then f = rms(x1) * w_f -> xqf, and for MoE layers g = rms(x1) * w_g -> xqg and
/// r = rms(x1) / sqrt(n) * w_r (f32, the router input).
void post_attn_fused(const float* x, const float* y, const float* w_pa, float* x1, const float* w_f, void* xqf,
                     const float* w_g, void* xqg, const float* w_r, float* r, int n, int rows, float eps,
                     cudaStream_t s);

/// xq = q8_1(gelu_tanh(gate) * up), row r of gate / up at + r * ld (n_ff values).
void geglu_quant(const float* gate, const float* up, int64_t ld, void* xq, int n_ff, int rows, cudaStream_t s);

/// The layer's end for a few rows: moe = sum_k w * scale[id] * ey[t * k + j] (when ey is not null), then ffn_post,
/// x_out = (x1 + rms(t) * pfn) * out_scale; and xq = q8_1(rms(x_out) * w_next) for the next layer (or the head).
void moe_post_fused(const float* ey, const int32_t* ids, const float* ew, const float* scale, int k, const float* mlp,
                    const float* x1, const float* pn1, const float* pn2, const float* pfn, float out_scale, float* x_out,
                    const float* w_next, void* xq, int n, int rows, float eps, cudaStream_t s, float* h_out = nullptr);

/// MTP: lo = max(0, pos - n_swa + 1) (sliding) or 0, hi = pos - 1 - the target cache before position pos.
void mtp_ranges(const int32_t* pos, int n_swa, int32_t* lo_swa, int32_t* hi_swa, int32_t* lo_g, int32_t* hi_g,
                cudaStream_t s);
/// out = concat(row tokens[0] of `table` * scale, h)  (n1 + n2 floats; the MTP input)
void embed_concat(const void* table, int type, int n1, const int32_t* token, float scale, const float* h, int n2,
                  float* out, cudaStream_t s);

/// logits[r][e] = W[e] . r[r]  (W f32 [n_expert][n])
void router_gemv(const float* W, const float* r, float* logits, int n_expert, int n, int rows, cudaStream_t s);

/// The flash-decoding combine, also writing q8_1 of the output for the o projection (xq may be null).
void attn_combine_quant(const void* part, float* out, void* xq, int rows, int n_head, int hd, int n_split,
                        cudaStream_t s);
/// attn_decode's first half only (the split partials), for attn_combine_quant.
void attn_partials(const float* q, const __half* kc, const __half* vc, const int32_t* lo, const int32_t* hi,
                   void* scratch, int rows, int n_head, int n_kv, int hd, int n_split, cudaStream_t s);

/// moe-glue: STRATA_OLD_GLUE=1 at startup - every glue path below falls back to the old kernels (read once).
bool glue_off();
/// moe-glue: post_attn_fused, moe_post_fused (k = 8), router_gemv and router_topk with bitwise the same results
/// (tools/gemma/glue): loads issued up front, q8_1 from registers, router rows spread over the SMs, the top-k picks
/// by warp reductions. n <= 3072 (else, and with glue_off(), the old kernels run).
void post_attn_fused2(const float* x, const float* y, const float* w_pa, float* x1, const float* w_f, void* xqf,
                      const float* w_g, void* xqg, const float* w_r, float* r, int n, int rows, float eps,
                      cudaStream_t s);
void moe_post_fused2(const float* ey, const int32_t* ids, const float* ew, const float* scale, int k, const float* mlp,
                     const float* x1, const float* pn1, const float* pn2, const float* pfn, float out_scale, float* x_out,
                     const float* w_next, void* xq, int n, int rows, float eps, cudaStream_t s, float* h_out = nullptr);
void router_gemv2(const float* W, const float* r, float* logits, int n_expert, int n, int rows, cudaStream_t s);
void router_topk2(const float* logits, int n_expert, int k, int rows, int32_t* ids, float* w, cudaStream_t s);
/// moe-glue: attn_partials with each warp's next keys' K / V rows loaded ahead; bitwise the same partials.
void attn_partials2(const float* q, const __half* kc, const __half* vc, const int32_t* lo, const int32_t* hi,
                    void* scratch, int rows, int n_head, int n_kv, int hd, int n_split, cudaStream_t s);
/// moe-glue: attn_combine_quant with the split statistics loaded in parallel; bitwise the same (n_split <= 32).
void attn_combine_quant2(const void* part, float* out, void* xq, int rows, int n_head, int hd, int n_split,
                         cudaStream_t s);
/// The harness: keys loaded ahead per warp (1 = default, 2, 3, 4).
void set_attn_prefetch(int pf);

/// ids[r] = argmax of row r (n values)
void argmax_rows(const float* x, int64_t n, int rows, int32_t* ids, cudaStream_t s);
/// dst row r = src row ids[r] (row_bytes each), r in [0, n): a quantized matrix's chosen rows back to back.
void gather_rows(const void* src, size_t row_bytes, const int32_t* ids, int n, void* dst, cudaStream_t s);

}  // namespace strata::gemma::k
