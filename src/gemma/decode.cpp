// src/gemma/decode.cpp - see include/strata/gemma/decode.hpp.
#include "strata/gemma/decode.hpp"

#include "strata/gemma/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::gemma {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

k::QkvArgs qkv_args(const Config& c, const Layer& L, const float* freq, __half* kc, __half* vc, const DecodeBufs& b,
                    int n) {
    k::QkvArgs a;
    a.q = b.q;
    a.k = b.k;
    a.v = L.wv ? b.v : nullptr;
    a.q_norm = L.q_norm.f32();
    a.k_norm = L.k_norm.f32();
    a.freq_factors = freq;
    a.pos = b.pos;
    a.kc = kc;
    a.vc = vc;
    a.rows = n;
    a.n_head = L.n_head;
    a.n_kv = L.n_head_kv;
    a.hd = L.head_dim;
    a.n_rot = L.n_rot;
    a.base = L.rope_base;
    a.eps = c.eps;
    return a;
}

// the baseline layer (gemma 8c7651c's Engine::layer_small): verify windows (n > 1) and STRATA_OLD_GLUE=1
void layer_old(const Config& c, const Layer& L, const float* freq, __half* kc, __half* vc, const float* w_next,
               float* h_out, const DecodeBufs& b, int n, cudaStream_t s) {
    using namespace strata::kernels;
    const int D = c.n_embd, qd = L.n_head * L.head_dim, kvd = L.n_head_kv * L.head_dim;
    const int K = c.n_expert_used, FE = c.n_ff_exp;

    native_mmvq(L.wq.type, L.wq.d, b.xq, b.q, D, qd, n, s);
    native_mmvq(L.wk.type, L.wk.d, b.xq, b.k, D, kvd, n, s);
    if (L.wv) native_mmvq(L.wv.type, L.wv.d, b.xq, b.v, D, kvd, n, s);
    k::qkv_prep(qkv_args(c, L, freq, kc, vc, b, n), s);
    const int32_t* lo = L.swa ? b.lo : b.lo + b.max_batch;
    const int32_t* hi = L.swa ? b.hi : b.hi + b.max_batch;
    k::attn_partials(b.q, kc, vc, lo, hi, b.att_scratch, n, L.n_head, L.n_head_kv, L.head_dim, b.n_split, s);
    k::attn_combine_quant(b.att_scratch, b.att, b.xq, n, L.n_head, L.head_dim, b.n_split, s);
    native_mmvq(L.wo.type, L.wo.d, b.xq, b.y, qd, D, n, s);

    const bool moe = L.moe();
    k::post_attn_fused(b.x, b.y, L.post_attn_norm.f32(), b.x1, L.ffn_norm.f32(), b.xqf, moe ? L.pre_ffw_norm_2.f32() : nullptr,
                       moe ? b.xqg : nullptr, moe ? L.router_scale.f32() : nullptr, b.r, D, n, c.eps, s);
    native_mmvq(L.ffn_gate.type, L.ffn_gate.d, b.xqf, b.gate, D, c.n_ff, n, s);
    native_mmvq(L.ffn_up.type, L.ffn_up.d, b.xqf, b.up, D, c.n_ff, n, s);
    k::geglu_quant(b.gate, b.up, c.n_ff, b.xqh, c.n_ff, n, s);
    native_mmvq(L.ffn_down.type, L.ffn_down.d, b.xqh, b.mlp, c.n_ff, D, n, s);
    if (moe) {
        k::router_gemv(L.router.f32(), b.r, b.rlog, c.n_expert, D, n, s);
        k::router_topk(b.rlog, c.n_expert, K, n, b.eid, b.ew, s);
        static const bool no_group = std::getenv("STRATA_NO_GROUP") != nullptr;
        if (n == 1 || no_group) {   // one token: the 4-warps-per-row id kernel is faster
            native_mmvq_id(L.gate_up_exps.type, L.gate_up_exps.d, L.gate_up_exps.nb2, b.eid, n * K, b.xqg, K, b.gu, D, 2 * FE, s);
            k::geglu_quant(b.gu, b.gu + FE, 2 * FE, b.xqe, FE, n * K, s);
            native_mmvq_id(L.down_exps.type, L.down_exps.d, L.down_exps.nb2, b.eid, n * K, b.xqe, 1, b.ey, FE, D, s);
        } else {   // a verify window: an expert several tokens chose is read once
            native_mmvq_group_ids(b.eid, n * K, b.groups, s);
            native_mmvq_grouped(L.gate_up_exps.type, L.gate_up_exps.d, L.gate_up_exps.nb2, b.groups, n * K, b.xqg, K, b.gu, D,
                                2 * FE, s);
            k::geglu_quant(b.gu, b.gu + FE, 2 * FE, b.xqe, FE, n * K, s);
            native_mmvq_grouped(L.down_exps.type, L.down_exps.d, L.down_exps.nb2, b.groups, n * K, b.xqe, 1, b.ey, FE, D, s);
        }
    }
    k::moe_post_fused(moe ? b.ey : nullptr, b.eid, b.ew, L.down_exps_scale ? L.down_exps_scale.f32() : nullptr, K, b.mlp, b.x1,
                      moe ? L.post_ffw_norm_1.f32() : nullptr, moe ? L.post_ffw_norm_2.f32() : nullptr,
                      L.post_ffw_norm.f32(), L.out_scale, b.x, w_next, b.xq, D, n, c.eps, s, h_out);
}

}  // namespace

void decode_layer(const Config& c, const Layer& L, const float* freq, __half* kc, __half* vc, const float* w_next,
                  float* h_out, const DecodeBufs& b, int n, cudaStream_t s, const DecodeFork* fork) {
    if (n != 1 || k::glue_off()) {
        layer_old(c, L, freq, kc, vc, w_next, h_out, b, n, s);
        return;
    }
    using namespace strata::kernels;
    const int D = c.n_embd, qd = L.n_head * L.head_dim, kvd = L.n_head_kv * L.head_dim;
    const int K = c.n_expert_used, FE = c.n_ff_exp;

    // q / k / v: one launch (each matrix's rows exactly as its own native_mmvq call computes them)
    if (L.wk.type == L.wq.type && (!L.wv || L.wv.type == L.wq.type)) {
        const void* w[3] = {L.wq.d, L.wk.d, L.wv.d};
        float* y[3] = {b.q, b.k, b.v};
        const int no[3] = {qd, kvd, kvd};
        native_mmvq_multi_w(L.wq.type, L.wv ? 3 : 2, w, y, no, b.xq, D, s);
    } else {
        native_mmvq(L.wq.type, L.wq.d, b.xq, b.q, D, qd, 1, s);
        native_mmvq(L.wk.type, L.wk.d, b.xq, b.k, D, kvd, 1, s);
        if (L.wv) native_mmvq(L.wv.type, L.wv.d, b.xq, b.v, D, kvd, 1, s);
    }
    k::qkv_prep(qkv_args(c, L, freq, kc, vc, b, 1), s);
    const int32_t* lo = L.swa ? b.lo : b.lo + b.max_batch;
    const int32_t* hi = L.swa ? b.hi : b.hi + b.max_batch;
    k::attn_partials2(b.q, kc, vc, lo, hi, b.att_scratch, 1, L.n_head, L.n_head_kv, L.head_dim, b.n_split, s);
    k::attn_combine_quant2(b.att_scratch, b.att, b.xq, 1, L.n_head, L.head_dim, b.n_split, s);
    native_mmvq(L.wo.type, L.wo.d, b.xq, b.y, qd, D, 1, s);

    const bool moe = L.moe();
    k::post_attn_fused2(b.x, b.y, L.post_attn_norm.f32(), b.x1, L.ffn_norm.f32(), b.xqf, moe ? L.pre_ffw_norm_2.f32() : nullptr,
                        moe ? b.xqg : nullptr, moe ? L.router_scale.f32() : nullptr, b.r, D, 1, c.eps, s);
    // the dense MLP on the side stream while the router and the experts run here
    const bool forked = moe && fork && fork->side;
    cudaStream_t ds = forked ? fork->side : s;
    if (forked) {
        ck(cudaEventRecord(fork->fork, s), "decode fork");
        ck(cudaStreamWaitEvent(ds, fork->fork, 0), "decode fork");
    }
    if (L.ffn_up.type == L.ffn_gate.type) {
        const void* w[2] = {L.ffn_gate.d, L.ffn_up.d};
        float* y[2] = {b.gate, b.up};
        const int no[2] = {c.n_ff, c.n_ff};
        native_mmvq_multi_w(L.ffn_gate.type, 2, w, y, no, b.xqf, D, ds);
    } else {
        native_mmvq(L.ffn_gate.type, L.ffn_gate.d, b.xqf, b.gate, D, c.n_ff, 1, ds);
        native_mmvq(L.ffn_up.type, L.ffn_up.d, b.xqf, b.up, D, c.n_ff, 1, ds);
    }
    k::geglu_quant(b.gate, b.up, c.n_ff, b.xqh, c.n_ff, 1, ds);
    native_mmvq(L.ffn_down.type, L.ffn_down.d, b.xqh, b.mlp, c.n_ff, D, 1, ds);
    if (moe) {
        k::router_gemv2(L.router.f32(), b.r, b.rlog, c.n_expert, D, 1, s);
        k::router_topk2(b.rlog, c.n_expert, K, 1, b.eid, b.ew, s);
        native_mmvq_id(L.gate_up_exps.type, L.gate_up_exps.d, L.gate_up_exps.nb2, b.eid, K, b.xqg, K, b.gu, D, 2 * FE, s);
        k::geglu_quant(b.gu, b.gu + FE, 2 * FE, b.xqe, FE, K, s);
        native_mmvq_id(L.down_exps.type, L.down_exps.d, L.down_exps.nb2, b.eid, K, b.xqe, 1, b.ey, FE, D, s);
    }
    if (forked) {
        ck(cudaEventRecord(fork->join, ds), "decode join");
        ck(cudaStreamWaitEvent(s, fork->join, 0), "decode join");
    }
    k::moe_post_fused2(moe ? b.ey : nullptr, b.eid, b.ew, L.down_exps_scale ? L.down_exps_scale.f32() : nullptr, K, b.mlp, b.x1,
                       moe ? L.post_ffw_norm_1.f32() : nullptr, moe ? L.post_ffw_norm_2.f32() : nullptr,
                       L.post_ffw_norm.f32(), L.out_scale, b.x, w_next, b.xq, D, 1, c.eps, s, h_out);
}

}  // namespace strata::gemma
