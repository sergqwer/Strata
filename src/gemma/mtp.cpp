// src/gemma/mtp.cpp - see include/strata/gemma/mtp.hpp. The graph is llama.cpp's src/models/gemma4-assistant.cpp:
//
//   xh  = concat(target_tok_embd[token] * sqrt(n_tgt), h)            h: the target's (or the previous step's) hidden
//   x   = pre_projection xh                                           [2 n_tgt -> n_embd]
//   4 x [ q = rope(rms_head(Wq rms(x) attn_norm) q_norm); a = attention over the target cache; x1 = x + rms(Wo a) pan;
//         x = (x1 + rms(ffn(rms(x1) ffn_norm)) post_ffw_norm) * layer_output_scale ]
//   x   = rms(x) output_norm;  logits = token_embd x;  h_next = post_projection x   [n_embd -> n_tgt]
#include "strata/gemma/mtp.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/gemma/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::gemma {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("mtp ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

Mtp::Mtp(const std::string& path, const Model& target, Engine& engine, int max_draft)
    : tgt_(target), eng_(engine), max_draft_(max_draft) {
    GgufFile g(path);
    const MetaValue* arch = g.get("general.architecture");
    if (!arch || arch->s != "gemma4-assistant") throw std::runtime_error("mtp: " + path + " is not a gemma4-assistant GGUF");
    auto u = [&](const std::string& k) -> uint64_t {
        const MetaValue* v = g.get("gemma4-assistant." + k);
        if (!v) throw std::runtime_error("mtp: missing gemma4-assistant." + k);
        return v->type == MetaType::ARRAY ? v->items.at(0).u : v->u;
    };
    auto f = [&](const std::string& k, double d) {
        const MetaValue* v = g.get("gemma4-assistant." + k);
        return v ? v->num() : d;
    };
    const int n_layer = (int) u("block_count");
    n_embd_ = (int) u("embedding_length");
    n_ff_ = (int) u("feed_forward_length");
    n_tgt_ = (int) u("embedding_length_out");
    n_swa_ = (int) u("attention.sliding_window");
    eps_ = (float) f("attention.layer_norm_rms_epsilon", 1e-6);
    if (n_tgt_ != target.cfg.n_embd) throw std::runtime_error("mtp: the assistant is for a target of width " + std::to_string(n_tgt_));
    const MetaValue* pat = g.get("gemma4-assistant.attention.sliding_window_pattern");
    const MetaValue* kvh = g.get("gemma4-assistant.attention.head_count_kv");
    const int hd_g = (int) u("attention.key_length"), hd_s = (int) u("attention.key_length_swa");
    const float base_g = (float) f("rope.freq_base", 1e6), base_s = (float) f("rope.freq_base_swa", 1e4);

    struct P { Tensor* t; const TensorInfo* ti; };
    std::vector<P> pend;
    size_t total = 0;
    auto want = [&](Tensor& t, const std::string& name) {
        const TensorInfo* ti = g.find(name);
        if (!ti) throw std::runtime_error("mtp: missing tensor " + name);
        t.type = (int) ti->type;
        for (size_t i = 0; i < 3; ++i) t.ne[i] = i < ti->shape.size() ? (int64_t) ti->shape[i] : 1;
        t.nb1 = row_bytes(t.type, t.ne[0]);
        t.nb2 = t.nb1 * (size_t) t.ne[1];
        total = (total + 255) / 256 * 256;
        t.d = reinterpret_cast<const void*>(total);
        total += t.nb2 * (size_t) t.ne[2];
        pend.push_back({&t, ti});
    };
    want(tok_embd_, "token_embd.weight");
    want(output_norm_, "output_norm.weight");
    want(pre_proj_, "nextn.pre_projection.weight");
    want(post_proj_, "nextn.post_projection.weight");
    want(rope_freqs_, "rope_freqs.weight");
    n_vocab_ = (int) tok_embd_.ne[1];
    layers_.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        L& l = layers_[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        l.swa = pat && il < (int) pat->items.size() ? pat->items[il].u != 0 : true;
        l.n_head = (int) u("attention.head_count");
        l.n_kv = kvh && kvh->type == MetaType::ARRAY ? (int) kvh->items.at(il).u : (int) u("attention.head_count_kv");
        l.hd = l.swa ? hd_s : hd_g;
        l.base = l.swa ? base_s : base_g;
        want(l.attn_norm, p + "attn_norm.weight");
        want(l.wq, p + "attn_q.weight");
        want(l.q_norm, p + "attn_q_norm.weight");
        want(l.wo, p + "attn_output.weight");
        want(l.post_attn_norm, p + "post_attention_norm.weight");
        want(l.ffn_norm, p + "ffn_norm.weight");
        want(l.ffn_gate, p + "ffn_gate.weight");
        want(l.ffn_up, p + "ffn_up.weight");
        want(l.ffn_down, p + "ffn_down.weight");
        want(l.post_ffw_norm, p + "post_ffw_norm.weight");
        l.out_scale = 1.f;
        if (const TensorInfo* s = g.find(p + "layer_output_scale.weight")) std::memcpy(&l.out_scale, g.tensor_data(*s), 4);
        // the target layer whose cache this layer reads must have the same shape
        const Layer& tl = target.layers[l.swa ? target.cfg.n_layer - 2 : target.cfg.n_layer - 1];
        if (tl.swa != l.swa || tl.head_dim != l.hd || tl.n_head_kv != l.n_kv)
            throw std::runtime_error("mtp: layer " + std::to_string(il) + " does not match the target cache it reads");
    }
    ck(cudaMalloc(&arena_, total), "malloc");
    bytes_ = total;
    for (const P& p : pend) {
        const size_t off = reinterpret_cast<size_t>(p.t->d);
        ck(cudaMemcpy(static_cast<uint8_t*>(arena_) + off, g.tensor_data(*p.ti), p.t->nb2 * (size_t) p.t->ne[2],
                      cudaMemcpyHostToDevice), "upload");
        p.t->d = static_cast<uint8_t*>(arena_) + off;
    }

    // work buffers
    int max_q = 0, max_hd = 0;
    for (const L& l : layers_) {
        max_q = std::max(max_q, l.n_head * l.hd);
        max_hd = std::max(max_hd, l.hd);
    }
    size_t off = 0;
    auto carve = [&](size_t bytes) { off = (off + 255) / 256 * 256; const size_t o = off; off += bytes; return o; };
    const size_t o_tok = carve(4), o_pos = carve(4), o_rng = carve(16), o_dr = carve(4 * (max_draft + 1)),
                 o_xh = carve(8 * n_tgt_), o_x = carve(4 * n_embd_), o_x1 = carve(4 * n_embd_), o_q = carve(4 * max_q),
                 o_att = carve(4 * max_q), o_y = carve(4 * n_embd_), o_g = carve(4 * n_ff_), o_u = carve(4 * n_ff_),
                 o_mlp = carve(4 * n_embd_), o_lg = carve(4 * (size_t) n_vocab_), o_h = carve(4 * n_tgt_),
                 o_xq = carve((size_t) std::max({2 * n_tgt_, max_q, n_ff_}) / 32 * 36), o_xqf = carve(n_embd_ / 32 * 36),
                 o_xqh = carve(n_ff_ / 32 * 36),
                 o_sc = carve(k::attn_decode_scratch_bytes(1, layers_[0].n_head, max_hd, engine.ctx()));
    ck(cudaMalloc(&work_, off), "malloc work");
    bytes_ += off;
    auto at = [&](size_t o) { return static_cast<uint8_t*>(work_) + o; };
    tok_ = reinterpret_cast<int32_t*>(at(o_tok));
    pos_ = reinterpret_cast<int32_t*>(at(o_pos));
    ranges_ = reinterpret_cast<int32_t*>(at(o_rng));
    drafts_ = reinterpret_cast<int32_t*>(at(o_dr));
    xh_ = reinterpret_cast<float*>(at(o_xh));
    x_ = reinterpret_cast<float*>(at(o_x));
    x1_ = reinterpret_cast<float*>(at(o_x1));
    q_ = reinterpret_cast<float*>(at(o_q));
    att_ = reinterpret_cast<float*>(at(o_att));
    y_ = reinterpret_cast<float*>(at(o_y));
    gate_ = reinterpret_cast<float*>(at(o_g));
    up_ = reinterpret_cast<float*>(at(o_u));
    mlp_ = reinterpret_cast<float*>(at(o_mlp));
    logits_ = reinterpret_cast<float*>(at(o_lg));
    h_ = reinterpret_cast<float*>(at(o_h));
    xq_ = at(o_xq);
    xqf_ = at(o_xqf);
    xqh_ = at(o_xqh);
    scratch_ = at(o_sc);
    graphs_.assign(max_draft + 1, nullptr);
}

Mtp::~Mtp() {
    for (void* g : graphs_)
        if (g) cudaGraphExecDestroy((cudaGraphExec_t) g);
    if (work_) cudaFree(work_);
    if (arena_) cudaFree(arena_);
}

void Mtp::step(int i) {
    using namespace strata::kernels;
    const cudaStream_t s = eng_.stream();
    const int32_t* tok = i == 0 ? tok_ : drafts_ + (i - 1);
    k::embed_concat(tgt_.tok_embd.d, tgt_.tok_embd.type, n_tgt_, tok, sqrtf((float) n_tgt_), h_, n_tgt_, xh_, s);
    native_quantize_q8_1(xh_, xq_, 2 * n_tgt_, 1, s);
    native_mmvq(pre_proj_.type, pre_proj_.d, xq_, x_, 2 * n_tgt_, n_embd_, 1, s);
    k::norm_quant(x_, layers_[0].attn_norm.f32(), xq_, n_embd_, 1, eps_, s);
    const int32_t *lo_s = ranges_, *hi_s = ranges_ + 1, *lo_g = ranges_ + 2, *hi_g = ranges_ + 3;
    for (size_t il = 0; il < layers_.size(); ++il) {
        const L& l = layers_[il];
        const int qd = l.n_head * l.hd;
        native_mmvq(l.wq.type, l.wq.d, xq_, q_, n_embd_, qd, 1, s);
        k::QkvArgs a;
        a.q = q_;
        a.q_norm = l.q_norm.f32();
        a.freq_factors = l.swa ? nullptr : rope_freqs_.f32();
        a.pos = pos_;
        a.rows = 1;
        a.n_head = l.n_head;
        a.n_kv = 0;   // query only
        a.hd = l.hd;
        a.n_rot = l.hd;
        a.base = l.base;
        a.eps = eps_;
        k::qkv_prep(a, s);
        const int tl = l.swa ? tgt_.cfg.n_layer - 2 : tgt_.cfg.n_layer - 1;
        k::attn_partials(q_, eng_.k_cache(tl), eng_.v_cache(tl), l.swa ? lo_s : lo_g, l.swa ? hi_s : hi_g, scratch_, 1,
                         l.n_head, l.n_kv, l.hd, eng_.n_split(), s);
        k::attn_combine_quant(scratch_, att_, xq_, 1, l.n_head, l.hd, eng_.n_split(), s);
        native_mmvq(l.wo.type, l.wo.d, xq_, y_, qd, n_embd_, 1, s);
        k::post_attn_fused(x_, y_, l.post_attn_norm.f32(), x1_, l.ffn_norm.f32(), xqf_, nullptr, nullptr, nullptr, nullptr,
                           n_embd_, 1, eps_, s);
        native_mmvq(l.ffn_gate.type, l.ffn_gate.d, xqf_, gate_, n_embd_, n_ff_, 1, s);
        native_mmvq(l.ffn_up.type, l.ffn_up.d, xqf_, up_, n_embd_, n_ff_, 1, s);
        k::geglu_quant(gate_, up_, n_ff_, xqh_, n_ff_, 1, s);
        native_mmvq(l.ffn_down.type, l.ffn_down.d, xqh_, mlp_, n_ff_, n_embd_, 1, s);
        const bool last = il + 1 == layers_.size();
        k::moe_post_fused(nullptr, nullptr, nullptr, nullptr, 0, mlp_, x1_, nullptr, nullptr, l.post_ffw_norm.f32(),
                          l.out_scale, x_, last ? output_norm_.f32() : layers_[il + 1].attn_norm.f32(), xq_, n_embd_, 1,
                          eps_, s);
    }
    // xq_ = q8_1(rms(x) * output_norm): the draft token and the hidden state for the next step
    native_mmvq(tok_embd_.type, tok_embd_.d, xq_, logits_, n_embd_, n_vocab_, 1, s);
    k::argmax_rows(logits_, n_vocab_, 1, drafts_ + i, s);
    native_mmvq(post_proj_.type, post_proj_.d, xq_, h_, n_embd_, n_tgt_, 1, s);
}

void Mtp::draft(int token, int pos, const float* h_dev, int n, std::vector<int32_t>& out) {
    out.clear();
    if (n <= 0) return;
    if (n > max_draft_) n = max_draft_;
    const cudaStream_t s = eng_.stream();
    const int32_t in[2] = {token, pos};
    ck(cudaMemcpyAsync(tok_, &in[0], 4, cudaMemcpyHostToDevice, s), "token");
    ck(cudaMemcpyAsync(pos_, &in[1], 4, cudaMemcpyHostToDevice, s), "pos");
    ck(cudaMemcpyAsync(h_, h_dev, sizeof(float) * n_tgt_, cudaMemcpyDeviceToDevice, s), "hidden");
    if (!graphs_[n]) {
        cudaGraph_t gph;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
        k::mtp_ranges(pos_, n_swa_, ranges_, ranges_ + 1, ranges_ + 2, ranges_ + 3, s);
        for (int i = 0; i < n; ++i) step(i);
        ck(cudaStreamEndCapture(s, &gph), "capture end");
        cudaGraphExec_t ge;
        ck(cudaGraphInstantiate(&ge, gph, 0), "instantiate");
        cudaGraphDestroy(gph);
        graphs_[n] = ge;
    }
    ck(cudaGraphLaunch((cudaGraphExec_t) graphs_[n], s), "launch");
    out.resize(n);
    ck(cudaMemcpyAsync(out.data(), drafts_, 4 * n, cudaMemcpyDeviceToHost, s), "drafts");
    ck(cudaStreamSynchronize(s), "draft");
}

}  // namespace strata::gemma
