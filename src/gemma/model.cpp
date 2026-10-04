// src/gemma/model.cpp - see include/strata/gemma/model.hpp.
#include "strata/gemma/model.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::gemma {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct TypeInfo { int block; size_t bytes; };
TypeInfo type_info(int t) {
    switch (t) {
        case T_F32: return {1, 4};
        case T_F16: case T_BF16: return {1, 2};
        case T_Q4_0: return {32, 18};
        case T_Q5_0: return {32, 22};
        case T_Q8_0: return {32, 34};
        case T_Q4_K: return {256, 144};
        case T_Q5_K: return {256, 176};
        case T_Q6_K: return {256, 210};
        default: throw std::runtime_error("unsupported ggml type " + std::to_string(t) + " (" + ggml_type_name(t) + ")");
    }
}

const MetaValue& need(const GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    if (!v) throw std::runtime_error("GGUF: missing " + key);
    return *v;
}
uint64_t get_u(const GgufFile& g, const std::string& key, uint64_t def, bool required = true) {
    const MetaValue* v = g.get(key);
    if (!v) {
        if (required) throw std::runtime_error("GGUF: missing " + key);
        return def;
    }
    return v->type == MetaType::ARRAY ? v->items.at(0).u : v->u;
}
double get_f(const GgufFile& g, const std::string& key, double def, bool required = true) {
    const MetaValue* v = g.get(key);
    if (!v) {
        if (required) throw std::runtime_error("GGUF: missing " + key);
        return def;
    }
    return v->num();
}
// a key that is either one value for every layer or an array with one per layer
uint64_t per_layer_u(const GgufFile& g, const std::string& key, int il) {
    const MetaValue& v = need(g, key);
    if (v.type != MetaType::ARRAY) return v.u;
    if ((uint64_t) il >= v.items.size()) throw std::runtime_error("GGUF: " + key + " has no entry for layer " + std::to_string(il));
    return v.items[il].u;
}

}  // namespace

size_t row_bytes(int type, int64_t n) {
    const TypeInfo ti = type_info(type);
    if (n % ti.block) throw std::runtime_error("row of " + std::to_string(n) + " values is not a whole number of blocks");
    return (size_t) (n / ti.block) * ti.bytes;
}

Model::~Model() {
    if (arena_) cudaFree(arena_);
}

std::unique_ptr<Model> Model::load(const std::string& path, bool verbose) {
    const auto t0 = std::chrono::steady_clock::now();
    GgufFile g(path);
    const MetaValue* arch = g.get("general.architecture");
    if (!arch || arch->s != "gemma4")
        throw std::runtime_error("architecture is '" + (arch ? arch->s : std::string("?")) + "', this engine runs 'gemma4'");
    if (get_u(g, "gemma4.embedding_length_per_layer_input", 0, false) != 0)
        throw std::runtime_error("per-layer embeddings (Gemma 4 E2B/E4B) are not implemented yet; this engine runs the MoE models");
    if (get_u(g, "gemma4.attention.shared_kv_layers", 0, false) != 0)
        throw std::runtime_error("shared KV layers are not implemented");

    std::unique_ptr<Model> m(new Model());
    Config& c = m->cfg;
    c.n_layer = (int) get_u(g, "gemma4.block_count", 0);
    c.n_embd = (int) get_u(g, "gemma4.embedding_length", 0);
    c.n_ff = (int) get_u(g, "gemma4.feed_forward_length", 0);
    c.n_expert = (int) get_u(g, "gemma4.expert_count", 0, false);
    c.n_expert_used = (int) get_u(g, "gemma4.expert_used_count", 0, false);
    c.n_ff_exp = (int) get_u(g, "gemma4.expert_feed_forward_length", 0, false);
    c.n_swa = (int) get_u(g, "gemma4.attention.sliding_window", 0);
    c.eps = (float) get_f(g, "gemma4.attention.layer_norm_rms_epsilon", 1e-6);
    c.softcap = (float) get_f(g, "gemma4.final_logit_softcapping", 0.0, false);
    const int hd_global = (int) get_u(g, "gemma4.attention.key_length", 0);
    const int hd_swa = (int) get_u(g, "gemma4.attention.key_length_swa", 0);
    const int rot_global = (int) get_u(g, "gemma4.rope.dimension_count", hd_global, false);
    const int rot_swa = (int) get_u(g, "gemma4.rope.dimension_count_swa", hd_swa, false);
    const float base_global = (float) get_f(g, "gemma4.rope.freq_base", 1e6);
    const float base_swa = (float) get_f(g, "gemma4.rope.freq_base_swa", 1e4, false);
    const MetaValue& pattern = need(g, "gemma4.attention.sliding_window_pattern");

    // ---- the tensors, laid out back to back in one arena (256-byte aligned)
    struct Pending { Tensor* dst; const TensorInfo* ti; };
    std::vector<Pending> pend;
    size_t total = 0;
    auto want = [&](Tensor& dst, const std::string& name, bool required = true) {
        const TensorInfo* ti = g.find(name);
        if (!ti) {
            if (required) throw std::runtime_error("GGUF: missing tensor " + name);
            return;
        }
        dst.type = (int) ti->type;
        for (size_t i = 0; i < 3; ++i) dst.ne[i] = i < ti->shape.size() ? (int64_t) ti->shape[i] : 1;
        dst.nb1 = row_bytes(dst.type, dst.ne[0]);
        dst.nb2 = dst.nb1 * (size_t) dst.ne[1];
        total = (total + 255) / 256 * 256;
        dst.d = reinterpret_cast<const void*>(total);   // an offset until the arena exists
        total += dst.nb2 * (size_t) dst.ne[2];
        // MMQ reads whole 256-value tiles: a row whose length is not a multiple of 512 overreads into the next
        // row (finite weights times zero-padded activations) and, on the last row, past the tensor - which must be
        // zeros, not the next tensor's f16 scales (a NaN times zero is NaN). ggml-cuda's buffers pad the same way.
        if (dst.type != T_F32 && dst.type != T_F16 && dst.type != T_BF16) total += row_bytes(dst.type, 512);
        pend.push_back({&dst, ti});
    };

    want(m->tok_embd, "token_embd.weight");
    want(m->output_norm, "output_norm.weight");
    want(m->rope_freqs, "rope_freqs.weight", false);
    c.n_vocab = (int) m->tok_embd.ne[1];
    m->layers.resize(c.n_layer);
    for (int il = 0; il < c.n_layer; ++il) {
        Layer& L = m->layers[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        L.swa = il < (int) pattern.items.size() ? pattern.items[il].u != 0 : true;
        L.n_head = (int) per_layer_u(g, "gemma4.attention.head_count", il);
        L.n_head_kv = (int) per_layer_u(g, "gemma4.attention.head_count_kv", il);
        L.head_dim = L.swa ? hd_swa : hd_global;
        L.n_rot = L.swa ? rot_swa : rot_global;
        L.rope_base = L.swa ? base_swa : base_global;
        want(L.attn_norm, p + "attn_norm.weight");
        want(L.wq, p + "attn_q.weight");
        want(L.wk, p + "attn_k.weight");
        want(L.wv, p + "attn_v.weight", false);
        want(L.wo, p + "attn_output.weight");
        want(L.q_norm, p + "attn_q_norm.weight");
        want(L.k_norm, p + "attn_k_norm.weight");
        want(L.post_attn_norm, p + "post_attention_norm.weight");
        want(L.ffn_norm, p + "ffn_norm.weight");
        want(L.ffn_gate, p + "ffn_gate.weight");
        want(L.ffn_up, p + "ffn_up.weight");
        want(L.ffn_down, p + "ffn_down.weight");
        want(L.post_ffw_norm, p + "post_ffw_norm.weight");
        want(L.router, p + "ffn_gate_inp.weight", false);
        if (L.router) {
            want(L.router_scale, p + "ffn_gate_inp.scale");
            want(L.pre_ffw_norm_2, p + "pre_ffw_norm_2.weight");
            want(L.post_ffw_norm_1, p + "post_ffw_norm_1.weight");
            want(L.post_ffw_norm_2, p + "post_ffw_norm_2.weight");
            want(L.gate_up_exps, p + "ffn_gate_up_exps.weight");
            want(L.down_exps, p + "ffn_down_exps.weight");
            want(L.down_exps_scale, p + "ffn_down_exps.scale", false);
        }
        if (const TensorInfo* s = g.find(p + "layer_output_scale.weight")) {
            if (s->type != T_F32) throw std::runtime_error("layer_output_scale is not F32");
            std::memcpy(&L.out_scale, g.tensor_data(*s), sizeof(float));
        }
        if (L.head_dim * L.n_head != L.wq.ne[1]) throw std::runtime_error(p + "attn_q rows != n_head * head_dim");
        if (L.head_dim * L.n_head_kv != L.wk.ne[1]) throw std::runtime_error(p + "attn_k rows != n_head_kv * head_dim");
        if (L.router && (L.router.type != T_F32 || L.router.ne[1] != c.n_expert))
            throw std::runtime_error(p + "ffn_gate_inp must be F32 [n_embd, n_expert]");
        c.max_head_dim = std::max(c.max_head_dim, L.head_dim);
        c.max_q_dim = std::max(c.max_q_dim, L.head_dim * L.n_head);
        c.max_kv_dim = std::max(c.max_kv_dim, L.head_dim * L.n_head_kv);
    }
    if (m->tok_embd.ne[0] != c.n_embd) throw std::runtime_error("token_embd width != embedding_length");

    // ---- upload: through a pinned staging buffer, so the copy runs at PCIe speed from the page cache
    ck(cudaMalloc(&m->arena_, total), "cudaMalloc(weights)");
    ck(cudaMemset(m->arena_, 0, total), "cudaMemset(weights)");
    m->arena_bytes = total;
    const size_t stage_bytes = 64ull << 20;
    void* stage[2] = {nullptr, nullptr};
    cudaEvent_t done[2];
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    for (int i = 0; i < 2; ++i) {
        ck(cudaMallocHost(&stage[i], stage_bytes), "cudaMallocHost(stage)");
        ck(cudaEventCreate(&done[i]), "event");
        ck(cudaEventRecord(done[i], s), "event");
    }
    int k = 0;
    for (const Pending& pd : pend) {
        const size_t off = reinterpret_cast<size_t>(pd.dst->d);
        const size_t n = pd.dst->nb2 * (size_t) pd.dst->ne[2];
        const uint8_t* src = g.tensor_data(*pd.ti);
        for (size_t o = 0; o < n; o += stage_bytes, k ^= 1) {
            const size_t len = std::min(stage_bytes, n - o);
            ck(cudaEventSynchronize(done[k]), "sync");
            std::memcpy(stage[k], src + o, len);
            ck(cudaMemcpyAsync(static_cast<uint8_t*>(m->arena_) + off + o, stage[k], len, cudaMemcpyHostToDevice, s), "upload");
            ck(cudaEventRecord(done[k], s), "event");
        }
        pd.dst->d = static_cast<uint8_t*>(m->arena_) + off;
    }
    ck(cudaStreamSynchronize(s), "upload sync");
    for (int i = 0; i < 2; ++i) {
        cudaFreeHost(stage[i]);
        cudaEventDestroy(done[i]);
    }
    cudaStreamDestroy(s);
    if (verbose) {
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "gemma: %d layers, n_embd %d, %d experts (top %d, ff %d), dense ff %d, vocab %d, swa %d; "
                             "%.2f GiB of weights on the GPU in %.1f s\n",
                     c.n_layer, c.n_embd, c.n_expert, c.n_expert_used, c.n_ff_exp, c.n_ff, c.n_vocab, c.n_swa,
                     total / 1073741824.0, sec);
    }
    return m;
}

}  // namespace strata::gemma
