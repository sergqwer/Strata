// tools/gemma/glue/common.hpp - the harness's plumbing: one or two layers' real weights from the GGUF on the GPU,
// device buffers, bitwise comparison and loop timing.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/gemma/model.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace glue {

using strata::gemma::Config;
using strata::gemma::Layer;
using strata::gemma::Tensor;

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// every device allocation of the harness, so the VRAM budget (450 MiB on the shared card) is visible
struct Dev {
    std::vector<void*> ptrs;
    size_t bytes = 0;
    template <class T> T* alloc(size_t n, bool zero = true) {
        void* p = nullptr;
        const size_t b = std::max<size_t>(n * sizeof(T), 256);
        ck(cudaMalloc(&p, b), "cudaMalloc");
        if (zero) ck(cudaMemset(p, 0, b), "cudaMemset");
        ptrs.push_back(p);
        bytes += b;
        return static_cast<T*>(p);
    }
    ~Dev() {
        for (void* p : ptrs) cudaFree(p);
    }
};

inline size_t type_row_bytes(int type, int64_t n) {
    switch (type) {
        case 0: return n * 4;
        case 1: case 30: return n * 2;
        case 2: return n / 32 * 18;
        case 6: return n / 32 * 22;
        case 8: return n / 32 * 34;
        case 12: return n / 256 * 144;
        case 13: return n / 256 * 176;
        case 14: return n / 256 * 210;
        default: throw std::runtime_error("type " + std::to_string(type));
    }
}

// Upload a tensor; keep only the first keep1 rows (ne1) / keep2 matrices (ne2) when >= 0. Quantized tensors get the
// engine's 512-value zero tail (Model::load).
inline Tensor upload(const strata::GgufFile& g, Dev& dev, const std::string& name, int64_t keep1 = -1, int64_t keep2 = -1,
                     bool required = true) {
    Tensor t;
    const strata::TensorInfo* ti = g.find(name);
    if (!ti) {
        if (required) throw std::runtime_error("missing tensor " + name);
        return t;
    }
    t.type = (int) ti->type;
    for (size_t i = 0; i < 3; ++i) t.ne[i] = i < ti->shape.size() ? (int64_t) ti->shape[i] : 1;
    t.nb1 = type_row_bytes(t.type, t.ne[0]);
    t.nb2 = t.nb1 * (size_t) t.ne[1];
    const uint8_t* src = g.tensor_data(*ti);
    int64_t n1 = keep1 >= 0 ? std::min(keep1, t.ne[1]) : t.ne[1];
    int64_t n2 = keep2 >= 0 ? std::min(keep2, t.ne[2]) : t.ne[2];
    const bool quant = t.type != 0 && t.type != 1 && t.type != 30;
    const size_t pad = quant ? type_row_bytes(t.type, 512) : 0;
    const size_t mat_bytes_new = t.nb1 * (size_t) n1;
    uint8_t* d = dev.alloc<uint8_t>(mat_bytes_new * n2 + pad);
    for (int64_t e = 0; e < n2; ++e)
        ck(cudaMemcpy(d + e * mat_bytes_new, src + e * t.nb2, mat_bytes_new, cudaMemcpyHostToDevice), "upload");
    t.ne[1] = n1;
    t.ne[2] = n2;
    t.nb2 = mat_bytes_new;
    t.d = d;
    return t;
}

inline uint64_t meta_u(const strata::GgufFile& g, const std::string& k, uint64_t def) {
    const strata::MetaValue* v = g.get(k);
    if (!v) return def;
    return v->type == strata::MetaType::ARRAY ? v->items.at(0).u : v->u;
}
inline double meta_f(const strata::GgufFile& g, const std::string& k, double def) {
    const strata::MetaValue* v = g.get(k);
    return v ? v->num() : def;
}

// Config + one layer as Model::load builds them (only the hyperparameters the harness needs)
inline Config read_config(const strata::GgufFile& g) {
    Config c;
    c.n_layer = (int) meta_u(g, "gemma4.block_count", 0);
    c.n_embd = (int) meta_u(g, "gemma4.embedding_length", 0);
    c.n_ff = (int) meta_u(g, "gemma4.feed_forward_length", 0);
    c.n_expert = (int) meta_u(g, "gemma4.expert_count", 0);
    c.n_expert_used = (int) meta_u(g, "gemma4.expert_used_count", 0);
    c.n_ff_exp = (int) meta_u(g, "gemma4.expert_feed_forward_length", 0);
    c.n_swa = (int) meta_u(g, "gemma4.attention.sliding_window", 0);
    c.eps = (float) meta_f(g, "gemma4.attention.layer_norm_rms_epsilon", 1e-6);
    c.softcap = (float) meta_f(g, "gemma4.final_logit_softcapping", 0.0);
    return c;
}

inline bool layer_is_swa(const strata::GgufFile& g, int il) {
    const strata::MetaValue* p = g.get("gemma4.attention.sliding_window_pattern");
    return !p || il >= (int) p->items.size() || p->items[il].u != 0;
}

// keep_experts < 0: all experts; else the first keep_experts experts and router rows
inline Layer load_layer(const strata::GgufFile& g, Dev& dev, int il, int keep_experts) {
    Layer L;
    const std::string p = "blk." + std::to_string(il) + ".";
    auto per_layer = [&](const std::string& key) -> uint64_t {
        const strata::MetaValue* v = g.get(key);
        if (!v) throw std::runtime_error("missing " + key);
        return v->type == strata::MetaType::ARRAY ? v->items.at(il).u : v->u;
    };
    L.swa = layer_is_swa(g, il);
    L.n_head = (int) per_layer("gemma4.attention.head_count");
    L.n_head_kv = (int) per_layer("gemma4.attention.head_count_kv");
    const int hd_g = (int) meta_u(g, "gemma4.attention.key_length", 0), hd_s = (int) meta_u(g, "gemma4.attention.key_length_swa", 0);
    L.head_dim = L.swa ? hd_s : hd_g;
    L.n_rot = L.swa ? (int) meta_u(g, "gemma4.rope.dimension_count_swa", hd_s) : (int) meta_u(g, "gemma4.rope.dimension_count", hd_g);
    L.rope_base = L.swa ? (float) meta_f(g, "gemma4.rope.freq_base_swa", 1e4) : (float) meta_f(g, "gemma4.rope.freq_base", 1e6);
    L.attn_norm = upload(g, dev, p + "attn_norm.weight");
    L.wq = upload(g, dev, p + "attn_q.weight");
    L.wk = upload(g, dev, p + "attn_k.weight");
    L.wv = upload(g, dev, p + "attn_v.weight", -1, -1, false);
    L.wo = upload(g, dev, p + "attn_output.weight");
    L.q_norm = upload(g, dev, p + "attn_q_norm.weight");
    L.k_norm = upload(g, dev, p + "attn_k_norm.weight");
    L.post_attn_norm = upload(g, dev, p + "post_attention_norm.weight");
    L.ffn_norm = upload(g, dev, p + "ffn_norm.weight");
    L.ffn_gate = upload(g, dev, p + "ffn_gate.weight");
    L.ffn_up = upload(g, dev, p + "ffn_up.weight");
    L.ffn_down = upload(g, dev, p + "ffn_down.weight");
    L.post_ffw_norm = upload(g, dev, p + "post_ffw_norm.weight");
    L.router = upload(g, dev, p + "ffn_gate_inp.weight", keep_experts, -1, false);
    if (L.router) {
        L.router_scale = upload(g, dev, p + "ffn_gate_inp.scale");
        L.pre_ffw_norm_2 = upload(g, dev, p + "pre_ffw_norm_2.weight");
        L.post_ffw_norm_1 = upload(g, dev, p + "post_ffw_norm_1.weight");
        L.post_ffw_norm_2 = upload(g, dev, p + "post_ffw_norm_2.weight");
        L.gate_up_exps = upload(g, dev, p + "ffn_gate_up_exps.weight", -1, keep_experts);
        L.down_exps = upload(g, dev, p + "ffn_down_exps.weight", -1, keep_experts);
        L.down_exps_scale = upload(g, dev, p + "ffn_down_exps.scale", -1, -1, false);
    }
    if (const strata::TensorInfo* s = g.find(p + "layer_output_scale.weight")) std::memcpy(&L.out_scale, g.tensor_data(*s), 4);
    return L;
}

// ---- data

inline std::vector<float> randn(size_t n, uint64_t seed, float sigma = 1.f) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, sigma);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
template <class T> void to_dev(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d"); }
template <class T> std::vector<T> from_dev(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

// bitwise comparison of two device buffers; prints the first difference
inline bool same_bits(const void* a, const void* b, size_t bytes, const char* what, bool as_float = true) {
    std::vector<uint8_t> ha(bytes), hb(bytes);
    ck(cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost), "d2h");
    ck(cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost), "d2h");
    if (std::memcmp(ha.data(), hb.data(), bytes) == 0) {
        std::printf("  %-34s %10zu bytes  IDENTICAL\n", what, bytes);
        return true;
    }
    size_t ndiff = 0, first = (size_t) -1;
    if (as_float && bytes % 4 == 0) {
        const float* fa = reinterpret_cast<const float*>(ha.data());
        const float* fb = reinterpret_cast<const float*>(hb.data());
        for (size_t i = 0; i < bytes / 4; ++i)
            if (std::memcmp(fa + i, fb + i, 4)) {
                if (first == (size_t) -1) first = i;
                ++ndiff;
            }
        std::printf("  %-34s %10zu bytes  DIFFER: %zu of %zu words, first [%zu] %.9g vs %.9g\n", what, bytes, ndiff,
                    bytes / 4, first, fa[first], fb[first]);
    } else {
        for (size_t i = 0; i < bytes; ++i)
            if (ha[i] != hb[i]) {
                if (first == (size_t) -1) first = i;
                ++ndiff;
            }
        std::printf("  %-34s %10zu bytes  DIFFER: %zu bytes, first at %zu\n", what, bytes, ndiff, first);
    }
    return false;
}

// ---- timing: a captured graph replayed `reps` times per sample (sized to ~target_ms), samples interleaved between
// the variants, min and median per replay

struct Timed {
    std::string name;
    cudaGraphExec_t exec = nullptr;
    int reps = 1;
    std::vector<float> ms;   // per replay
};

inline cudaGraphExec_t capture(cudaStream_t s, const std::function<void()>& body) {
    cudaGraph_t g;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
    body();
    ck(cudaStreamEndCapture(s, &g), "capture end");
    cudaGraphExec_t e;
    ck(cudaGraphInstantiate(&e, g, 0), "instantiate");
    cudaGraphDestroy(g);
    return e;
}

inline void time_interleaved(std::vector<Timed>& v, cudaStream_t s, int samples, double target_ms) {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "ev");
    ck(cudaEventCreate(&b), "ev");
    for (auto& t : v) {   // warm up and size the loop
        ck(cudaGraphLaunch(t.exec, s), "launch");
        ck(cudaEventRecord(a, s), "rec");
        for (int i = 0; i < 3; ++i) ck(cudaGraphLaunch(t.exec, s), "launch");
        ck(cudaEventRecord(b, s), "rec");
        ck(cudaEventSynchronize(b), "sync");
        float ms;
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
        t.reps = std::max(1, (int) (target_ms / (ms / 3)));
    }
    for (int r = 0; r < samples; ++r)
        for (auto& t : v) {
            ck(cudaEventRecord(a, s), "rec");
            for (int i = 0; i < t.reps; ++i) ck(cudaGraphLaunch(t.exec, s), "launch");
            ck(cudaEventRecord(b, s), "rec");
            ck(cudaEventSynchronize(b), "sync");
            float ms;
            ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
            t.ms.push_back(ms / t.reps);
        }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

inline void report(const std::vector<Timed>& v, const char* unit_name = "replay") {
    for (const auto& t : v) {
        std::vector<float> m = t.ms;
        std::sort(m.begin(), m.end());
        std::printf("  %-40s min %9.4f ms  median %9.4f ms per %s (%d reps x %zu samples)\n", t.name.c_str(), m.front(),
                    m[m.size() / 2], unit_name, t.reps, m.size());
    }
}

}  // namespace glue
