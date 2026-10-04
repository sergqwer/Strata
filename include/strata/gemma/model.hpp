// include/strata/gemma/model.hpp - Gemma 4 (26B-A4B MoE) weights resident on one GPU.
//
// The model is read from a llama.cpp GGUF (architecture "gemma4") and every tensor is copied into ONE device arena,
// left in its GGUF encoding: the decode GEMVs (native_mmvq) and the prompt GEMMs (llama.cpp's MMQ) read the quantized
// blocks directly. Hyperparameters come from the GGUF metadata, never from constants, so a pruned file (fewer experts,
// lora/prune_gguf.py) or another Gemma 4 MoE size loads the same way.
//
// The layer, as llama.cpp's src/models/gemma4.cpp computes it (that file is the specification; parity is measured
// against it with tools/gemma/ref_logits):
//
//   h   = rms(x) * attn_norm
//   q   = rms_head(Wq h) * q_norm, rope          k = rms_head(Wk h) * k_norm, rope     v = rms_head(Wv h)   (no weight)
//         (global layers have no Wv: v = rms_head(Wk h), taken BEFORE k_norm and rope)
//   a   = softmax(q k^T) v                       (scale 1.0; sliding layers see the last n_swa positions)
//   x1  = x + rms(Wo a) * post_attention_norm
//   mlp = rms(Wdown geglu(Wgate f, Wup f)) * post_ffw_norm_1,   f = rms(x1) * ffn_norm
//   moe = rms(sum_k w_k s_e Wdown_e geglu(Wgu_e g)) * post_ffw_norm_2,   g = rms(x1) * pre_ffw_norm_2
//         router: logits = Wr (rms(x1) / sqrt(n_embd) * ffn_gate_inp.scale); top-k of softmax, renormalized
//   x'  = (x1 + rms(mlp + moe) * post_ffw_norm) * layer_output_scale
//
// and the head: logits = 30 * tanh((token_embd . rms(x) * output_norm) / 30) - tied embeddings, final softcap.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::gemma {

// ggml type ids (ggml.h) of the formats this engine reads
enum : int { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q5_0 = 6, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };

/// Bytes of one row of `n` values in ggml type `type`.
size_t row_bytes(int type, int64_t n);

/// A tensor in the device arena, in GGUF order: ne[0] is the row length (the input dimension of a matrix), ne[1] the
/// rows (outputs), ne[2] the experts of a stacked expert tensor.
struct Tensor {
    const void* d = nullptr;
    int type = -1;
    int64_t ne[3] = {1, 1, 1};
    size_t nb1 = 0;   // bytes of one row
    size_t nb2 = 0;   // bytes of one ne[0] x ne[1] matrix (an expert's stride)
    explicit operator bool() const { return d != nullptr; }
    const float* f32() const { return static_cast<const float*>(d); }
};

struct Layer {
    bool swa = true;
    int n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    float rope_base = 10000.f;
    float out_scale = 1.f;   // layer_output_scale (a scalar read at load)
    Tensor attn_norm, wq, wk, wv, wo, q_norm, k_norm, post_attn_norm;
    Tensor ffn_norm, ffn_gate, ffn_up, ffn_down, post_ffw_norm_1, post_ffw_norm;
    Tensor router, router_scale, pre_ffw_norm_2, post_ffw_norm_2;
    Tensor gate_up_exps, down_exps, down_exps_scale;
    bool moe() const { return static_cast<bool>(router); }
};

struct Config {
    int n_embd = 0, n_layer = 0, n_vocab = 0;
    int n_ff = 0;                       // the dense MLP
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
    int n_swa = 0;
    float eps = 1e-6f, softcap = 0.f;
    int max_head_dim = 0, max_q_dim = 0, max_kv_dim = 0;   // across layers, for buffer sizing
};

class Model {
public:
    /// Load `path` onto the current CUDA device. Throws std::runtime_error with the reason.
    static std::unique_ptr<Model> load(const std::string& path, bool verbose = true);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    Config cfg;
    std::vector<Layer> layers;
    Tensor tok_embd, output_norm, rope_freqs;
    size_t arena_bytes = 0;

private:
    Model() = default;
    void* arena_ = nullptr;
};

}  // namespace strata::gemma
