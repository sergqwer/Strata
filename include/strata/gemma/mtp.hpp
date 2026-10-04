// include/strata/gemma/mtp.hpp - the Gemma 4 "assistant" (gemma4-assistant GGUF, Google's multi-token-prediction
// drafter) for speculative decoding: a 4-layer, 1024-wide model that reads the target's last hidden state and token
// and attends to the TARGET's KV cache (its sliding layers to the target's last sliding layer, its global layer to the
// target's last layer) - it has no K/V of its own. As in llama.cpp (common/speculative.cpp, is_mem_shared), every
// draft step sits at the same position (the position of the newest target token) and passes its own projected hidden
// state to the next step. Greedy: the drafts are the argmax tokens; the target verifies them in one window.
#pragma once

#include "strata/gemma/engine.hpp"
#include "strata/gemma/model.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::gemma {

class Mtp {
public:
    Mtp(const std::string& path, const Model& target, Engine& engine, int max_draft = 6);
    ~Mtp();
    Mtp(const Mtp&) = delete;
    Mtp& operator=(const Mtp&) = delete;

    /// Draft n tokens after `token` (the newest target token, not yet in the target cache, at position `pos`), from
    /// the target hidden state h_dev (n_embd of the target, on the device). Runs on the engine's stream; returns the
    /// drafted ids in `out`.
    void draft(int token, int pos, const float* h_dev, int n, std::vector<int32_t>& out);
    size_t bytes() const { return bytes_; }
    int max_draft() const { return max_draft_; }

private:
    struct L {
        bool swa;
        int n_head, n_kv, hd;
        float base, out_scale;
        Tensor attn_norm, wq, q_norm, wo, post_attn_norm, ffn_norm, ffn_gate, ffn_up, ffn_down, post_ffw_norm;
    };
    void step(int i);

    const Model& tgt_;
    Engine& eng_;
    int max_draft_;
    int n_embd_ = 0, n_ff_ = 0, n_tgt_ = 0, n_vocab_ = 0, n_swa_ = 0;
    float eps_ = 1e-6f;
    std::vector<L> layers_;
    Tensor tok_embd_, output_norm_, pre_proj_, post_proj_, rope_freqs_;
    void* arena_ = nullptr;
    size_t bytes_ = 0;
    // work buffers (device)
    void* work_ = nullptr;
    int32_t *tok_ = nullptr, *pos_ = nullptr, *ranges_ = nullptr, *drafts_ = nullptr;
    float *xh_ = nullptr, *x_ = nullptr, *x1_ = nullptr, *q_ = nullptr, *att_ = nullptr, *y_ = nullptr, *gate_ = nullptr,
          *up_ = nullptr, *mlp_ = nullptr, *logits_ = nullptr, *h_ = nullptr;
    void *xq_ = nullptr, *xqf_ = nullptr, *xqh_ = nullptr, *scratch_ = nullptr;
    std::vector<void*> graphs_;   // per draft length
};

}  // namespace strata::gemma
