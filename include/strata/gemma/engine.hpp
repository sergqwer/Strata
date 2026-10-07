// include/strata/gemma/engine.hpp - one sequence of Gemma 4 on the GPU: the KV cache, the work buffers and the forward
// pass, for a prompt chunk (MMQ + cuBLAS attention) or a few tokens (native_mmvq + flash-decoding).
#pragma once

#include "strata/gemma/model.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::gemma {

/// One input row: a token id, or (id < 0) an embedding row of n_embd floats taken from Batch::embd in order
/// (an image's soft tokens, which enter unscaled).
struct Batch {
    std::vector<int32_t> tokens;
    std::vector<float> embd;              // n_embd floats per negative token, in order
    std::vector<int32_t> spans;           // image spans as [begin, end) position pairs: non-causal on sliding layers
};

class Engine {
public:
    /// ctx: the most positions a sequence holds; max_batch: the most rows one forward call takes.
    Engine(const Model& m, int ctx, int max_batch);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    enum class Logits { None, Last, All };
    /// Run rows [0, n) of `b` at positions n_past .. n_past + n - 1 and append them to the cache. Logits::Last scores
    /// the last row, Logits::All every row (n <= kSmall): their argmax is ready (argmax()), the full logits are
    /// copied to the host only when logits_host() asks.
    void forward(const Batch& b, Logits mode);
    void forward(const Batch& b, bool logits) { forward(b, logits ? Logits::Last : Logits::None); }
    /// Forget the sequence (the cache is overwritten from position 0 on the next forward).
    void reset() { n_past_ = 0; }
    /// Keep only the first n positions (speculative verify: drop the rejected draft rows; their cache rows are
    /// simply overwritten later).
    void truncate(int n) { if (n < n_past_) n_past_ = n; }
    /// The post-final-norm hidden state (the head's input, llama.cpp's h_nextn) of scored row i, on the device.
    const float* hidden_dev(int i = -1) const;
    const __half* k_cache(int il) const { return kc_[il]; }
    /// Bytes of one position's K and V over all layers: what a saved prefix costs per token.
    size_t kv_row_bytes() const;
    /// Copy positions [0, n) of every layer's K and V to `host` (pinned, n * kv_row_bytes() bytes), enqueued on the
    /// engine's stream; kv_load copies them back and the next forward continues at position n. A prompt's shared
    /// prefix (system text, question, reference picture) is then computed once and reused.
    void kv_save(int n, void* host) const;
    void kv_load(int n, const void* host);
    const __half* v_cache(int il) const { return vc_[il]; }
    int ctx() const { return ctx_; }
    /// The last layer's expert choices of the last forward (rows x n_expert_used), for diagnostics.
    std::vector<int32_t> debug_expert_ids(int rows) const;
    int n_split() const { return n_split_; }

    int n_past() const { return n_past_; }
    int n_vocab() const { return m_.cfg.n_vocab; }
    /// The scored rows of the last forward: row i in [0, n_scored()).
    int n_scored() const { return n_scored_; }
    /// The argmax of scored row i, a vocabulary id.
    int argmax(int i = -1) const {
        const int a = h_amax_[i < 0 ? n_scored_ - 1 : i];
        return head_ids_.empty() ? a : head_ids_[std::min<size_t>(a, head_ids_.size() - 1)];
    }
    /// Row i's logits on the host: head_n() floats, in head_ids() order (by id when head_ids() is empty).
    const float* logits_host(int i = -1);
    /// Score only these vocabulary rows (ascending ids, at most kHeadRows) from the next forward on, or every row
    /// (empty). A grammar that can only ever emit a few hundred tokens needs no more of the 262k-row head (0.6 GB of
    /// the ~3.3 GB a decode step reads); its constrained choice - the best token it allows - is the same either way.
    void set_head_rows(const std::vector<int32_t>& ids);
    const std::vector<int32_t>& head_ids() const { return head_ids_; }
    int head_n() const { return head_ids_.empty() ? m_.cfg.n_vocab : (int) head_ids_.size(); }
    static constexpr int kHeadRows = 1024;
    const float* logits_dev() const { return logits_; }
    /// CUDA graphs for the decode path (default on; STRATA_NO_GRAPH=1 turns them off).
    void set_graphs(bool on) { graphs_ = on; }
    /// Time the prompt path's phases with CUDA events (one forward after another; for diagnosis): milliseconds per
    /// phase summed over the layers and the forwards since the last profile_take().
    void set_profile(bool on) { prof_ = on; }
    std::vector<std::pair<const char*, double>> profile_take();
    cudaStream_t stream() const { return s_; }
    size_t buffer_bytes() const { return buf_bytes_; }
    /// Rows at or below this use the decode path (native_mmvq, flash-decoding); above, the prompt path.
    static constexpr int kSmall = 8;

private:
    void layer_small(int il, int n);
    void layer_big(int il, int n);
    void head(int row);
    void run_small(int n, Logits mode);
    template <class T> T* carve(size_t count);

    const Model& m_;
    const int ctx_, max_batch_;
    int n_past_ = 0;
    cudaStream_t s_ = nullptr;
    void* cublas_ = nullptr;
    void* mmq_ = nullptr;

    // device arena
    void* arena_ = nullptr;
    size_t buf_bytes_ = 0, carve_off_ = 0;
    std::vector<__half*> kc_, vc_;          // per layer [ctx][n_kv][hd]
    float *x_ = nullptr, *x1_ = nullptr, *h_ = nullptr, *f_ = nullptr, *g_ = nullptr, *r_ = nullptr;
    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *y_ = nullptr, *mlp_ = nullptr, *moe_ = nullptr;
    float *gate_ = nullptr, *up_ = nullptr, *hid_ = nullptr;
    float *rlog_ = nullptr, *ew_ = nullptr;
    int32_t *eid_ = nullptr;
    float *gu_ = nullptr, *eh_ = nullptr, *ey_ = nullptr;
    int32_t *bounds_ = nullptr, *src_ = nullptr, *inv_ = nullptr, *counts_ = nullptr, *iota_ = nullptr;
    float *S_ = nullptr;
    __half *P_ = nullptr, *q16_ = nullptr;
    void *xq_ = nullptr, *xq2_ = nullptr, *att_scratch_ = nullptr;
    int32_t *tok_ = nullptr, *pos_ = nullptr, *lo_ = nullptr, *hi_ = nullptr, *spans_ = nullptr;
    float *embd_ = nullptr, *logits_ = nullptr;
    const void** ptrs_ = nullptr;           // cuBLAS batched pointer arrays (device)
    std::vector<const void*> h_ptrs_;
    std::vector<float> h_logits_;
    std::vector<int32_t> h_amax_, h_bounds_;
    int32_t* amax_ = nullptr;
    float* hnorm_ = nullptr;   // [kSmall][n_embd]
    void* groups_ = nullptr;   // the verify window's (token, expert) pairs grouped by expert
    int hrow0_ = 0;            // hnorm_ row of scored row 0
    void *xqf_ = nullptr, *xqg_ = nullptr, *xqh_ = nullptr, *xqe_ = nullptr;
    int n_scored_ = 0, logits_row_ = -1, n_split_ = 1;
    bool graphs_ = true;
    std::vector<void*> graph_exec_;   // [(n * 3 + mode) * 2 + subset head]: cudaGraphExec_t of run_small
    std::vector<int32_t> head_ids_;   // set_head_rows: the scored vocabulary rows (empty: all)
    void* head_w_ = nullptr;          // their head rows gathered back to back, padded to kHeadRows (the last row repeated)
    int32_t* head_ids_dev_ = nullptr;
    int head_rows() const { return head_ids_.empty() ? m_.cfg.n_vocab : kHeadRows; }   // rows the head kernels run over
    int n_spans_ = 0;
    size_t xq_bytes_ = 0;
    bool prof_ = false;
    std::vector<void*> pev_;          // cudaEvent_t per (layer, phase boundary)
    std::vector<double> ptot_;        // ms per phase
    void mark(int il, int k);
};

}  // namespace strata::gemma
