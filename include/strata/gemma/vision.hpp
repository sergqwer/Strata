// include/strata/gemma/vision.hpp - Gemma 4's vision encoder (the "gemma4v" mmproj) on the GPU.
//
// What llama.cpp's tools/mtmd/models/gemma4v.cpp builds, computed natively:
//   patches (16x16, pixels * 2 - 1) -> conv as a GEMM -> + learned x / y position rows
//   27 x [ h = rms(x) ln1; q, k = rms_head * norm, 2-D NEOX rope (x on the first half of a head, y on the second,
//          base 100); v = rms_head; a = softmax(q k^T) v (scale 1, every patch sees every patch);
//          x += rms(Wo a) attn_post_norm;  x += rms(Wdown (gelu_quick(Wgate h') * Wup h')) ffn_post_norm ]
//   3x3 average pool -> * sqrt(n_embd) -> (x - std_bias) * std_scale -> rms -> input projection to the LM width.
// The matrices are BF16 (activations are rounded to BF16 for the GEMMs, as ggml's cuBLAS path does); attention runs
// in an own flash-attention kernel on tensor cores (head_dim 72, padded to 80), where llama.cpp falls back to a
// generic tile kernel that took half of its encode time.
#pragma once

#include "strata/gemma/image.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::gemma {

class Vision {
public:
    /// Load an mmproj GGUF onto the current device. tokens: the image token budget (llama-server's
    /// --image-min-tokens = --image-max-tokens); max_patches bounds the work buffers.
    /// int8: the encoder layers' q/k/v, gate/up and down matrices as per-row int8 (quantized at load) and their
    /// GEMMs as int8 x int8 -> int32 with per-patch activation scales (W8A8); attn_out and the projection stay bf16.
    Vision(const std::string& mmproj, int tokens = 280, int max_patches = 4096, bool int8 = false);
    ~Vision();
    Vision(const Vision&) = delete;
    Vision& operator=(const Vision&) = delete;

    /// Encode one preprocessed image (preprocess() below): returns the number of soft tokens; their
    /// embeddings (n_tokens x n_embd_out floats) are appended to `out`.
    int encode(const ImageU8& img, std::vector<float>& out);
    /// Several preprocessed images of one size in one pass (video frames): returns the soft tokens per image;
    /// their embeddings are appended image after image. At most max_batch(img) images of that size.
    int encode_batch(const std::vector<const ImageU8*>& imgs, std::vector<float>& out);
    int max_batch(const ImageU8& img) const;
    /// Time the encoder's phases with CUDA events (diagnosis): milliseconds per phase, summed over the layers and
    /// the encodes since the last profile_take().
    void set_profile(bool on);
    std::vector<std::pair<const char*, double>> profile_take();
    /// Decode + resize as llama.cpp does for this model; with tokens > 0, transformers' resize for that budget
    /// (preprocess_gemma4_hf: video frames, 70 tokens).
    ImageU8 preprocess(const ImageU8& raw, int tokens = 0) const;
    int n_embd_out() const { return n_out_; }
    size_t weight_bytes() const { return weight_bytes_; }
    cudaStream_t stream() const { return s_; }
    /// Milliseconds of the last encode (GPU).
    float last_ms() const { return last_ms_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    int tokens_, n_out_ = 0;
    size_t weight_bytes_ = 0;
    cudaStream_t s_ = nullptr;
    float last_ms_ = 0;
};

}  // namespace strata::gemma
