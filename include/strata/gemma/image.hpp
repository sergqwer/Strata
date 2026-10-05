// include/strata/gemma/image.hpp - image decoding and Gemma 4's resize, bit-for-bit as llama.cpp's mtmd does them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::gemma {

struct ImageU8 {
    int nx = 0, ny = 0;
    std::vector<uint8_t> rgb;   // ny rows of nx RGB pixels
};

/// PNG / JPEG / BMP / GIF... (stb_image) to RGB.
bool decode_image(const uint8_t* data, size_t len, ImageU8& out, std::string& err);

/// mtmd's Gemma 4 (dyn_size) preprocessing: the size that keeps the aspect ratio, aligned to patch * merge pixels,
/// with min_tokens..max_tokens pooled tokens; Pillow bicubic; centered on black when the ratio cannot be kept exactly.
ImageU8 preprocess_gemma4(const ImageU8& img, int patch, int merge, int min_tokens, int max_tokens);
/// transformers' Gemma 4 resize (get_aspect_ratio_preserving_size): the largest size at the image's aspect ratio
/// within max_tokens soft tokens, both sides floored to patch * merge; no padding. The format Gemma 4 sees video
/// frames in (70 tokens each), and the one a LoRA trained with the HF processor saw.
ImageU8 preprocess_gemma4_hf(const ImageU8& img, int patch, int merge, int max_tokens);

}  // namespace strata::gemma
