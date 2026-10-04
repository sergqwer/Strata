// src/gemma/image.cpp - see include/strata/gemma/image.hpp.
//
// The resize is llama.cpp's tools/mtmd/mtmd-image.cpp img_tool (MIT, The ggml authors), itself a port of Pillow's
// Resample.c (HPND license): the same fixed-point separable filter, so an image becomes the same pixels here as in
// llama-server, and the model sees what it was evaluated (and fine-tuned) on.
#include "strata/gemma/image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb/stb_image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::gemma {
namespace {

constexpr int PRECISION_BITS = 32 - 8 - 2;

double bicubic(double x) {   // Pillow's a = -0.5
    if (x < 0.0) x = -x;
    constexpr double a = -0.5;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

uint8_t clip8(int v) { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t) v; }

int precompute_weights(int in_size, int out_size, std::vector<int>& bounds, std::vector<int32_t>& weights) {
    const double filter_support = 2.0;
    double filterscale, scale;
    filterscale = scale = (double) in_size / out_size;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = filter_support * filterscale;
    const int ksize = (int) std::ceil(support) * 2 + 1;
    std::vector<double> pre((size_t) out_size * ksize);
    bounds.resize((size_t) out_size * 2);
    for (int xx = 0; xx < out_size; xx++) {
        const double center = (xx + 0.5) * scale;
        double ww = 0.0;
        const double ss = 1.0 / filterscale;
        int xmin = (int) (center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int) (center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        int x = 0;
        for (; x < xmax; x++) {
            const double w = bicubic((x + xmin - center + 0.5) * ss);
            pre[(size_t) xx * ksize + x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++)
            if (ww != 0.0) pre[(size_t) xx * ksize + x] /= ww;
        for (; x < ksize; x++) pre[(size_t) xx * ksize + x] = 0;
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    weights.resize((size_t) out_size * ksize);
    const double fxp = std::ldexp(1.0, PRECISION_BITS);
    for (size_t i = 0; i < weights.size(); i++) weights[i] = (int32_t) (pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5));
    return ksize;
}

std::vector<uint8_t> resample_h(const uint8_t* src, int in_nx, int in_ny, int out_nx, int ksize,
                                const std::vector<int>& bounds, const std::vector<int32_t>& w) {
    std::vector<uint8_t> out((size_t) out_nx * in_ny * 3);
    for (int yy = 0; yy < in_ny; yy++) {
        const uint8_t* row = src + (size_t) yy * in_nx * 3;
        uint8_t* dst = out.data() + (size_t) yy * out_nx * 3;
        for (int xx = 0; xx < out_nx; xx++) {
            const int xmin = bounds[xx * 2], xcnt = bounds[xx * 2 + 1];
            const int32_t* k = &w[(size_t) xx * ksize];
            const uint8_t* p = row + (size_t) xmin * 3;
            int32_t s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
            for (int x = 0; x < xcnt; x++, p += 3) {
                s0 += p[0] * k[x];
                s1 += p[1] * k[x];
                s2 += p[2] * k[x];
            }
            dst[xx * 3 + 0] = clip8(s0 >> PRECISION_BITS);
            dst[xx * 3 + 1] = clip8(s1 >> PRECISION_BITS);
            dst[xx * 3 + 2] = clip8(s2 >> PRECISION_BITS);
        }
    }
    return out;
}

std::vector<uint8_t> resample_v(const uint8_t* src, int in_nx, int out_ny, int ksize, const std::vector<int>& bounds,
                                const std::vector<int32_t>& w) {
    const size_t row_elems = (size_t) in_nx * 3;
    std::vector<uint8_t> out(row_elems * out_ny);
    std::vector<int32_t> acc(row_elems);
    for (int yy = 0; yy < out_ny; yy++) {
        const int ymin = bounds[yy * 2], ycnt = bounds[yy * 2 + 1];
        const int32_t* k = &w[(size_t) yy * ksize];
        std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));
        for (int y = 0; y < ycnt; y++) {
            const uint8_t* r = src + (size_t) (ymin + y) * row_elems;
            const int32_t wy = k[y];
            for (size_t i = 0; i < row_elems; i++) acc[i] += r[i] * wy;
        }
        uint8_t* dst = out.data() + (size_t) yy * row_elems;
        for (size_t i = 0; i < row_elems; i++) dst[i] = clip8(acc[i] >> PRECISION_BITS);
    }
    return out;
}

ImageU8 resize_pillow(const ImageU8& img, int tw, int th) {
    ImageU8 dst;
    dst.nx = tw;
    dst.ny = th;
    const bool nh = tw != img.nx, nv = th != img.ny;
    std::vector<int> bh, bv;
    std::vector<int32_t> wh, wv;
    int kh = 0, kv = 0;
    if (nh) kh = precompute_weights(img.nx, tw, bh, wh);
    if (nv) kv = precompute_weights(img.ny, th, bv, wv);
    if (nh && nv) {
        auto tmp = resample_h(img.rgb.data(), img.nx, img.ny, tw, kh, bh, wh);
        dst.rgb = resample_v(tmp.data(), tw, th, kv, bv, wv);
    } else if (nh) {
        dst.rgb = resample_h(img.rgb.data(), img.nx, img.ny, tw, kh, bh, wh);
    } else if (nv) {
        dst.rgb = resample_v(img.rgb.data(), img.nx, th, kv, bv, wv);
    } else {
        dst.rgb = img.rgb;
    }
    return dst;
}

}  // namespace

bool decode_image(const uint8_t* data, size_t len, ImageU8& out, std::string& err) {
    int nx = 0, ny = 0, nc = 0;
    unsigned char* px = stbi_load_from_memory(data, (int) len, &nx, &ny, &nc, 3);
    if (!px) {
        err = std::string("cannot decode the image: ") + (stbi_failure_reason() ? stbi_failure_reason() : "?");
        return false;
    }
    out.nx = nx;
    out.ny = ny;
    out.rgb.assign(px, px + (size_t) nx * ny * 3);
    stbi_image_free(px);
    return true;
}

// mtmd's img_tool::calc_size_preserved_ratio (transformers' smart_resize) and resize with PAD_CEIL
ImageU8 preprocess_gemma4(const ImageU8& img, int patch, int merge, int min_tokens, int max_tokens) {
    const int align = patch * merge;
    const int min_pixels = min_tokens * align * align, max_pixels = max_tokens * align * align;
    const int width = img.nx, height = img.ny;
    if (width <= 0 || height <= 0) throw std::runtime_error("empty image");
    auto round_by = [align](float x) { return (int) std::round(x / (float) align) * align; };
    auto ceil_by = [align](float x) { return (int) std::ceil(x / (float) align) * align; };
    auto floor_by = [align](float x) { return (int) std::floor(x / (float) align) * align; };
    int w_bar = std::max(align, round_by((float) width));
    int h_bar = std::max(align, round_by((float) height));
    if (max_pixels > 0 && h_bar * w_bar > max_pixels) {
        const float beta = std::sqrt((float) height * width / max_pixels);
        h_bar = std::max(align, floor_by(height / beta));
        w_bar = std::max(align, floor_by(width / beta));
    } else if (min_pixels > 0 && h_bar * w_bar < min_pixels) {
        const float beta = std::sqrt((float) min_pixels / ((float) height * width));
        h_bar = ceil_by(height * beta);
        w_bar = ceil_by(width * beta);
    }
    if (w_bar == width && h_bar == height) return img;
    // PAD_CEIL: fit inside the target keeping the aspect ratio (rounding up), center it on black
    const float scale = std::min((float) w_bar / width, (float) h_bar / height);
    const int new_w = std::min((int) std::ceil(width * scale), w_bar);
    const int new_h = std::min((int) std::ceil(height * scale), h_bar);
    const ImageU8 r = resize_pillow(img, new_w, new_h);
    ImageU8 dst;
    dst.nx = w_bar;
    dst.ny = h_bar;
    dst.rgb.assign((size_t) w_bar * h_bar * 3, 0);
    const int ox = (w_bar - new_w) / 2, oy = (h_bar - new_h) / 2;
    for (int y = 0; y < new_h; ++y)
        std::copy(r.rgb.begin() + (size_t) y * new_w * 3, r.rgb.begin() + (size_t) (y + 1) * new_w * 3,
                  dst.rgb.begin() + ((size_t) (y + oy) * w_bar + ox) * 3);
    return dst;
}

}  // namespace strata::gemma
