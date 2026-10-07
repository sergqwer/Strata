// tools/gemma/vis_tune_test.cpp - Vision end to end with the GEMM tuner (src/gemma/vision_gemm.cuh) on and off
// (STRATA_VIT_GEMM_TUNE=0): the same synthetic pictures encoded twice, embeddings compared, encode time min over reps.
// On a GPU shared with production use a cut-down mmproj (vis_small_mmproj.py, 2 layers ~90 MB) and small max_patches:
//   (CMake) add_executable(vis-tune-test tools/gemma/vis_tune_test.cpp) + target_link_libraries(... strata_gemma)
//   vis-tune-test mmproj-2l.gguf 1260
#include "strata/gemma/vision.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace strata::gemma;
static ImageU8 noise(int cols, int rows, uint32_t seed) {
    ImageU8 im; im.nx = cols * 16; im.ny = rows * 16; im.rgb.resize((size_t) im.nx * im.ny * 3);
    for (auto& v : im.rgb) { seed = seed * 1664525u + 1013904223u; v = (uint8_t) (seed >> 24); }
    return im;
}
int main(int argc, char** argv) {
    const char* mm = argv[1];
    const int maxp = argc > 2 ? atoi(argv[2]) : 1260;
    struct T { int cols, rows, b; };
    const T tests[] = {{30, 21, 2}, {30, 21, 1}, {24, 24, 1}, {21, 30, 2}, {27, 27, 1}, {12, 9, 3}};
    std::vector<std::vector<float>> out[2];
    std::vector<float> best[2];
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 0) setenv("STRATA_VIT_GEMM_LOG", "1", 1);
        else setenv("STRATA_VIT_GEMM_TUNE", "0", 1);
        Vision v(mm, 280, maxp);
        for (const T& t : tests) {
            std::vector<ImageU8> ims;
            for (int k = 0; k < t.b; ++k) ims.push_back(noise(t.cols, t.rows, 7 + k));
            std::vector<const ImageU8*> p;
            for (auto& im : ims) p.push_back(&im);
            std::vector<float> e;
            float mn = 1e9f;
            for (int r = 0; r < 8; ++r) { e.clear(); v.encode_batch(p, e); mn = std::min(mn, v.last_ms()); }
            out[pass].push_back(e);
            best[pass].push_back(mn);
        }
    }
    for (size_t i = 0; i < out[0].size(); ++i) {
        const auto& a = out[0][i]; const auto& b = out[1][i];
        double ss = 0, sd = 0, mx = 0, md = 0;
        for (size_t j = 0; j < a.size(); ++j) { ss += (double) b[j] * b[j]; sd += (double) (a[j] - b[j]) * (a[j] - b[j]); mx = std::max(mx, (double) std::fabs(b[j])); md = std::max(md, (double) std::fabs(a[j] - b[j])); }
        printf("grid %2dx%2d x%d (%5d rows): %zu floats, rel rms diff %.2e, max|d|/max %.2e | encode min: tuned %.2f ms, default %.2f ms\n",
               tests[i].cols, tests[i].rows, tests[i].b, tests[i].cols * tests[i].rows * tests[i].b, a.size(),
               std::sqrt(sd / std::max(ss, 1e-30)), md / std::max(mx, 1e-30), best[0][i], best[1][i]);
    }
    return 0;
}
