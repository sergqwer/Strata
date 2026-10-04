// strata-gemma-vision - encode an image with the native Gemma 4 vision encoder; optionally compare with an embedding
// dump of llama.cpp's mtmd (tools/gemma/vis_bench --dump) and time it.
//   strata-gemma-vision --mmproj mm.gguf --image a.png [--ref ref.emb] [-r 5] [--tokens 280]
#include "strata/gemma/vision.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace strata::gemma;

int main(int argc, char** argv) {
    std::string mmproj, image, ref;
    int reps = 3, tokens = 280;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mmproj") mmproj = argv[++i];
        else if (a == "--image") image = argv[++i];
        else if (a == "--ref") ref = argv[++i];
        else if (a == "-r") reps = std::stoi(argv[++i]);
        else if (a == "--tokens") tokens = std::stoi(argv[++i]);
    }
    try {
        Vision v(mmproj, tokens);
        std::ifstream f(image, std::ios::binary);
        std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), {});
        ImageU8 raw;
        std::string err;
        if (!decode_image(buf.data(), buf.size(), raw, err)) throw std::runtime_error(err);
        std::vector<float> emb;
        int n = 0;
        for (int r = 0; r < reps; ++r) {
            emb.clear();
            const ImageU8 pre = v.preprocess(raw);
            n = v.encode(pre, emb);
            std::printf("rep %d: %dx%d -> %dx%d, %d tokens, encode %.1f ms (GPU)\n", r, raw.nx, raw.ny, pre.nx, pre.ny, n, v.last_ms());
        }
        if (!ref.empty()) {
            std::ifstream rf(ref, std::ios::binary);
            int32_t hdr[2];
            rf.read(reinterpret_cast<char*>(hdr), 8);
            std::vector<float> re((size_t) hdr[0] * hdr[1]);
            rf.read(reinterpret_cast<char*>(re.data()), re.size() * 4);
            if (hdr[0] != n || hdr[1] != v.n_embd_out()) {
                std::printf("ref has %d tokens x %d, ours %d x %d\n", hdr[0], hdr[1], n, v.n_embd_out());
                return 1;
            }
            double ss = 0, sd = 0, dot = 0, na = 0, nb = 0;
            float mx = 0;
            for (size_t i = 0; i < re.size(); ++i) {
                const double d = emb[i] - re[i];
                ss += (double) re[i] * re[i];
                sd += d * d;
                mx = std::max(mx, (float) std::fabs(d));
                dot += (double) emb[i] * re[i];
                na += (double) emb[i] * emb[i];
                nb += (double) re[i] * re[i];
            }
            std::printf("vs llama.cpp: rel rms diff %.3e, max|d| %.4f, cosine %.6f\n", std::sqrt(sd / ss), mx,
                        dot / std::sqrt(na * nb));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
