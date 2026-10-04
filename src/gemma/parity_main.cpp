// strata-gemma-parity - the engine against llama.cpp: replay a tools/gemma/ref_logits dump (teacher-forced) and
// compare the logits at every scored position.
//
//   strata-gemma-parity -m model.gguf --ref ref.bin [--ctx 2048] [--batch 2048] [--split N]
//
// --split N feeds the prompt in chunks of N rows (N <= 8 exercises the decode path on the prompt).
#include "strata/gemma/engine.hpp"
#include "strata/gemma/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using namespace strata::gemma;

namespace {

struct Ref {
    int n_vocab = 0, n_prompt = 0, n_last = 0, n_gen = 0;
    std::vector<int32_t> prompt, gen;
    std::vector<float> last, gen_logits;
};

Ref read_ref(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "SGREF1", 6) != 0) throw std::runtime_error("not a ref_logits dump");
    Ref r;
    int32_t h[4];
    f.read(reinterpret_cast<char*>(h), sizeof h);
    r.n_vocab = h[0];
    r.n_prompt = h[1];
    r.n_last = h[2];
    r.n_gen = h[3];
    r.prompt.resize(r.n_prompt);
    f.read(reinterpret_cast<char*>(r.prompt.data()), r.n_prompt * 4);
    r.last.resize((size_t) r.n_last * r.n_vocab);
    f.read(reinterpret_cast<char*>(r.last.data()), r.last.size() * 4);
    r.gen.resize(r.n_gen);
    f.read(reinterpret_cast<char*>(r.gen.data()), r.n_gen * 4);
    r.gen_logits.resize((size_t) r.n_gen * r.n_vocab);
    f.read(reinterpret_cast<char*>(r.gen_logits.data()), r.gen_logits.size() * 4);
    if (!f) throw std::runtime_error("truncated ref dump");
    return r;
}

struct Cmp {
    float max_abs = 0, rms = 0;
    bool top1 = false;
    int top10 = 0;
    int ours = 0, theirs = 0;
};

Cmp compare(const float* a, const float* b, int n) {
    Cmp c;
    double ss = 0;
    for (int i = 0; i < n; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        c.max_abs = std::max(c.max_abs, d);
        ss += (double) d * d;
    }
    c.rms = (float) std::sqrt(ss / n);
    std::vector<int> ia(n), ib(n);
    std::iota(ia.begin(), ia.end(), 0);
    std::iota(ib.begin(), ib.end(), 0);
    std::partial_sort(ia.begin(), ia.begin() + 10, ia.end(), [&](int x, int y) { return a[x] > a[y]; });
    std::partial_sort(ib.begin(), ib.begin() + 10, ib.end(), [&](int x, int y) { return b[x] > b[y]; });
    c.ours = ia[0];
    c.theirs = ib[0];
    c.top1 = ia[0] == ib[0];
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j) c.top10 += ia[i] == ib[j];
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, ref_path;
    int ctx = 2048, batch = 2048, split = 0;
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model = next();
        else if (a == "--ref") ref_path = next();
        else if (a == "--ctx") ctx = std::stoi(next());
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--split") split = std::stoi(next());
        else if (a == "--bench") bench = true;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    try {
        const Ref ref = read_ref(ref_path);
        auto m = Model::load(model);
        if (m->cfg.n_vocab != ref.n_vocab) throw std::runtime_error("vocab size differs from the ref dump");
        Engine eng(*m, ctx, batch);
        std::fprintf(stderr, "engine buffers %.2f GiB\n", eng.buffer_bytes() / 1073741824.0);
        for (int32_t t : ref.prompt)
            if (t < 0) throw std::runtime_error("this tool replays text-only dumps");

        if (bench) {   // how many distinct experts do windows of consecutive tokens choose (the last layer)?
            Batch b;
            b.tokens = ref.prompt;
            eng.forward(b, true);
            const auto ids = eng.debug_expert_ids((int) ref.prompt.size());
            const int K = m->cfg.n_expert_used;
            for (int w : {2, 4, 8}) {
                double tot = 0;
                int cnt = 0;
                for (size_t t0 = 0; t0 + w <= ref.prompt.size(); t0 += w, ++cnt) {
                    std::vector<int32_t> u(ids.begin() + t0 * K, ids.begin() + (t0 + w) * K);
                    std::sort(u.begin(), u.end());
                    tot += std::unique(u.begin(), u.end()) - u.begin();
                }
                std::printf("windows of %d tokens: %.1f distinct experts of %d picks\n", w, tot / cnt, w * K);
            }
            eng.reset();
        }
        // the prompt
        const int step = split > 0 ? split : (int) ref.prompt.size();
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t o = 0; o < ref.prompt.size(); o += step) {
            Batch b;
            b.tokens.assign(ref.prompt.begin() + o, ref.prompt.begin() + std::min(ref.prompt.size(), o + step));
            eng.forward(b, o + step >= ref.prompt.size());
        }
        const double t_prompt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        Cmp c = compare(eng.logits_host(), ref.last.data() + (size_t) (ref.n_last - 1) * ref.n_vocab, ref.n_vocab);
        std::printf("prompt %d tokens in %.1f ms: max|d| %.4f rms %.4f top1 %s (ours %d, ref %d) top10 %d/10\n",
                    ref.n_prompt, t_prompt * 1000, c.max_abs, c.rms, c.top1 ? "ok" : "DIFF", c.ours, c.theirs, c.top10);

        // teacher-forced generation: feed ref token i, compare with the ref's logits after it
        int top1_ok = c.top1, scored = 1;
        float worst = c.max_abs;
        double t_dec = 0;
        for (int i = 0; i + 1 < ref.n_gen; ++i) {
            Batch b;
            b.tokens = {ref.gen[i]};
            const auto t1 = std::chrono::steady_clock::now();
            eng.forward(b, true);
            t_dec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            c = compare(eng.logits_host(), ref.gen_logits.data() + (size_t) (i + 1) * ref.n_vocab, ref.n_vocab);
            top1_ok += c.top1;
            ++scored;
            worst = std::max(worst, c.max_abs);
            if (!c.top1 || i < 3)
                std::printf("  gen %2d: max|d| %.4f rms %.4f top1 %s (ours %d, ref %d) top10 %d/10\n", i + 1, c.max_abs,
                            c.rms, c.top1 ? "ok" : "DIFF", c.ours, c.theirs, c.top10);
        }
        std::printf("top-1 agreement %d/%d, worst max|d| %.4f, decode %.2f ms/token\n", top1_ok, scored, worst,
                    ref.n_gen > 1 ? t_dec * 1000 / (ref.n_gen - 1) : 0.0);
        if (bench) {   // the decode / verify path: n rows at the end of the prompt, logits for every row
            const int base = eng.n_past();
            for (int n = 1; n <= Engine::kSmall; ++n) {
                Batch b;
                for (int i = 0; i < n; ++i) b.tokens.push_back(ref.gen[i % ref.n_gen]);
                for (int w = 0; w < 3; ++w) { eng.truncate(base); eng.forward(b, Engine::Logits::All); }
                const auto t1 = std::chrono::steady_clock::now();
                const int reps = 20;
                for (int w = 0; w < reps; ++w) { eng.truncate(base); eng.forward(b, Engine::Logits::All); }
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / reps;
                std::printf("  %d rows: %.2f ms (%.2f ms/row)\n", n, ms, ms / n);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
