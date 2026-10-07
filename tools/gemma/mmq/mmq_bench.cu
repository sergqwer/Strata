// tools/gemma/mmq/mmq_bench.cu - the prompt path's MMQ products, old against new, on one layer's real weights.
//
//   mmq_bench --gguf /root/models/c26-drag-q8dense.gguf --layer 0 --routing /root/sg-tools/routing-26b.json
//             [--tokens 692,917,1190,1787] [--dist skew,uniform] [--what moe,dense] [--reps 5] [--loop-ms 100]
//             [--check-only]
//
// "old" is the engine's code before the tile list, call for call (layer_big): the expert rows quantized one gathered
// row at a time, the bounds copied to the host + a stream sync for max_rows, llama.cpp's grid; k::geglu into a float
// buffer, then quantize.  "new" is quantize_scatter + Context::run_tiles + geglu_quantize.  Every output buffer of
// the new code is compared with the old one BYTE FOR BYTE (memcmp; the destination is filled with 0xff before each
// run, so a value the new code fails to write cannot pass).  Timing: loops of ~loop-ms of back-to-back calls, old
// and new interleaved, reps times; min and median of the per-call mean.  Activations are synthetic (normal with
// per-channel scales and rare outliers), the routing is drawn per token (Gumbel top-8) from the layer's routing
// mass in routing-26b.json ("skew") or from equal weights ("uniform").
#include "strata/gemma/kernels.hpp"
#include "strata/gemma/mmq.hpp"

#include "ggml.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace strata::gemma;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct DevBuf {
    void* p = nullptr;
    size_t n = 0;
    DevBuf() = default;
    explicit DevBuf(size_t bytes) : n(bytes) {
        ck(cudaMalloc(&p, bytes + 4096), "cudaMalloc");   // +4 KB: MMQ reads a little past a row's end
        ck(cudaMemset(p, 0, bytes + 4096), "cudaMemset");
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
    ~DevBuf() {
        if (p) cudaFree(p);
    }
    template <class T> T* as() const { return (T*) p; }
};

// ------------------------------------------------------------------------------------------------ inputs

struct Weight {
    ggml_type type;
    int64_t ne0 = 0, ne1 = 0, ne2 = 1;
    size_t nb2 = 0;   // bytes per expert matrix
    DevBuf d;
};

struct Gguf {
    gguf_context* g = nullptr;
    int fd = -1;
    explicit Gguf(const char* path) {
        gguf_init_params ip = {true, nullptr};
        g = gguf_init_from_file(path, ip);
        if (!g) {
            std::fprintf(stderr, "cannot read %s\n", path);
            std::exit(1);
        }
        fd = open(path, O_RDONLY);
    }
    ~Gguf() {
        gguf_free(g);
        close(fd);
    }
    Weight load(const std::string& name, const std::vector<int>* experts = nullptr) const {
        const int64_t id = gguf_find_tensor(g, name.c_str());
        if (id < 0) {
            std::fprintf(stderr, "tensor %s not found\n", name.c_str());
            std::exit(1);
        }
        Weight w;
        w.type = gguf_get_tensor_type(g, id);
        const int64_t* ne = gguf_get_tensor_ne(g, id);
        w.ne0 = ne[0];
        w.ne1 = ne[1];
        w.ne2 = ne[2];
        const size_t bytes = gguf_get_tensor_size(g, id);
        w.nb2 = bytes / (size_t) w.ne2;
        std::vector<uint8_t> h(bytes);
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t r = pread(fd, h.data() + done, bytes - done, (off_t) (off + done));
            if (r <= 0) {
                std::fprintf(stderr, "read %s failed\n", name.c_str());
                std::exit(1);
            }
            done += (size_t) r;
        }
        if (experts) {   // a subset of the expert matrices, packed in the given order
            std::vector<uint8_t> sub(experts->size() * w.nb2);
            for (size_t i = 0; i < experts->size(); ++i)
                std::memcpy(sub.data() + i * w.nb2, h.data() + (size_t) (*experts)[i] * w.nb2, w.nb2);
            h.swap(sub);
            w.ne2 = (int64_t) experts->size();
        }
        w.d = DevBuf(h.size());
        ck(cudaMemcpy(w.d.p, h.data(), h.size(), cudaMemcpyHostToDevice), "upload");
        return w;
    }
};

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    double normal() { return std::sqrt(-2.0 * std::log(uni())) * std::cos(6.283185307179586 * uni()); }
};

// activations like the residual stream after an RMS norm: per-channel scales, rare large outliers
std::vector<float> activations(int64_t rows, int64_t cols, uint64_t seed) {
    Rng r(seed);
    std::vector<float> scale(cols);
    for (auto& s : scale) s = (float) std::exp(0.5 * r.normal());
    std::vector<float> v((size_t) rows * cols);
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t c = 0; c < cols; ++c) {
            double x = r.normal() * scale[c];
            if (r.uni() < 1e-3) x *= 20.0;
            v[(size_t) i * cols + c] = (float) x;
        }
    return v;
}

std::vector<std::vector<double>> read_routing(const char* path) {
    std::vector<std::vector<double>> layers;
    FILE* f = std::fopen(path, "rb");
    if (!f) return layers;
    std::string s;
    char buf[65536];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, r);
    std::fclose(f);
    int depth = 0;
    std::vector<double> cur;
    for (size_t i = 0; i < s.size();) {
        const char c = s[i];
        if (c == '[') {
            ++depth;
            if (depth == 2) cur.clear();
            ++i;
        } else if (c == ']') {
            if (depth == 2) layers.push_back(cur);
            --depth;
            ++i;
        } else if ((c >= '0' && c <= '9') || c == '-' || c == '.') {
            char* end = nullptr;
            cur.push_back(std::strtod(s.c_str() + i, &end));
            i = (size_t) (end - s.c_str());
        } else {
            ++i;
        }
    }
    return layers;
}

struct Routing {
    std::vector<int32_t> eid, bounds, src, inv;   // eid [n*K], bounds [E+1], src [rows] (row -> token), inv [n*K]
    int max_rows = 0;
};

// Gumbel top-K per token from expert weights p (the 96 kept experts, ascending original index)
Routing route(int n, int K, const std::vector<double>& p, uint64_t seed) {
    const int E = (int) p.size();
    Rng r(seed);
    Routing R;
    R.eid.resize((size_t) n * K);
    std::vector<std::pair<double, int>> key(E);
    for (int t = 0; t < n; ++t) {
        for (int e = 0; e < E; ++e) key[e] = {std::log(p[e]) - std::log(-std::log(r.uni())), e};
        std::partial_sort(key.begin(), key.begin() + K, key.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (int s = 0; s < K; ++s) R.eid[(size_t) t * K + s] = key[s].second;
    }
    std::vector<int> cnt(E, 0);
    for (int32_t e : R.eid) ++cnt[e];
    R.bounds.assign(E + 1, 0);
    for (int e = 0; e < E; ++e) R.bounds[e + 1] = R.bounds[e] + cnt[e];
    for (int e = 0; e < E; ++e) R.max_rows = std::max(R.max_rows, cnt[e]);
    std::vector<int> cur(R.bounds.begin(), R.bounds.end() - 1);
    R.src.resize((size_t) n * K);
    R.inv.resize((size_t) n * K);
    for (int t = 0; t < n; ++t)
        for (int s = 0; s < K; ++s) {
            const int p_ = t * K + s, row = cur[R.eid[p_]]++;
            R.src[row] = t;
            R.inv[p_] = row;
        }
    return R;
}

// ------------------------------------------------------------------------------------------------ checks, timing

int g_fail = 0;

// the old and the new run write the same buffer: fill it with 0xff, run, copy back
std::vector<uint8_t> run_capture(const std::function<void()>& f, void* dst, size_t bytes, cudaStream_t s) {
    ck(cudaMemsetAsync(dst, 0xff, bytes, s), "fill");
    f();
    std::vector<uint8_t> h(bytes);
    ck(cudaMemcpyAsync(h.data(), dst, bytes, cudaMemcpyDeviceToHost, s), "capture");
    ck(cudaStreamSynchronize(s), "capture");
    return h;
}

bool compare(const char* what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t elem = 4,
             bool counts = true) {
    const bool same = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()) == 0;
    if (same) {
        std::printf("    %-34s bit-identical (%zu bytes)\n", what, a.size());
        return true;
    }
    if (counts) ++g_fail;
    size_t ndiff = 0, first = SIZE_MAX;
    double maxrel = 0;
    for (size_t i = 0; i + elem <= std::min(a.size(), b.size()); i += elem)
        if (std::memcmp(a.data() + i, b.data() + i, elem) != 0) {
            ++ndiff;
            if (first == SIZE_MAX) first = i / elem;
            if (elem == 4) {
                float x, y;
                std::memcpy(&x, a.data() + i, 4);
                std::memcpy(&y, b.data() + i, 4);
                if (std::isfinite(x) && std::isfinite(y) && x != 0) maxrel = std::max(maxrel, (double) std::fabs((y - x) / x));
            }
        }
    std::printf("    %-34s DIFFERENT: %zu of %zu elements, first at %zu, max rel %.3g\n", what, ndiff, a.size() / elem,
                first, maxrel);
    return false;
}

// llama.cpp's launch for a dense product (mul_mat_q_switch_J + launch_mul_mat_q, Ampere configs): stream-k splits
// tiles over the SMs (and sums their partials in another order) when whole tiles would fill under 90% of the waves
const char* dense_mode(int64_t w_rows, int64_t n, int nsm) {
    const bool fb = w_rows % 128 != 0;
    static const int jnf[] = {8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128}, jfb[] = {8, 16, 32, 64, 128};
    int J = 0, best = 1 << 30;
    for (int j : fb ? std::vector<int>(jfb, jfb + 5) : std::vector<int>(jnf, jnf + 11)) {
        const int t = (int) ((n + j - 1) / j);
        if (t < best) {
            best = t;
            J = j;
        }
        if (best == 1) break;
    }
    const int64_t nt = (int64_t) best * ((w_rows + 127) / 128), waves = (nt + nsm - 1) / nsm;
    return 100 * nt / (nsm * waves) >= 90 ? "tiling" : "stream-k";
    (void) J;
}

struct Stat {
    double min = 0, med = 0;
};
Stat stat(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size() / 2]};
}

int g_reps = 5;
double g_loop_ms = 100;

// ms per call: back-to-back calls for ~g_loop_ms between two events
double loop_ms(const std::function<void()>& f, cudaStream_t s) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    f();
    cudaEventRecord(a, s);
    f();
    cudaEventRecord(b, s);
    ck(cudaEventSynchronize(b), "loop");
    float one = 0;
    cudaEventElapsedTime(&one, a, b);
    const int iters = std::max(3, std::min(2000, (int) (g_loop_ms / std::max(one, 0.01f))));
    cudaEventRecord(a, s);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(b, s);
    ck(cudaEventSynchronize(b), "loop");
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

// interleaved: old, new, old, new, ...; returns {old, new}
std::pair<Stat, Stat> ab(const std::function<void()>& f_old, const std::function<void()>& f_new, cudaStream_t s) {
    std::vector<double> o, n;
    for (int r = 0; r < g_reps; ++r) {
        o.push_back(loop_ms(f_old, s));
        n.push_back(loop_ms(f_new, s));
    }
    return {stat(o), stat(n)};
}

std::map<std::string, std::pair<double, double>> g_sum;   // per phase: old / new median ms per layer

void report(const char* what, const std::pair<Stat, Stat>& r, const std::string& key = "") {
    std::printf("    %-34s old %7.3f ms (min %7.3f)   new %7.3f ms (min %7.3f)   %+.1f%%\n", what, r.first.med, r.first.min,
                r.second.med, r.second.min, 100.0 * (r.second.med / r.first.med - 1.0));
    if (!key.empty()) {
        g_sum[key].first += r.first.med;
        g_sum[key].second += r.second.med;
    }
}

bool g_check_only = false;
double g_vram_mb = 450;        // the whole process (the CUDA context and code take ~175 MiB of it)
int g_frac = 0;                // MoE: 1 = all 96 experts, 2 = every 2nd by routing mass with top-4 (same rows per
                               // expert, half the weights and rows), 0 = 1 if it fits the budget, else 2, else 4
std::vector<std::vector<std::pair<std::string, int>>> g_variants;   // knob settings to time against the old code

}  // namespace

// ------------------------------------------------------------------------------------------------ main

int main(int argc, char** argv) {
    const char* gguf_path = "/root/models/c26-drag-q8dense.gguf";
    const char* routing_path = "/root/sg-tools/routing-26b.json";
    int layer = 0;
    std::vector<int> tokens = {692, 917, 1190, 1787};
    std::vector<std::string> dists = {"skew", "uniform"};
    bool do_moe = true, do_dense = true;
    auto split = [](const std::string& s) {
        std::vector<std::string> v;
        size_t a = 0;
        while (a <= s.size()) {
            const size_t b = s.find(',', a);
            v.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
            if (b == std::string::npos) break;
            a = b + 1;
        }
        return v;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--gguf") gguf_path = argv[++i];
        else if (a == "--routing") routing_path = argv[++i];
        else if (a == "--layer") layer = std::atoi(next().c_str());
        else if (a == "--tokens") {
            tokens.clear();
            for (auto& t : split(next())) tokens.push_back(std::atoi(t.c_str()));
        } else if (a == "--dist") dists = split(next());
        else if (a == "--what") {
            const std::string w = next();
            do_moe = w.find("moe") != std::string::npos;
            do_dense = w.find("dense") != std::string::npos;
        } else if (a == "--reps") g_reps = std::atoi(next().c_str());
        else if (a == "--loop-ms") g_loop_ms = std::atof(next().c_str());
        else if (a == "--check-only") g_check_only = true;
        else if (a == "--vram-mb") g_vram_mb = std::atof(next().c_str());
        else if (a == "--expert-frac") g_frac = std::atoi(next().c_str());
        else if (a == "--variants") {   // e.g. "u8=0;u8=1;fast=0;c0=96,u8=1"
            const std::string v = next();
            size_t p0 = 0;
            while (p0 <= v.size()) {
                const size_t p1 = v.find(';', p0);
                std::vector<std::pair<std::string, int>> var;
                for (auto& kv : split(v.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0))) {
                    const size_t e = kv.find('=');
                    if (e != std::string::npos) var.push_back({kv.substr(0, e), std::atoi(kv.substr(e + 1).c_str())});
                }
                g_variants.push_back(var);
                if (p1 == std::string::npos) break;
                p0 = p1 + 1;
            }
        }
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }

    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    size_t free0 = 0, total = 0;
    cudaMemGetInfo(&free0, &total);
    mmq::Context mq;
    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    Gguf gg(gguf_path);
    const auto routing = read_routing(routing_path);
    const std::string blk = "blk." + std::to_string(layer) + ".";
    const int D = 2816, K = 8;
    std::printf("layer %d, legacy env %d\n", layer, (int) mmq::legacy());

    int nmax = 0;
    for (int n : tokens) nmax = std::max(nmax, n);
    std::vector<int32_t> h_iota((size_t) nmax * K + 2);
    for (size_t i = 0; i < h_iota.size(); ++i) h_iota[i] = (int32_t) i;
    DevBuf iota(h_iota.size() * 4);
    ck(cudaMemcpy(iota.p, h_iota.data(), h_iota.size() * 4, cudaMemcpyHostToDevice), "iota");
    std::vector<int32_t> h_bounds(257);

    auto product = [&](const Weight& w, const void* xq, const int32_t* bounds, int64_t rows, int64_t max_rows, float* dst,
                       int n_exp) {
        mmq::Product p;
        p.w = w.d.p;
        p.type = w.type;
        p.w_rows = w.ne1;
        p.w_cols = w.ne0;
        p.expert_bytes = w.nb2;
        p.n = n_exp;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = iota.as<int32_t>();
        p.total_rows = rows;
        p.max_rows = max_rows;
        p.dst = dst;
        p.ld_dst = w.ne1;
        return p;
    };

    if (do_moe) {
        if (routing.size() <= (size_t) layer || routing[layer].size() < 96) {
            std::fprintf(stderr, "routing for layer %d missing in %s\n", layer, routing_path);
            return 1;
        }
        // the 96 kept experts: the most-routed of the original 128, in ascending original index (as pruned)
        const std::vector<double>& m = routing[layer];
        std::vector<int> idx(m.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return m[a] > m[b]; });
        idx.resize(96);
        std::sort(idx.begin(), idx.end());
        std::vector<double> mass;   // of kept expert e (0..95)
        for (int i : idx) mass.push_back(m[i]);
        std::vector<int> by_mass(96);
        for (int e = 0; e < 96; ++e) by_mass[e] = e;
        std::sort(by_mass.begin(), by_mass.end(), [&](int a, int b) { return mass[a] > mass[b]; });
        const size_t gu_bytes = (size_t) 96 * 1408 * 2816 / 32 * 18, dn_bytes = gu_bytes / 2;
        const double ctx_mb = 180;

        for (int pass = 0; pass < 2; ++pass)   // pass 0: quantize + gate_up, pass 1: geglu+quantize + down
            for (int n : tokens) {
                const int64_t FE = 704, wrows = pass == 0 ? 2 * FE : D, wcols = pass == 0 ? D : FE;
                // the expert fraction: the biggest phase of this pass must fit the budget
                auto need_mb = [&](int f) {
                    const double rows = (double) n * K / f, wb = (pass == 0 ? gu_bytes : dn_bytes) / (double) f;
                    const double xq = rows * ((wcols + 511) / 512 * 512) / 128 * 144, out = rows * wrows * 4;
                    const double in = pass == 0 ? (double) n * D * 4 : rows * 2 * FE * 4 + rows * FE * 4;
                    return (wb + xq + std::max(in, out)) / 1048576.0 + ctx_mb;
                };
                int f = g_frac;
                if (f <= 0) f = need_mb(1) <= g_vram_mb ? 1 : need_mb(2) <= g_vram_mb ? 2 : 4;
                if (need_mb(f) > g_vram_mb) {
                    std::printf("  n=%d: skipped, ~%.0f MiB > --vram-mb %.0f\n", n, need_mb(f), g_vram_mb);
                    continue;
                }
                const int E = 96 / f, KK = K / f;
                std::vector<int> sel;   // every f-th by routing mass, kept in ascending index order
                for (int r = 0; r < 96; r += f) sel.push_back(by_mass[r]);
                std::sort(sel.begin(), sel.end());
                Weight w = gg.load(blk + (pass == 0 ? "ffn_gate_up_exps.weight" : "ffn_down_exps.weight"), &sel);
                std::vector<double> p_skew, p_uni(E, 1.0 / E);
                double tot = 0;
                for (int e : sel) tot += mass[e];
                for (int e : sel) p_skew.push_back(mass[e] / tot);
                std::printf("\n== MoE %s, n=%d: %s %lld x %lld, %d experts, top-%d%s (~%.0f MiB)\n",
                            pass == 0 ? "gate_up" : "down", n, ggml_type_name(w.type), (long long) w.ne0, (long long) w.ne1,
                            E, KK, f > 1 ? " (every 2nd expert: the same rows per expert, half the work)" : "", need_mb(f));
                for (const auto& dist : dists) {
                    const Routing R = route(n, KK, dist == "skew" ? p_skew : p_uni, 1234 + n);
                    const int64_t rows = (int64_t) n * KK;
                    const std::string key = std::to_string(n) + " " + dist + (f > 1 ? " /" + std::to_string(f) : "");
                    std::printf("  %s: max_rows %d (mean %.1f)\n", dist.c_str(), R.max_rows, (double) rows / E);
                    DevBuf bounds((E + 1) * 4), src(rows * 4), inv(rows * 4);
                    ck(cudaMemcpy(bounds.p, R.bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "bounds");
                    ck(cudaMemcpy(src.p, R.src.data(), rows * 4, cudaMemcpyHostToDevice), "src");
                    ck(cudaMemcpy(inv.p, R.inv.data(), rows * 4, cudaMemcpyHostToDevice), "inv");
                    DevBuf xq(mmq::q8_bytes(rows, wcols));
                    const size_t xq_used = (size_t) rows * ((wcols + 511) / 512 * 512) / 128 * 144;
                    if (pass == 0) {   // quantize: gathered rows (old) vs once per token (new)
                        const auto hx = activations(n, D, 77 + n);
                        DevBuf x((size_t) n * D * 4);
                        ck(cudaMemcpy(x.p, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
                        auto q_old = [&] { mmq::quantize(x.as<float>(), src.as<int32_t>(), xq.p, w.type, D, D, rows, s); };
                        auto q_new = [&] { mmq::quantize_scatter(x.as<float>(), inv.as<int32_t>(), xq.p, w.type, D, D, n, KK, rows, s); };
                        compare("quantize (scatter vs gather)", run_capture(q_old, xq.p, xq_used, s), run_capture(q_new, xq.p, xq_used, s), 1);
                        if (!g_check_only) report("quantize", ab(q_old, q_new, s), "quantize " + key);
                        q_old();
                    } else {           // the down input: geglu of a gate_up-like output, fused vs two kernels
                        const auto hg = activations(rows, 2 * FE, 99 + n);
                        DevBuf gu((size_t) rows * 2 * FE * 4), eh((size_t) rows * FE * 4);
                        ck(cudaMemcpy(gu.p, hg.data(), hg.size() * 4, cudaMemcpyHostToDevice), "gu");
                        auto gq_old = [&] {
                            k::geglu(gu.as<float>(), gu.as<float>() + FE, eh.as<float>(), rows, (int) FE, 2 * FE, s);
                            mmq::quantize(eh.as<float>(), nullptr, xq.p, w.type, FE, FE, rows, s);
                        };
                        auto gq_new = [&] { mmq::geglu_quantize(gu.as<float>(), gu.as<float>() + FE, 2 * FE, xq.p, w.type, FE, rows, s); };
                        compare("geglu+quantize (fused vs two)", run_capture(gq_old, xq.p, xq_used, s), run_capture(gq_new, xq.p, xq_used, s), 1);
                        if (!g_check_only) report("geglu+quantize", ab(gq_old, gq_new, s), "geglu_quantize " + key);
                        gq_old();
                    }
                    const size_t out_bytes = (size_t) rows * w.ne1 * 4;
                    DevBuf out(out_bytes);
                    mmq::Product p = product(w, xq.p, bounds.as<int32_t>(), rows, R.max_rows, out.as<float>(), E);
                    // the old product as layer_big runs it: the bounds to the host, a stream sync, max_rows, the grid
                    auto with_sync = [&] {
                        ck(cudaMemcpyAsync(h_bounds.data(), bounds.p, (E + 1) * 4, cudaMemcpyDeviceToHost, s), "b");
                        ck(cudaStreamSynchronize(s), "b");
                        mmq::Product q = p;
                        q.max_rows = 1;
                        for (int e = 0; e < E; ++e) q.max_rows = std::max<int64_t>(q.max_rows, h_bounds[e + 1] - h_bounds[e]);
                        mq.run(q, s);
                    };
                    auto old_only = [&] { mq.run(p, s); };
                    auto tiles = [&] { mq.run_tiles(p, s); };
                    const char* nm = pass == 0 ? "gate_up" : "down";
                    const auto ref = run_capture(with_sync, out.p, out_bytes, s);
                    auto vars = g_variants;
                    if (vars.empty()) vars.push_back({});
                    for (const auto& var : vars) {
                        std::string desc;
                        for (const auto& [kn, kv] : var) {
                            mmq::knob(kn.c_str(), kv);
                            desc += (desc.empty() ? "" : " ") + kn + "=" + std::to_string(kv);
                        }
                        char lb[96];
                        std::snprintf(lb, sizeof lb, "%s (new%s%s)", nm, desc.empty() ? "" : ": ", desc.c_str());
                        compare(lb, ref, run_capture(tiles, out.p, out_bytes, s));
                        if (!g_check_only) report(lb, ab(old_only, tiles, s), var.empty() ? std::string(nm) + " " + key : "");
                    }
                    if (!g_check_only) report((std::string(nm) + " + bounds sync (old)").c_str(), ab(with_sync, tiles, s),
                                              std::string(nm) + "+sync " + key);
                }
            }
    }

    if (do_dense) {
        std::vector<std::string> names = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                          "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
        std::printf("\n== dense (layer %d)\n", layer);
        for (const auto& nm : names) {
            if (gguf_find_tensor(gg.g, (blk + nm).c_str()) < 0) continue;
            Weight w = gg.load(blk + nm);
            for (int n : tokens) {
                const auto hx = activations(n, w.ne0, 55 + n + w.ne0);
                DevBuf x((size_t) n * w.ne0 * 4), xq(mmq::q8_bytes(n, w.ne0)), out((size_t) n * w.ne1 * 4), b(8);
                ck(cudaMemcpy(x.p, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
                const int32_t hb[2] = {0, n};
                ck(cudaMemcpy(b.p, hb, 8, cudaMemcpyHostToDevice), "b");
                mmq::quantize(x.as<float>(), nullptr, xq.p, w.type, w.ne0, w.ne0, n, s);
                mmq::Product p = product(w, xq.p, b.as<int32_t>(), n, n, out.as<float>(), 1);
                char label[96];
                std::snprintf(label, sizeof label, "%s n=%d (%lldx%lld)", nm.c_str(), n, (long long) w.ne1, (long long) w.ne0);
                const auto ref = run_capture([&] { mq.run(p, s); }, out.p, (size_t) n * w.ne1 * 4, s);
                const auto alt = run_capture([&] { mq.run_dense(p, s); }, out.p, (size_t) n * w.ne1 * 4, s);
                const char* mode = dense_mode(w.ne1, n, nsm);
                compare((std::string(label) + " [" + mode + "]").c_str(), ref, alt);
                if (!g_check_only) {
                    const auto r = ab([&] { mq.run(p, s); }, [&] { mq.run_dense(p, s); }, s);
                    report(label, r, "dense " + nm + " " + std::to_string(n));
                    const double ops = 2.0 * n * w.ne0 * w.ne1 / 1e12;
                    std::printf("      %.1f -> %.1f TOPS\n", ops / (r.first.med * 1e-3), ops / (r.second.med * 1e-3));
                }
            }
        }
    }

    if (!g_sum.empty()) {
        std::printf("\nsummary (median ms per call; one layer):\n");
        for (auto& [k_, v] : g_sum) std::printf("  %-40s old %8.3f  new %8.3f  %+.1f%%\n", k_.c_str(), v.first, v.second, 100.0 * (v.second / v.first - 1.0));
    }
    size_t free1 = 0;
    cudaMemGetInfo(&free1, &total);
    std::printf("\n%d check(s) failed; VRAM in use by this process now ~%.0f MiB (context + buffers still held)\n", g_fail,
                (double) (free0 - free1) / 1048576.0);
    return g_fail ? 1 : 0;
}
