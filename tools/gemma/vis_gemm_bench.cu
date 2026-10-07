// tools/gemma/vis_gemm_bench.cu - the vision encoder's GEMMs: cuBLAS default vs cuBLASLt algorithms, per shape and per
// encoder pass (src/gemma/vision_gemm.cuh, the tuner Vision uses). Build (from the repo root):
//   nvcc -O2 -std=c++20 -arch=sm_86 -Isrc/gemma -o vis_gemm_bench tools/gemma/vis_gemm_bench.cu -lcublasLt -lcublas
// Data: one encoder layer's real matrices as raw files in $GT_DATA (default ./data): wqkv / wo / wgu / wdown / proj
// (bf16, [N][K]), ln1 (f32), patch (f32 [1152][3][16][16]) - extracted from the mmproj with gguf-py.
//   vis_gemm_bench shapes <M,M,..> <f32|bf16|both> <shape,..> [rounds] [exh]   exhaustive study per shape
//   vis_gemm_bench pass <M,M,..> <f32|bf16> [rounds]                            one encoder pass, before vs after
//   vis_gemm_bench agg <M,M,..> <f32|bf16> [rounds]   (GT_FIXED, GT_VARIANTS, GT_DUMP) the same with fixed algorithms,
//                                                     per-position minima dumped so several short runs can be pooled
//   vis_gemm_bench sustain <M,..> <f32|bf16|both> <shape,..> [rounds]          single launches vs back-to-back loops
//   vis_gemm_bench tunecost <Mmax> <f32|bf16>                                   the tuner's one-time cost at load
//   vis_gemm_bench loopcmp <M> <f32|bf16> <shape> [rounds]  (GT_CFGS="id:tile:stages:swizzle;..", GT_LOOP or
//                                                     GT_LOOP_MS, GT_TRACE) the default vs fixed algorithms, single
//                                                     launches vs sustained loops - under the power cap only loops of
//                                                     ~100 ms rank the big GEMMs right (see src/gemma/vision_gemm.cuh)
// (pass: the 27 layers' qkv / attn_out / gate_up / down + the patch GEMM + the projection over M / 9 tokens; bf16 =
// the four layer GEMMs with bf16 output, patch / projection f32.)
// Y [M][N] (f32 | bf16) = X [M][K] . W [N][K]^T, bf16 (or f16 for the patch conv) inputs, FP32 accumulation.
// cuBLAS default (cublasGemmEx, what vision.cu calls) vs every cuBLASLt candidate: the heuristic's list, plus its
// (algo, tile, stages) combinations re-run with split-K 1..4 and CTA swizzling 0/1, plus (mode exh) every algo id x tile
// x stages. Timings: min over rounds, candidates interleaved and rotated (production traffic shares the card).
// Real weights of encoder layer 13 (data/*.bin, extracted from the mmproj), realistic activations (rms * ln, geglu).
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>
#include "vision_gemm.cuh"
#include <chrono>
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#define CK(x) do { auto e_ = (x); if ((int) e_ != 0) { fprintf(stderr, "error %d: %s line %d\n", (int) e_, #x, __LINE__); exit(1); } } while (0)

using bf16 = __nv_bfloat16;
static cublasHandle_t g_h;
static cublasLtHandle_t g_lt;
static void* g_ws;
static size_t g_wsz = 32u << 20;
static std::string g_dir = "data";

static std::vector<uint8_t> readf(const std::string& n) {
    FILE* f = fopen((g_dir + "/" + n).c_str(), "rb");
    if (!f) { fprintf(stderr, "no %s\n", n.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(sz); if (fread(v.data(), 1, sz, f) != sz) exit(1); fclose(f); return v;
}
template <class T> static T* upload(const void* p, size_t bytes) { T* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, p, bytes, cudaMemcpyHostToDevice)); return d; }

struct Cfg {
    int id = -1, tile = 0, stages = 0, splitk = 1, red = 0, swz = 0, custom = 0, inner = 0, cluster = 0;
    bool operator<(const Cfg& o) const { return std::memcmp(this, &o, sizeof(Cfg)) < 0; }
};
static Cfg cfg_of(const cublasLtMatmulAlgo_t& a) {
    Cfg c; size_t w;
    uint32_t u; uint16_t s;
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_ID, &c.id, 4, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &u, 4, &w); c.tile = u;
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &u, 4, &w); c.stages = u;
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &u, 4, &w); c.splitk = u;
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &u, 4, &w); c.red = u;
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &u, 4, &w); c.swz = u;
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &u, 4, &w); c.custom = u;
    s = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &s, 2, &w); c.inner = s;
    s = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &s, 2, &w); c.cluster = s;
    return c;
}
static const char* tile_name(int t) {
    static const char* n[] = {"undef", "8x8", "8x16", "16x8", "8x32", "16x16", "32x8", "8x64", "16x32", "32x16", "64x8",
                              "32x32", "32x64", "64x32", "32x128", "64x64", "128x32", "64x128", "128x64", "64x256",
                              "128x128", "256x64", "64x512", "128x256", "256x128", "512x64", "64x96", "96x64", "96x128",
                              "128x160", "160x128", "192x128", "128x192", "128x96", "32x256", "256x32"};
    static char buf[16];
    if (t >= 0 && t < (int) (sizeof(n) / sizeof(*n))) return n[t];
    snprintf(buf, sizeof buf, "t%d", t); return buf;
}
static std::string stages_name(int s) {
    char b[16];
    if (s >= 1 && s <= 24) snprintf(b, sizeof b, "%dx%d", 16 << ((s - 1) / 6), (s - 1) % 6 + 1);
    else if (s == 25) return "32x10"; else if (s == 26) return "8x4"; else if (s == 27) return "16x10";
    else if (s == 28) return "8x5"; else if (s == 31) return "8x3"; else snprintf(b, sizeof b, "s%d", s);
    return b;
}
static std::string cfg_str(const Cfg& c) {
    if (c.id < 0) return "cublasGemmEx default";
    char b[128];
    snprintf(b, sizeof b, "a%d %s st%s sk%d r%d sw%d c%d", c.id, tile_name(c.tile), stages_name(c.stages).c_str(), c.splitk,
             c.red, c.swz, c.custom);
    return b;
}

struct Case {
    const char* name; int M, N, K; cudaDataType at, ct; const void* W; const void* X; void* Y;
};

struct LtDesc {
    cublasLtMatmulDesc_t op = nullptr; cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    void make(const Case& c) {
        CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        CK(cublasLtMatrixLayoutCreate(&la, c.at, c.K, c.N, c.K));
        CK(cublasLtMatrixLayoutCreate(&lb, c.at, c.K, c.M, c.K));
        CK(cublasLtMatrixLayoutCreate(&lc, c.ct, c.N, c.M, c.N));
    }
    void destroy() { cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc); }
};

static const float kOne = 1.f, kZero = 0.f;
static void run_default(const Case& c) {
    CK(cublasGemmEx(g_h, CUBLAS_OP_T, CUBLAS_OP_N, c.N, c.M, c.K, &kOne, c.W, c.at, c.K, c.X, c.at, c.K, &kZero, c.Y, c.ct,
                    c.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
static void run_lt(const Case& c, const LtDesc& d, const cublasLtMatmulAlgo_t& a) {
    CK(cublasLtMatmul(g_lt, d.op, &kOne, c.W, d.la, c.X, d.lb, &kZero, c.Y, d.lc, c.Y, d.lc, &a, g_ws, g_wsz, 0));
}

struct Cand { Cfg cfg; cublasLtMatmulAlgo_t algo; bool def = false; float best = 1e9f; int heur = -1; };

static void time_cands(const Case& c, const LtDesc& d, std::vector<Cand*>& cs, int rounds) {
    const int n = (int) cs.size();
    std::vector<cudaEvent_t> e(2 * n);
    for (auto& x : e) CK(cudaEventCreate(&x));
    for (int r = -1; r < rounds; ++r) {
        for (int j = 0; j < n; ++j) {
            const int i = (j + std::max(r, 0) * 7) % n;
            CK(cudaEventRecord(e[2 * i]));
            if (cs[i]->def) run_default(c); else run_lt(c, d, cs[i]->algo);
            CK(cudaEventRecord(e[2 * i + 1]));
        }
        CK(cudaDeviceSynchronize());
        if (r < 0) continue;   // warm-up round
        for (int i = 0; i < n; ++i) { float ms; CK(cudaEventElapsedTime(&ms, e[2 * i], e[2 * i + 1])); cs[i]->best = std::min(cs[i]->best, ms); }
    }
    for (auto& x : e) cudaEventDestroy(x);
}

// all candidates: heuristic list + split-K / swizzle variants of its (algo, tile, stages) + (exh) every id x tile x stages
static std::vector<Cand> candidates(const Case& c, const LtDesc& d, bool exh, int* n_heur) {
    std::vector<Cand> out;
    std::map<Cfg, int> seen;
    auto add = [&](const cublasLtMatmulAlgo_t& a, int heur) {
        cublasLtMatmulHeuristicResult_t r;
        if (cublasLtMatmulAlgoCheck(g_lt, d.op, d.la, d.lb, d.lc, d.lc, &a, &r) != CUBLAS_STATUS_SUCCESS) return;
        if (r.workspaceSize > g_wsz) return;
        Cfg cf = cfg_of(a);
        if (seen.count(cf)) return;
        seen[cf] = (int) out.size();
        Cand x; x.cfg = cf; x.algo = a; x.heur = heur; out.push_back(x);
    };
    cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
    CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_wsz, sizeof(g_wsz)));
    cublasLtMatmulHeuristicResult_t res[64]; int n = 0;
    CK(cublasLtMatmulAlgoGetHeuristic(g_lt, d.op, d.la, d.lb, d.lc, d.lc, pref, 64, res, &n));
    cublasLtMatmulPreferenceDestroy(pref);
    for (int i = 0; i < n; ++i) add(res[i].algo, i);
    *n_heur = (int) out.size();
    // variants
    std::vector<Cand> base = out;
    for (const Cand& b : base) {
        for (int sk : {1, 2, 3, 4, 6, 8})
            for (int sw : {0, 1})
                for (int red : {0, 1, 2, 4}) {
                    if ((sk == 1) != (red == 0)) continue;
                    cublasLtMatmulAlgo_t a = b.algo;
                    uint32_t u = sk; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &u, 4);
                    u = red; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &u, 4);
                    u = sw; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &u, 4);
                    add(a, -1);
                }
    }
    if (exh) {
        int ids[128], nid = 0;
        CK(cublasLtMatmulAlgoGetIds(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, c.at, c.at, c.ct, c.ct, 128, ids, &nid));
        for (int k = 0; k < nid; ++k) {
            cublasLtMatmulAlgo_t a;
            if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, c.at, c.at, c.ct, c.ct, ids[k], &a)) continue;
            size_t w = 0;
            cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &w);
            std::vector<uint32_t> tiles(w / 4);
            if (w) cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), w, &w);
            if (tiles.empty()) tiles.push_back(0);
            cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &w);
            std::vector<uint32_t> st(w / 4);
            if (w) cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_STAGES_IDS, st.data(), w, &w);
            if (st.empty()) st.push_back(0);
            int32_t cmax = 0; cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &cmax, 4, &w);
            for (uint32_t t : tiles)
                for (uint32_t s : st)
                    for (int co = 0; co <= std::min(cmax, 3); ++co)
                        for (int sw : {0, 1}) {
                            cublasLtMatmulAlgo_t b = a;
                            cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_TILE_ID, &t, 4);
                            cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_STAGES_ID, &s, 4);
                            uint32_t u = co; cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &u, 4);
                            u = sw; cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &u, 4);
                            add(b, -2);
                        }
        }
    }
    return out;
}

static void compare(const Case& c, std::vector<float>& ref, std::vector<float>& got, double* rel_rms, double* rel_max, size_t* ndiff) {
    double ss = 0, mx = 0, dmax = 0; size_t nd = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        ss += (double) ref[i] * ref[i]; mx = std::max(mx, (double) std::fabs(ref[i]));
        double dd = std::fabs((double) ref[i] - got[i]); dmax = std::max(dmax, dd); nd += dd != 0;
    }
    *rel_rms = dmax / std::sqrt(ss / ref.size()); *rel_max = dmax / mx; *ndiff = nd;
}
static void fetch(const Case& c, std::vector<float>& v) {
    const size_t n = (size_t) c.M * c.N;
    v.resize(n);
    if (c.ct == CUDA_R_32F) { CK(cudaMemcpy(v.data(), c.Y, n * 4, cudaMemcpyDeviceToHost)); return; }
    std::vector<bf16> b(n); CK(cudaMemcpy(b.data(), c.Y, n * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) v[i] = __bfloat162float(b[i]);
}

// probe configs (id, tile, stages, splitk, swizzle): always timed in the final stage, so cases can be compared
struct Probe { int id, tile, stages, sk, swz; const char* tag; };
static const Probe kProbes[] = {
    {21, 20, 11, 1, 0, "P1"}, {21, 20, 11, 1, 1, "P2"}, {5, 20, 0, 1, 1, "P3"}, {5, 20, 0, 1, 0, "P4"},
    {6, 20, 15, 1, 1, "P5"}, {21, 23, 9, 1, 0, "P6"}, {21, 24, 9, 1, 0, "P7"}, {21, 20, 10, 1, 1, "P8"},
};
static bool probe_match(const Cfg& c, const Probe& p) {
    return c.id == p.id && c.tile == p.tile && c.stages == p.stages && c.splitk == p.sk && c.swz == p.swz && c.custom == 0;
}

// one case: successive halving (all candidates 2 rounds -> best 24: 8 rounds -> best 6 + probes + heuristic #0: rounds)
static void tune_case(const Case& c, bool exh, int rounds, bool verbose, bool numerics) {
    LtDesc d; d.make(c);
    int nh = 0;
    const double t0 = now_s();
    std::vector<Cand> cs = candidates(c, d, exh, &nh);
    const double t1 = now_s();
    Cand def; def.def = true;
    auto stage = [&](std::vector<Cand*> in, int r, size_t keep) {
        std::vector<Cand*> run; run.push_back(&def);
        for (auto* x : in) { x->best = 1e9f; run.push_back(x); }
        def.best = 1e9f;
        time_cands(c, d, run, r);
        std::sort(in.begin(), in.end(), [](Cand* a, Cand* b) { return a->best < b->best; });
        if (in.size() > keep) in.resize(keep);
        return in;
    };
    std::vector<Cand*> all; for (auto& x : cs) all.push_back(&x);
    std::vector<Cand*> top = stage(all, 2, 24);
    top = stage(top, 8, 6);
    for (auto& x : cs) {
        bool want = x.heur == 0;
        for (const Probe& p : kProbes) want |= probe_match(x.cfg, p);
        if (want && std::find(top.begin(), top.end(), &x) == top.end()) top.push_back(&x);
    }
    const double t2 = now_s();
    std::vector<Cand*> fin = stage(top, rounds, 64);
    const double t3 = now_s();
    fprintf(stderr, "  [%s: candidates %.2f s, screening %.2f s, final %.2f s]\n", c.name, t1 - t0, t2 - t1, t3 - t2);
    Cand* best = fin[0];
    Cand* h0 = nullptr; for (auto* x : fin) if (x->heur == 0) h0 = x;
    const double f = 2.0 * c.M * c.N * c.K;
    double rr = 0, rm = 0; size_t nd = 0;
    if (numerics) {
        std::vector<float> ref, got;
        run_default(c); CK(cudaDeviceSynchronize()); fetch(c, ref);
        run_lt(c, d, best->algo); CK(cudaDeviceSynchronize()); fetch(c, got);
        compare(c, ref, got, &rr, &rm, &nd);
    }
    printf("%-9s M=%5d N=%5d K=%5d %s | default %7.3f ms %5.1f TF | heur#0 %7.3f ms %5.1f TF | best %7.3f ms %5.1f TF %+5.1f%% [%s]%s  (%zu cands, %d heur)",
           c.name, c.M, c.N, c.K, c.ct == CUDA_R_32F ? "f32 " : "bf16", def.best, f / def.best / 1e9, h0 ? h0->best : 0.f,
           h0 ? f / h0->best / 1e9 : 0.0, best->best, f / best->best / 1e9, 100.0 * (def.best / best->best - 1),
           cfg_str(best->cfg).c_str(), best->heur >= 0 ? " (heur)" : best->heur == -1 ? " (variant)" : " (exh)", cs.size(), nh);
    if (numerics) printf(" | max|d|/rms %.2e max|d|/max %.2e ndiff %zu", rr, rm, nd);
    printf("\n");
    // machine-readable: case, default, best, probes (relative to default: >1 = faster)
    printf("ROW %s %d %d %d %s def=%.4f best=%.4f h0=%.4f", c.name, c.M, c.N, c.K, c.ct == CUDA_R_32F ? "f32" : "bf16", def.best,
           best->best, h0 ? h0->best : 0.f);
    for (const Probe& p : kProbes) {
        float t = 0;
        for (auto* x : fin) if (probe_match(x->cfg, p)) t = x->best;
        printf(" %s=%.4f", p.tag, t);
    }
    printf(" win=[%s]\n", cfg_str(best->cfg).c_str());
    if (verbose)
        for (auto* x : fin) printf("      %7.3f ms  %s%s\n", x->best, cfg_str(x->cfg).c_str(), x->heur >= 0 ? " (heur)" : "");
    fflush(stdout);
    d.destroy();
}

__global__ void geglu_k(const float* gu, bf16* h, int64_t n, int F) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * F) return;
    const int64_t r = i / F, c = i % F;
    const float g = gu[r * 2 * F + c], u = gu[r * 2 * F + F + c];
    h[i] = __float2bfloat16(g * (1.0f / (1.0f + expf(-1.702f * g))) * u);
}

int main(int argc, char** argv) {
    // usage: gtune <mode: shapes|list> [Ms comma] [out: f32|bf16|both] [shapes comma] [rounds] [exh]
    std::string mode = argc > 1 ? argv[1] : "shapes";
    std::vector<int> Ms = {4959};
    if (argc > 2) { Ms.clear(); for (char* p = strtok(argv[2], ","); p; p = strtok(nullptr, ",")) Ms.push_back(atoi(p)); }
    std::string outs = argc > 3 ? argv[3] : "both";
    std::string shp = argc > 4 ? argv[4] : "qkv,attn_out,gate_up,down";
    int rounds = argc > 5 ? atoi(argv[5]) : 25;
    if (mode == "pass" || mode == "agg" || mode == "tunecost") { shp = "patch,qkv,attn_out,gate_up,down,proj"; rounds = argc > 4 ? atoi(argv[4]) : 20; }
    bool exh = argc > 6 && std::string(argv[6]) == "exh";
    if (const char* e = getenv("GT_DATA")) g_dir = e;
    const int D = 1152, F = 4304, Mmax = *std::max_element(Ms.begin(), Ms.end());
    CK(cublasCreate(&g_h)); CK(cublasLtCreate(&g_lt)); CK(cudaMalloc(&g_ws, g_wsz));
    CK(cublasSetWorkspace(g_h, g_ws, g_wsz));
    auto has = [&](const char* s) { return ("," + shp + ",").find(std::string(",") + s + ",") != std::string::npos; };
    std::vector<cudaDataType> cts;
    if (outs == "f32" || outs == "both") cts.push_back(CUDA_R_32F);
    if (outs == "bf16" || outs == "both") cts.push_back(CUDA_R_16BF);
    const size_t yel = cts.size() == 1 && cts[0] == CUDA_R_16BF ? 2 : 4;
    // allocate only what the requested shapes need (VRAM is shared with production)
    size_t ybytes = 0;
    for (auto [nm, n] : {std::pair{"patch", D}, {"qkv", 3 * D}, {"attn_out", D}, {"gate_up", 2 * F}, {"down", D}, {"proj", 2816}})
        if (has(nm)) ybytes = std::max(ybytes, (size_t) Mmax * n * (std::string(nm) == "patch" ? 4 : yel));
    const bool need_gu = has("gate_up") || has("down"), need_hb = has("qkv") || has("attn_out") || need_gu || has("proj");
    auto up_w = [&](const char* f, bool need) -> bf16* { if (!need) return nullptr; auto v = readf(f); return upload<bf16>(v.data(), v.size()); };
    bf16 *d_qkv = up_w("wqkv.bin", has("qkv")), *d_o = up_w("wo.bin", has("attn_out")), *d_gu = up_w("wgu.bin", need_gu),
         *d_down = up_w("wdown.bin", has("down")), *d_proj = up_w("proj.bin", has("proj"));
    __half* d_patch = nullptr;
    if (has("patch")) {
        auto patch = readf("patch.bin");
        std::vector<__half> ph(patch.size() / 4);
        for (size_t i = 0; i < ph.size(); ++i) ph[i] = __float2half(reinterpret_cast<const float*>(patch.data())[i]);
        d_patch = upload<__half>(ph.data(), ph.size() * 2);
    }
    // activations: hb = rms(randn) * ln1 (bf16) [Mmax][D]; fb = geglu(hb . Wgu^T) [Mmax][F]; patches U(-1,1)
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f, 1.f);
    bf16* d_hb = nullptr;
    if (need_hb) {
        auto ln1 = readf("ln1.bin");
        std::vector<bf16> hb((size_t) Mmax * D);
        const float* l1 = reinterpret_cast<const float*>(ln1.data());
        for (int r = 0; r < Mmax; ++r) {
            std::vector<float> v(D); double ss = 0;
            for (int i = 0; i < D; ++i) { v[i] = nd(rng); ss += v[i] * v[i]; }
            const float sc = 1.f / std::sqrt(ss / D + 1e-6);
            for (int i = 0; i < D; ++i) hb[(size_t) r * D + i] = __float2bfloat16(v[i] * sc * l1[i]);
        }
        d_hb = upload<bf16>(hb.data(), hb.size() * 2);
    }
    bf16* d_ab = d_hb;   // attn_out's input: the same rms-scale rows (saves VRAM)
    void* d_y; CK(cudaMalloc(&d_y, ybytes));
    bf16* d_fb = nullptr;
    if (has("down")) {   // gate_up in row chunks that fit d_y, geglu each
        CK(cudaMalloc(&d_fb, (size_t) Mmax * F * 2));
        const int chunk = (int) std::min<size_t>(Mmax, ybytes / ((size_t) 2 * F * 4));
        for (int r0 = 0; r0 < Mmax; r0 += chunk) {
            const int rows = std::min(chunk, Mmax - r0);
            Case c{"gu", rows, 2 * F, D, CUDA_R_16BF, CUDA_R_32F, d_gu, d_hb + (size_t) r0 * D, d_y};
            run_default(c);
            geglu_k<<<(unsigned) (((int64_t) rows * F + 255) / 256), 256>>>((const float*) d_y, d_fb + (size_t) r0 * F, rows, F);
        }
        CK(cudaDeviceSynchronize());
        if (!has("gate_up")) { cudaFree(d_gu); d_gu = nullptr; }
    }
    __half* d_pt = nullptr;
    if (has("patch")) {
        std::vector<__half> pt((size_t) Mmax * 768);
        std::uniform_real_distribution<float> ud(-1.f, 1.f);
        for (auto& x : pt) x = __float2half(ud(rng));
        d_pt = upload<__half>(pt.data(), pt.size() * 2);
    }
    size_t fr, tot; cudaMemGetInfo(&fr, &tot);
    printf("# device free %.0f MiB after allocations; workspace %zu MiB; rounds %d%s\n", fr / 1048576.0, g_wsz >> 20, rounds, exh ? " exhaustive" : "");
    if (mode == "shapes") {
        for (int M : Ms)
            for (cudaDataType ct : cts) {
                if (has("patch") && ct == CUDA_R_32F) tune_case({"patch", M, D, 768, CUDA_R_16F, ct, d_patch, d_pt, d_y}, exh, rounds, false, true);
                if (has("qkv")) tune_case({"qkv", M, 3 * D, D, CUDA_R_16BF, ct, d_qkv, d_hb, d_y}, exh, rounds, false, true);
                if (has("attn_out")) tune_case({"attn_out", M, D, D, CUDA_R_16BF, ct, d_o, d_ab, d_y}, exh, rounds, false, true);
                if (has("gate_up")) tune_case({"gate_up", M, 2 * F, D, CUDA_R_16BF, ct, d_gu, d_hb, d_y}, exh, rounds, false, true);
                if (has("down")) tune_case({"down", M, D, F, CUDA_R_16BF, ct, d_down, d_fb, d_y}, exh, rounds, false, true);
                if (has("proj")) tune_case({"proj", M / 9, 2816, D, CUDA_R_16BF, ct, d_proj, d_hb, d_y}, exh, rounds, false, true);
            }
    } else if (mode == "pass") {
        // the same buffers for every layer (VRAM): one output buffer for all GEMMs, inputs hb / fb / patches
        const int nl = 27;
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cublasSetStream(g_h, st));
        for (int M : Ms) {
            const cudaDataType lt = outs == "bf16" ? CUDA_R_16BF : CUDA_R_32F;   // the four layer GEMMs' output
            const int T = M / 9;
            struct G { const char* nm; const void* W; const void* X; int M, N, K; cudaDataType ab, c; };
            std::vector<G> seq;
            seq.push_back({"patch", d_patch, d_pt, M, D, 768, CUDA_R_16F, CUDA_R_32F});
            for (int l = 0; l < nl; ++l) {
                seq.push_back({"qkv", d_qkv, d_hb, M, 3 * D, D, CUDA_R_16BF, lt});
                seq.push_back({"attn_out", d_o, d_ab, M, D, D, CUDA_R_16BF, lt});
                seq.push_back({"gate_up", d_gu, d_hb, M, 2 * F, D, CUDA_R_16BF, lt});
                seq.push_back({"down", d_down, d_fb, M, D, F, CUDA_R_16BF, lt});
            }
            seq.push_back({"proj", d_proj, d_hb, T, 2816, D, CUDA_R_16BF, CUDA_R_32F});
            strata::gemma::VisionGemm vg;
            vg.init(g_h, st);
            auto pass_vg = [&] { for (const G& g : seq) vg.run(g.W, g.ab, g.X, g.ab, d_y, g.c, g.M, g.N, g.K, g.nm); };
            vg.tuning = true;
            const double t0 = now_s();
            pass_vg();
            CK(cudaStreamSynchronize(st));
            const double tt = now_s() - t0;
            vg.tuning = false;
            // numerics: every distinct GEMM (patch, qkv, attn_out, gate_up, down, projection), default vs tuned output
            for (size_t i : {(size_t) 0, (size_t) 1, (size_t) 2, (size_t) 3, (size_t) 4, seq.size() - 1}) {
                const G& g = seq[i];
                Case c{g.nm, g.M, g.N, g.K, g.ab, g.c, g.W, g.X, d_y};
                std::vector<float> ref, got;
                run_default(c); CK(cudaStreamSynchronize(st)); fetch(c, ref);
                vg.run(g.W, g.ab, g.X, g.ab, d_y, g.c, g.M, g.N, g.K, g.nm); CK(cudaStreamSynchronize(st)); fetch(c, got);
                double rr, rm; size_t nd;
                compare(c, ref, got, &rr, &rm, &nd);
                printf("  numerics %-8s M=%5d %s: max|d|/rms %.2e, max|d|/max|y| %.2e, %zu of %zu elements differ\n", g.nm, g.M,
                       g.c == CUDA_R_32F ? "f32 " : "bf16", rr, rm, nd, ref.size());
            }
            // timing: passes of every variant in turn each round (all default; the tuner for every GEMM; the tuner for
            // one kind only - under the power cap a faster kernel can slow its neighbours), an event between every two
            // GEMMs. Production's kernels time-slice with ours, so a whole pass is never clean: the per-position minimum
            // over the rounds, summed ("clean"), is the pass without the interruptions.
            const size_t ns = seq.size();
            std::vector<std::string> variants = {"default", "tuned"};
            if (const char* e = getenv("GT_VARIANTS")) for (char* q = strtok(strdup(e), ","); q; q = strtok(nullptr, ",")) variants.push_back(q);
            const size_t nv = variants.size();
            std::vector<cudaEvent_t> ev(nv * (ns + 1));
            for (auto& x : ev) CK(cudaEventCreate(&x));
            std::vector<std::vector<float>> mn(nv, std::vector<float>(ns, 1e9f));
            std::vector<std::vector<std::vector<float>>> smp(nv, std::vector<std::vector<float>>(ns));   // [variant][position][round]
            std::vector<float> whole(nv, 1e9f);
            auto pass_ev = [&](size_t v, cudaEvent_t* E) {
                for (size_t i = 0; i < ns; ++i) {
                    CK(cudaEventRecord(E[i], st));
                    const G& g = seq[i];
                    if (v == 1 || (v > 1 && variants[v] == g.nm)) vg.run(g.W, g.ab, g.X, g.ab, d_y, g.c, g.M, g.N, g.K, g.nm);
                    else { Case c{g.nm, g.M, g.N, g.K, g.ab, g.c, g.W, g.X, d_y}; run_default(c); }
                }
                CK(cudaEventRecord(E[ns], st));
            };
            for (int r = -1; r < rounds; ++r) {
                for (size_t j = 0; j < nv; ++j) { const size_t v = (j + std::max(r, 0)) % nv; pass_ev(v, &ev[v * (ns + 1)]); }
                CK(cudaStreamSynchronize(st));
                if (r < 0) continue;
                for (size_t v = 0; v < nv; ++v) {
                    cudaEvent_t* E = &ev[v * (ns + 1)];
                    float a;
                    CK(cudaEventElapsedTime(&a, E[0], E[ns])); whole[v] = std::min(whole[v], a);
                    for (size_t i = 0; i < ns; ++i) { CK(cudaEventElapsedTime(&a, E[i], E[i + 1])); mn[v][i] = std::min(mn[v][i], a); smp[v][i].push_back(a); }
                }
            }
            for (auto& x : ev) cudaEventDestroy(x);
            std::vector<double> clean(nv, 0);
            std::vector<std::map<std::string, double>> kind(nv);
            for (size_t v = 0; v < nv; ++v)
                for (size_t i = 0; i < ns; ++i) { clean[v] += mn[v][i]; kind[v][seq[i].nm] += mn[v][i]; }
            printf("PASS M=%d (proj %d rows) layer-GEMM output %s: clean default %.2f ms, tuned %.2f ms: %.2f ms saved (%.1f%%) | "
                   "whole-pass min: default %.2f, tuned %.2f ms | tuning %.0f ms wall (%d shapes, %.0f ms timing)\n", M, T,
                   outs.c_str(), clean[0], clean[1], clean[0] - clean[1], 100.0 * (clean[0] - clean[1]) / clean[0], whole[0],
                   whole[1], tt * 1e3, vg.n_tuned, vg.tune_ms);
            // paired: per position the median over the rounds of (variant - default) in the same round, summed - slow
            // clock drifts cancel within a round, time-slice interruptions are outliers the median ignores
            auto median = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.f : v[v.size() / 2]; };
            std::vector<double> paired(nv, 0), med(nv, 0);
            for (size_t v = 0; v < nv; ++v)
                for (size_t i = 0; i < ns; ++i) {
                    std::vector<float> d(smp[v][i].size());
                    for (size_t r = 0; r < d.size(); ++r) d[r] = smp[v][i][r] - smp[0][i][r];
                    paired[v] += median(d);
                    med[v] += median(smp[v][i]);
                }
            printf("    paired median difference vs default (ms, negative = faster); median-sum pass time:\n");
            for (size_t v = 1; v < nv; ++v)
                printf("      %-9s %+7.2f ms (%+.1f%%)   median-sum %.2f vs default %.2f ms\n", variants[v].c_str(), paired[v],
                       100.0 * paired[v] / med[0], med[v], med[0]);
            printf("    per kind (clean sums, ms):   ");
            for (auto& [k, x] : kind[0]) printf(" %9s", k.c_str());
            printf("     total\n");
            for (size_t v = 0; v < nv; ++v) {
                printf("    %-9s%s", variants[v].c_str(), v > 1 ? " tuned only  " : "             ");
                for (auto& [k, x] : kind[v]) printf(" %9.2f", x);
                printf("  %8.2f (%+.2f ms)\n", clean[v], clean[v] - clean[0]);
            }
            printf("%s", vg.summary().c_str());
            fflush(stdout);
        }
    } else if (mode == "sustain") {
        // one shape: the tuner's choice vs the default, (a) single launches with a sync after each, (b) loops of L
        // launches back to back (sustained, the power cap reacts) - min and median over alternating repetitions
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cublasSetStream(g_h, st));
        const int L = getenv("GT_LOOP") ? atoi(getenv("GT_LOOP")) : 40;
        for (int M : Ms)
            for (cudaDataType ct : cts) {
                struct S { const char* nm; const void* W; const void* X; int N, K; cudaDataType ab; };
                std::vector<S> ss;
                if (has("patch") && ct == CUDA_R_32F) ss.push_back({"patch", d_patch, d_pt, D, 768, CUDA_R_16F});
                if (has("qkv")) ss.push_back({"qkv", d_qkv, d_hb, 3 * D, D, CUDA_R_16BF});
                if (has("attn_out")) ss.push_back({"attn_out", d_o, d_ab, D, D, CUDA_R_16BF});
                if (has("gate_up")) ss.push_back({"gate_up", d_gu, d_hb, 2 * F, D, CUDA_R_16BF});
                if (has("down")) ss.push_back({"down", d_down, d_fb, D, F, CUDA_R_16BF});
                for (const S& x : ss) {
                    strata::gemma::VisionGemm vg;
                    vg.init(g_h, st);
                    vg.tuning = true;
                    vg.run(x.W, x.ab, x.X, x.ab, d_y, ct, M, x.N, x.K, x.nm);
                    vg.tuning = false;
                    Case c{x.nm, M, x.N, x.K, x.ab, ct, x.W, x.X, d_y};
                    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                    std::vector<float> s1[2], sl[2];
                    for (int r = -1; r < rounds; ++r)
                        for (int k = 0; k < 2; ++k) {
                            const int v = (k + r) & 1;   // 0 default, 1 tuned
                            auto one = [&] { if (v) vg.run(x.W, x.ab, x.X, x.ab, d_y, ct, M, x.N, x.K, x.nm); else run_default(c); };
                            CK(cudaEventRecord(e0, st)); one(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
                            float a; CK(cudaEventElapsedTime(&a, e0, e1)); if (r >= 0) s1[v].push_back(a);
                            CK(cudaEventRecord(e0, st)); for (int l = 0; l < L; ++l) one(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
                            CK(cudaEventElapsedTime(&a, e0, e1)); if (r >= 0) sl[v].push_back(a / L);
                        }
                    if (getenv("GT_TRACE")) {   // per launch index in a loop of 60: how fast the power cap reacts
                        const int TL = 60;
                        std::vector<cudaEvent_t> te(TL + 1);
                        for (auto& x : te) CK(cudaEventCreate(&x));
                        std::vector<std::vector<float>> tr[2];
                        for (int v = 0; v < 2; ++v) tr[v].assign(TL, {});
                        for (int r = 0; r < 8; ++r)
                            for (int v = 0; v < 2; ++v) {
                                CK(cudaStreamSynchronize(st));
                                usleep(30000);   // let the GPU (our context) idle first
                                for (int l = 0; l < TL; ++l) {
                                    CK(cudaEventRecord(te[l], st));
                                    if (v) vg.run(x.W, x.ab, x.X, x.ab, d_y, ct, M, x.N, x.K, x.nm); else run_default(c);
                                }
                                CK(cudaEventRecord(te[TL], st)); CK(cudaStreamSynchronize(st));
                                for (int l = 0; l < TL; ++l) { float a; CK(cudaEventElapsedTime(&a, te[l], te[l + 1])); tr[v][l].push_back(a); }
                            }
                        auto md = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
                        printf("TRACE %s M=%d: launch index: default median / tuned median (ms)\n", x.nm, M);
                        for (int l = 0; l < TL; l += (l < 10 ? 1 : 5)) printf("  %2d: %.3f / %.3f\n", l, md(tr[0][l]), md(tr[1][l]));
                        for (auto& q : te) cudaEventDestroy(q);
                    }
                    auto mnf = [](std::vector<float> v) { return *std::min_element(v.begin(), v.end()); };
                    auto mdf = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
                    printf("SUST %-8s M=%5d %s | single: default min %.3f med %.3f, tuned min %.3f med %.3f (%+.1f%% / %+.1f%%) | "
                           "loop of %d: default min %.3f med %.3f, tuned min %.3f med %.3f (%+.1f%% / %+.1f%%)\n",
                           x.nm, M, ct == CUDA_R_32F ? "f32 " : "bf16", mnf(s1[0]), mdf(s1[0]), mnf(s1[1]), mdf(s1[1]),
                           100.0 * (mnf(s1[0]) / mnf(s1[1]) - 1), 100.0 * (mdf(s1[0]) / mdf(s1[1]) - 1), L, mnf(sl[0]), mdf(sl[0]),
                           mnf(sl[1]), mdf(sl[1]), 100.0 * (mnf(sl[0]) / mnf(sl[1]) - 1), 100.0 * (mdf(sl[0]) / mdf(sl[1]) - 1));
                    printf("%s", vg.summary().c_str());
                    fflush(stdout);
                    cudaEventDestroy(e0); cudaEventDestroy(e1);
                }
            }
    } else if (mode == "agg") {
        // fixed algorithms per GEMM kind (GT_FIXED="qkv=21:20:11:1,gate_up=21:20:11:1,..." = algo id : tile : stages :
        // swizzle), passes of variants in turn (default; all fixed; GT_VARIANTS: one kind fixed only), per-position
        // minima appended to GT_DUMP ("<M> <variant> <position> <kind> <ms>") so several short runs can be pooled
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cublasSetStream(g_h, st));
        std::map<std::string, Probe> fixed;
        if (const char* e = getenv("GT_FIXED"))
            for (char* q = strtok(strdup(e), ","); q; q = strtok(nullptr, ",")) {
                char nm[32]; int id, t, stg, sw;
                if (sscanf(q, "%31[^=]=%d:%d:%d:%d", nm, &id, &t, &stg, &sw) == 5) fixed[nm] = {id, t, stg, 1, sw, ""};
            }
        FILE* dump = getenv("GT_DUMP") ? fopen(getenv("GT_DUMP"), "a") : nullptr;
        for (int M : Ms) {
            const cudaDataType lt = outs == "bf16" ? CUDA_R_16BF : CUDA_R_32F;
            const int T = M / 9;
            struct G { const char* nm; const void* W; const void* X; int M, N, K; cudaDataType ab, c; };
            std::vector<G> seq;
            seq.push_back({"patch", d_patch, d_pt, M, D, 768, CUDA_R_16F, CUDA_R_32F});
            for (int l = 0; l < 27; ++l) {
                seq.push_back({"qkv", d_qkv, d_hb, M, 3 * D, D, CUDA_R_16BF, lt});
                seq.push_back({"attn_out", d_o, d_ab, M, D, D, CUDA_R_16BF, lt});
                seq.push_back({"gate_up", d_gu, d_hb, M, 2 * F, D, CUDA_R_16BF, lt});
                seq.push_back({"down", d_down, d_fb, M, D, F, CUDA_R_16BF, lt});
            }
            seq.push_back({"proj", d_proj, d_hb, T, 2816, D, CUDA_R_16BF, CUDA_R_32F});
            // one Lt plan per kind with a fixed algorithm
            std::map<std::string, std::pair<LtDesc, cublasLtMatmulAlgo_t>> plans;
            for (const G& g : seq) {
                if (plans.count(g.nm) || !fixed.count(g.nm)) continue;
                const Probe& f = fixed[g.nm];
                Case c{g.nm, g.M, g.N, g.K, g.ab, g.c, g.W, g.X, d_y};
                LtDesc d; d.make(c);
                cublasLtMatmulAlgo_t a;
                CK(cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, g.ab, g.ab, g.c, g.c, f.id, &a));
                uint32_t u = f.tile; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &u, 4);
                u = f.stages; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &u, 4);
                u = f.swz; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &u, 4);
                cublasLtMatmulHeuristicResult_t r;
                if (cublasLtMatmulAlgoCheck(g_lt, d.op, d.la, d.lb, d.lc, d.lc, &a, &r) != CUBLAS_STATUS_SUCCESS) { printf("  %s: config not supported\n", g.nm); continue; }
                plans[g.nm] = {d, a};
            }
            std::vector<std::string> variants = {"default", "fixed"};
            if (const char* e = getenv("GT_VARIANTS")) for (char* q = strtok(strdup(e), ","); q; q = strtok(nullptr, ",")) variants.push_back(q);
            const size_t ns = seq.size(), nv = variants.size();
            std::vector<cudaEvent_t> ev(nv * (ns + 1));
            for (auto& x : ev) CK(cudaEventCreate(&x));
            std::vector<std::vector<float>> mn(nv, std::vector<float>(ns, 1e9f));
            auto pass_ev = [&](size_t v, cudaEvent_t* E) {
                for (size_t i = 0; i < ns; ++i) {
                    CK(cudaEventRecord(E[i], st));
                    const G& g = seq[i];
                    Case c{g.nm, g.M, g.N, g.K, g.ab, g.c, g.W, g.X, d_y};
                    auto it = plans.find(g.nm);
                    const bool use = it != plans.end() && (v == 1 || (v > 1 && variants[v] == g.nm));
                    if (use) CK(cublasLtMatmul(g_lt, it->second.first.op, &kOne, g.W, it->second.first.la, g.X, it->second.first.lb, &kZero,
                                               d_y, it->second.first.lc, d_y, it->second.first.lc, &it->second.second, g_ws, g_wsz, st));
                    else run_default(c);
                }
                CK(cudaEventRecord(E[ns], st));
            };
            for (int r = -1; r < rounds; ++r) {
                for (size_t j = 0; j < nv; ++j) { const size_t v = (j + std::max(r, 0)) % nv; pass_ev(v, &ev[v * (ns + 1)]); }
                CK(cudaStreamSynchronize(st));
                if (r < 0) continue;
                for (size_t v = 0; v < nv; ++v)
                    for (size_t i = 0; i < ns; ++i) { float a; CK(cudaEventElapsedTime(&a, ev[v * (ns + 1) + i], ev[v * (ns + 1) + i + 1])); mn[v][i] = std::min(mn[v][i], a); }
            }
            for (auto& x : ev) cudaEventDestroy(x);
            printf("AGG M=%d %s:", M, outs.c_str());
            for (size_t v = 0; v < nv; ++v) {
                double t = 0; for (size_t i = 0; i < ns; ++i) t += mn[v][i];
                printf(" %s %.2f", variants[v].c_str(), t);
                if (dump) for (size_t i = 0; i < ns; ++i) fprintf(dump, "%d %s %s %zu %s %.5f\n", M, outs.c_str(), variants[v].c_str(), i, seq[i].nm, mn[v][i]);
            }
            printf("\n");
            fflush(stdout);
        }
        if (dump) fclose(dump);
    } else if (mode == "tunecost") {
        // the one-time cost: VisionGemm tuning the six GEMMs at every class Vision tunes (630..5040 step 630, 576, 4761)
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cublasSetStream(g_h, st));
        const cudaDataType lt = outs == "bf16" ? CUDA_R_16BF : CUDA_R_32F;
        strata::gemma::VisionGemm vg;
        vg.init(g_h, st);
        vg.tuning = true;
        std::vector<int> cls;
        for (int b = 1; b * 630 <= 5040; ++b) cls.push_back(630 * b);
        cls.push_back(576); cls.push_back(4761);
        const double t0 = now_s();
        for (int M : cls) {
            if (M > Mmax) continue;
            vg.run(d_patch, CUDA_R_16F, d_pt, CUDA_R_16F, d_y, CUDA_R_32F, M, D, 768, "patch");
            vg.run(d_qkv, CUDA_R_16BF, d_hb, CUDA_R_16BF, d_y, lt, M, 3 * D, D, "qkv");
            vg.run(d_o, CUDA_R_16BF, d_ab, CUDA_R_16BF, d_y, lt, M, D, D, "attn_out");
            vg.run(d_gu, CUDA_R_16BF, d_hb, CUDA_R_16BF, d_y, lt, M, 2 * F, D, "gate_up");
            vg.run(d_down, CUDA_R_16BF, d_fb, CUDA_R_16BF, d_y, lt, M, D, F, "down");
            vg.run(d_proj, CUDA_R_16BF, d_hb, CUDA_R_16BF, d_y, CUDA_R_32F, M / 9, 2816, D, "proj");
        }
        CK(cudaStreamSynchronize(st));
        printf("TUNECOST %s: %d shapes in %.0f ms wall (%.0f ms timing); one call each: default %.2f ms -> %.2f ms\n",
               outs.c_str(), vg.n_tuned, (now_s() - t0) * 1e3, vg.tune_ms, vg.ms_default, vg.ms_chosen);
        printf("%s", vg.summary().c_str());
    } else if (mode == "loopcmp") {
        // one shape (first of the shape list), the default vs fixed algorithms (GT_CFGS="21:20:11:0;5:20:0:1"): single
        // launches and loops of GT_LOOP launches back to back, alternating, min / median per launch
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cublasSetStream(g_h, st));
        int L = getenv("GT_LOOP") ? atoi(getenv("GT_LOOP")) : 40;
        const int M = Ms[0];
        const cudaDataType ct = cts[0];
        const std::string nm = shp.substr(0, shp.find(','));
        Case c = nm == "qkv" ? Case{"qkv", M, 3 * D, D, CUDA_R_16BF, ct, d_qkv, d_hb, d_y}
               : nm == "attn_out" ? Case{"attn_out", M, D, D, CUDA_R_16BF, ct, d_o, d_ab, d_y}
               : nm == "down" ? Case{"down", M, D, F, CUDA_R_16BF, ct, d_down, d_fb, d_y}
               : nm == "patch" ? Case{"patch", M, D, 768, CUDA_R_16F, CUDA_R_32F, d_patch, d_pt, d_y}
               : nm == "proj" ? Case{"proj", M, 2816, D, CUDA_R_16BF, CUDA_R_32F, d_proj, d_hb, d_y}
               : Case{"gate_up", M, 2 * F, D, CUDA_R_16BF, ct, d_gu, d_hb, d_y};
        LtDesc d; d.make(c);
        std::vector<std::pair<std::string, cublasLtMatmulAlgo_t>> algos;
        if (const char* e = getenv("GT_CFGS"))
            for (char* q = strtok(strdup(e), ";"); q; q = strtok(nullptr, ";")) {
                int id, t, stg, sw;
                if (sscanf(q, "%d:%d:%d:%d", &id, &t, &stg, &sw) != 4) continue;
                cublasLtMatmulAlgo_t a;
                CK(cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, c.at, c.at, c.ct, c.ct, id, &a));
                uint32_t u = t; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &u, 4);
                u = stg; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &u, 4);
                u = sw; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &u, 4);
                cublasLtMatmulHeuristicResult_t r;
                if (cublasLtMatmulAlgoCheck(g_lt, d.op, d.la, d.lb, d.lc, d.lc, &a, &r) == CUBLAS_STATUS_SUCCESS) algos.push_back({q, a});
            }
        const size_t nv = algos.size() + 1;
        std::vector<std::vector<float>> s1(nv), sl(nv);
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        if (getenv("GT_LOOP_MS")) {   // loops of about that many milliseconds (from the default's single launch)
            float best = 1e9f;
            for (int r = 0; r < 5; ++r) { CK(cudaEventRecord(e0, st)); run_default(c); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); float a; CK(cudaEventElapsedTime(&a, e0, e1)); best = std::min(best, a); }
            L = std::max(2, (int) (atof(getenv("GT_LOOP_MS")) / best));
        }
        for (int r = -1; r < rounds; ++r)
            for (size_t k = 0; k < nv; ++k) {
                const size_t v = (k + std::max(r, 0)) % nv;
                auto one = [&] { if (v == 0) run_default(c); else CK(cublasLtMatmul(g_lt, d.op, &kOne, c.W, d.la, c.X, d.lb, &kZero, c.Y, d.lc, c.Y, d.lc, &algos[v - 1].second, g_ws, g_wsz, st)); };
                CK(cudaEventRecord(e0, st)); one(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
                float a; CK(cudaEventElapsedTime(&a, e0, e1)); if (r >= 0) s1[v].push_back(a);
                CK(cudaEventRecord(e0, st)); for (int l = 0; l < L; ++l) one(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&a, e0, e1)); if (r >= 0) sl[v].push_back(a / L);
            }
        auto mnf = [](std::vector<float> v) { return *std::min_element(v.begin(), v.end()); };
        auto mdf = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        if (getenv("GT_TRACE")) {   // per launch index after 50 ms idle: when does the power cap set in
            const int TL = L;
            std::vector<cudaEvent_t> te(TL + 1);
            for (auto& x : te) CK(cudaEventCreate(&x));
            std::vector<std::vector<std::vector<float>>> tr(nv, std::vector<std::vector<float>>(TL));
            for (int r = 0; r < 4; ++r)
                for (size_t v = 0; v < nv; ++v) {
                    CK(cudaStreamSynchronize(st)); usleep(50000);
                    for (int l = 0; l < TL; ++l) {
                        CK(cudaEventRecord(te[l], st));
                        if (v == 0) run_default(c); else CK(cublasLtMatmul(g_lt, d.op, &kOne, c.W, d.la, c.X, d.lb, &kZero, c.Y, d.lc, c.Y, d.lc, &algos[v - 1].second, g_ws, g_wsz, st));
                    }
                    CK(cudaEventRecord(te[TL], st)); CK(cudaStreamSynchronize(st));
                    for (int l = 0; l < TL; ++l) { float a; CK(cudaEventElapsedTime(&a, te[l], te[l + 1])); tr[v][l].push_back(a); }
                }
            printf("TRACE (median of 4, ms per launch; elapsed ms at the default's index in brackets)\n idx ");
            for (size_t v = 0; v < nv; ++v) printf(" %14s", v ? algos[v - 1].first.c_str() : "default");
            printf("\n");
            double el = 0;
            for (int l = 0; l < TL; ++l) {
                el += mdf(tr[0][l]);
                if (l < 12 || l % 4 == 0) {
                    printf(" %3d [%6.1f]", l, el);
                    for (size_t v = 0; v < nv; ++v) printf(" %14.3f", mdf(tr[v][l]));
                    printf("\n");
                }
            }
        }
        for (size_t v = 0; v < nv; ++v)
            printf("LOOP %s M=%d %s %-14s single min %.3f med %.3f | loop of %d: min %.3f med %.3f\n", c.name, M,
                   ct == CUDA_R_32F ? "f32" : "bf16", v ? algos[v - 1].first.c_str() : "default", mnf(s1[v]), mdf(s1[v]), L,
                   mnf(sl[v]), mdf(sl[v]));
    } else if (mode == "list") {
        for (int M : Ms)
            for (cudaDataType ct : cts) {
                Case cs[] = {{"qkv", M, 3 * D, D, CUDA_R_16BF, ct, d_qkv, d_hb, d_y}, {"attn_out", M, D, D, CUDA_R_16BF, ct, d_o, d_ab, d_y},
                             {"gate_up", M, 2 * F, D, CUDA_R_16BF, ct, d_gu, d_hb, d_y}, {"down", M, D, F, CUDA_R_16BF, ct, d_down, d_fb, d_y}};
                for (auto& c : cs) {
                    if (!has(c.name)) continue;
                    tune_case(c, exh, rounds, true, true);
                }
            }
    }
    return 0;
}
