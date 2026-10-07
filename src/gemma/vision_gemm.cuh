// src/gemma/vision_gemm.cuh - the vision encoder's GEMMs with the algorithm picked per shape (included by vision.cu).
//
// Y [M][N] = X [M][K] . W [N][K]^T: W / X bf16 (or f16), FP32 accumulation, Y f32 or bf16 - in cuBLAS's column-major
// terms D [N x M] = op_T(W [K x N]) . X [K x M], what Vision called cublasGemmEx(CUBLAS_GEMM_DEFAULT) for.
//
// What was measured (RTX A4000, 140 W power cap, tools/gemma/vis_gemm_bench.cu): timed one launch at a time, the best
// cuBLASLt algorithm beats cuBLAS's default by 5-15% on the big encoder GEMMs (qkv, gate_up, down at 4000-5700 rows),
// mostly 128x128 tiles that fill the 48 SMs better. That does not survive sustained load: in loops of ~100 ms (the
// power cap acting) the same kernels are 0-8% SLOWER than the default's larger tiles, which move less data per FLOP,
// and inside a whole encoder pass a faster qkv slows the gate_up after it. The GPU runs at the cap whenever it is busy,
// so there the default stays. What does survive sustained load: shorter GEMMs where cuBLAS picks a poor kernel - the
// f16 patch GEMM (-38%), the projection (-25%), and the layer GEMMs at video-frame / drag-piece row counts (qkv -17%,
// down -27% at 630 rows, attn_out -17% at 1890).
//
// So only GEMMs of at most max_macs multiply-adds (M N K, default 13e9: qkv up to ~3300 rows, down ~2600, gate_up
// ~1300, attn_out / patch / projection always) are tuned: timed once per (N, K, types, M) among cublasGemmEx's
// default, cuBLASLt's heuristic top 3 and the fixed configurations below (each won an exhaustive search over algo id x
// tile x stages x split-K x swizzle at some shape), the fastest kept if `margin` (8%) faster than the default on fresh
// rounds. Timing happens only while `tuning` is set (Vision's constructor runs synthetic encodes at the common row
// counts); afterwards a call takes the choice of the nearest tuned M with the same (N, K, types), checked for its exact
// shape with cublasLtMatmulAlgoCheck, else the default: no timing on the request path. The fixed list has no split-K;
// every candidate accumulates in FP32 along K: identical bits for the bf16 GEMMs where the default does not split K,
// ~1e-6 relative otherwise (the f16 patch GEMM, the projection, split-K defaults at small M).
// Not thread-safe: Vision's encodes are serialized.
#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace strata::gemma {

class VisionGemm {
public:
    VisionGemm() = default;
    VisionGemm(const VisionGemm&) = delete;
    VisionGemm& operator=(const VisionGemm&) = delete;
    ~VisionGemm() { release(); }

    /// blas: the handle (on stream s) the default path calls; ws_bytes of cuBLASLt workspace (allocated here)
    void init(cublasHandle_t blas, cudaStream_t s, size_t ws_bytes = 4u << 20) {
        blas_ = blas;
        s_ = s;
        lt_check(cublasLtCreate(&lt_), "create");
        lt_check(cublasLtMatmulDescCreate(&op_, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
        const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        lt_check(cublasLtMatmulDescSetAttribute(op_, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "desc");
        lt_check(cublasLtMatmulDescSetAttribute(op_, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "desc");
        ws_bytes_ = ws_bytes;
        if (ws_bytes_ && cudaMalloc(&ws_, ws_bytes_) != cudaSuccess) throw std::runtime_error("vision gemm: workspace");
    }

    /// Y [M][N] = X [M][K] . W [N][K]^T, FP32 accumulation; W of type wt, X of type xt (bf16 | f16), Y of type yt
    void run(const void* W, cudaDataType wt, const void* X, cudaDataType xt, void* Y, cudaDataType yt, int M, int N,
             int K, const char* what) {
        if (M <= 0) return;
        if (!enabled || wt != xt || (double) M * N * K > max_macs) return run_default(W, wt, X, xt, Y, yt, M, N, K, what);
        const Shape sh{N, K, (int) wt, (int) yt};
        if (tuning) {
            auto& v = tuned_[sh];
            if (std::none_of(v.begin(), v.end(), [&](const Choice& c) { return c.M == M; })) {
                tune(sh, W, X, Y, M);
                plans_.erase({sh, M});   // re-planned with the new choice
            }
        }
        Plan& p = plan(sh, M);
        if (!p.lt || !aligned(W, p.align_a) || !aligned(X, p.align_b) || !aligned(Y, p.align_c))
            return run_default(W, wt, X, xt, Y, yt, M, N, K, what);
        static const float one = 1.f, zero = 0.f;
        lt_check(cublasLtMatmul(lt_, op_, &one, W, p.la, X, p.lb, &zero, Y, p.lc, Y, p.lc, &p.algo, ws_, ws_bytes_, s_), what);
    }

    bool tuning = false;      // run() times the candidates for an (N, K, types, M) not tuned yet and keeps the fastest
    bool enabled = true;      // false: cublasGemmEx's default everywhere (the behaviour before the tuner)
    double max_macs = 13e9;   // larger GEMMs keep the default (it is the faster one at the power cap, see above)
    float margin = 0.08f;     // a candidate replaces the default only this much faster
    double tune_ms = 0;    // wall time spent tuning
    int n_tuned = 0;
    float ms_default = 0, ms_chosen = 0;   // summed over the tuned shapes (one call each)

    /// one line per tuned (shape, M): default ms -> chosen ms and the algorithm
    std::string summary() const {
        std::string out;
        char b[256];
        for (const auto& [sh, v] : tuned_)
            for (const Choice& c : v) {
                std::snprintf(b, sizeof b, "  N %5d K %5d %s->%s M %5d: default %.3f ms, chosen %.3f ms (%+.1f%%) %s\n",
                              sh.N, sh.K, type_name(sh.ab), type_name(sh.c), c.M, c.ms_def, c.ms,
                              100.0 * (c.ms_def / c.ms - 1.0), c.lt ? cfg_str(cfg_of(c.algo)).c_str() : "cublasGemmEx default");
                out += b;
            }
        return out;
    }

private:
    struct Shape {
        int N, K, ab, c;
        bool operator<(const Shape& o) const { return std::tie(N, K, ab, c) < std::tie(o.N, o.K, o.ab, o.c); }
    };
    struct Choice { int M; bool lt; cublasLtMatmulAlgo_t algo; float ms_def, ms; };
    struct Plan {
        bool lt = false;
        cublasLtMatmulAlgo_t algo{};
        cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
        uint32_t align_a = 256, align_b = 256, align_c = 256;
    };
    struct Cfg { int32_t id = -1; uint32_t tile = 0, stages = 0, splitk = 1, red = 0, swz = 0, custom = 0; };
    // the fixed candidates (algo id, tile, stages, CTA swizzling): each won the exhaustive search at some (shape, M)
    struct Fixed { int32_t id; uint32_t tile, stages, swz; };
    static constexpr Fixed kFixed[] = {
        {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x5, 0},
        {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x5, 1},
        {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x4, 1},
        {5, CUBLASLT_MATMUL_TILE_128x128, 0, 1},
        {6, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x5, 1},
        {6, CUBLASLT_MATMUL_TILE_128x64, 0, 0},
        {21, CUBLASLT_MATMUL_TILE_64x128, CUBLASLT_MATMUL_STAGES_32x6, 1},
        {21, CUBLASLT_MATMUL_TILE_64x64, CUBLASLT_MATMUL_STAGES_32x6, 1},
    };

    cublasHandle_t blas_ = nullptr;
    cudaStream_t s_ = nullptr;
    cublasLtHandle_t lt_ = nullptr;
    cublasLtMatmulDesc_t op_ = nullptr;
    void* ws_ = nullptr;
    size_t ws_bytes_ = 0;
    std::vector<cudaEvent_t> evs_;   // tuning: an event pair per candidate
    std::map<Shape, std::vector<Choice>> tuned_;
    std::map<std::pair<Shape, int>, Plan> plans_;   // per exact M: layouts + the algorithm (or the default)

    static void lt_check(cublasStatus_t e, const char* what) {
        if (e != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error(std::string("vision cuBLASLt ") + what + ": " + std::to_string((int) e));
    }
    static bool aligned(const void* p, uint32_t a) { return a == 0 || reinterpret_cast<uintptr_t>(p) % a == 0; }
    static const char* type_name(int t) {
        return t == CUDA_R_32F ? "f32" : t == CUDA_R_16BF ? "bf16" : t == CUDA_R_16F ? "f16" : "?";
    }
    static Cfg cfg_of(const cublasLtMatmulAlgo_t& a) {
        Cfg c;
        size_t w = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_ID, &c.id, sizeof(c.id), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &c.tile, sizeof(c.tile), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &c.stages, sizeof(c.stages), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &c.splitk, sizeof(c.splitk), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &c.red, sizeof(c.red), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &c.swz, sizeof(c.swz), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &c.custom, sizeof(c.custom), &w);
        return c;
    }
    static bool same(const Cfg& a, const Cfg& b) {
        return a.id == b.id && a.tile == b.tile && a.stages == b.stages && a.splitk == b.splitk && a.red == b.red &&
               a.swz == b.swz && a.custom == b.custom;
    }
    static std::string cfg_str(const Cfg& c) {
        char b[96];
        std::snprintf(b, sizeof b, "cuBLASLt algo %d tile %u stages %u split-k %u swizzle %u", c.id, c.tile, c.stages,
                      c.splitk, c.swz);
        return b;
    }

    void run_default(const void* W, cudaDataType wt, const void* X, cudaDataType xt, void* Y, cudaDataType yt, int M,
                     int N, int K, const char* what) {
        static const float one = 1.f, zero = 0.f;
        const cublasStatus_t e = cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, wt, K, X, xt, K, &zero,
                                              Y, yt, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (e != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error(std::string("vision cuBLAS ") + what + ": " + std::to_string((int) e));
    }

    void layouts(Plan& p, const Shape& sh, int M) {
        lt_check(cublasLtMatrixLayoutCreate(&p.la, (cudaDataType) sh.ab, sh.K, sh.N, sh.K), "layout");
        lt_check(cublasLtMatrixLayoutCreate(&p.lb, (cudaDataType) sh.ab, sh.K, M, sh.K), "layout");
        lt_check(cublasLtMatrixLayoutCreate(&p.lc, (cudaDataType) sh.c, sh.N, M, sh.N), "layout");
    }
    static void free_layouts(Plan& p) {
        if (p.la) cublasLtMatrixLayoutDestroy(p.la);
        if (p.lb) cublasLtMatrixLayoutDestroy(p.lb);
        if (p.lc) cublasLtMatrixLayoutDestroy(p.lc);
        p.la = p.lb = p.lc = nullptr;
    }
    // usable for this layout within the workspace: the algorithm's alignment needs into p
    bool check(Plan& p, const cublasLtMatmulAlgo_t& a) {
        cublasLtMatmulHeuristicResult_t r;
        if (cublasLtMatmulAlgoCheck(lt_, op_, p.la, p.lb, p.lc, p.lc, &a, &r) != CUBLAS_STATUS_SUCCESS) return false;
        if (r.workspaceSize > ws_bytes_) return false;
        size_t w = 0;
        uint32_t al[4] = {256, 256, 256, 256};
        const cublasLtMatmulAlgoCapAttributes_t caps[4] = {
            CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES};
        for (int i = 0; i < 4; ++i) cublasLtMatmulAlgoCapGetAttribute(&a, caps[i], &al[i], sizeof(al[i]), &w);
        p.align_a = al[0];
        p.align_b = al[1];
        p.align_c = std::max(al[2], al[3]);
        return true;
    }

    Plan& plan(const Shape& sh, int M) {
        auto it = plans_.find({sh, M});
        if (it != plans_.end()) return it->second;
        Plan p;
        layouts(p, sh, M);
        auto tv = tuned_.find(sh);
        if (tv != tuned_.end() && !tv->second.empty()) {
            const Choice* best = nullptr;   // the nearest tuned M (ratio)
            double bd = 1e30;
            for (const Choice& c : tv->second) {
                const double d = std::fabs(std::log((double) M / c.M));
                if (d < bd || (d == bd && c.M > best->M)) { bd = d; best = &c; }
            }
            if (best->lt && check(p, best->algo)) {
                p.lt = true;
                p.algo = best->algo;
            }
        }
        if (plans_.size() > 4096) {   // bounded: row counts are many but few distinct ones recur
            for (auto& [k, q] : plans_) free_layouts(q);
            plans_.clear();
        }
        return plans_.emplace(std::make_pair(sh, M), p).first->second;
    }

    // time the candidates on this call's own buffers, keep the fastest for (sh, M)
    void tune(const Shape& sh, const void* W, const void* X, void* Y, int M) {
        const auto t0 = std::chrono::steady_clock::now();
        struct Cand { bool lt; cublasLtMatmulAlgo_t algo; Cfg cfg; float best; };
        std::vector<Cand> cs;
        cs.push_back({false, {}, {}, 1e30f});   // cublasGemmEx's default
        Plan p;
        layouts(p, sh, M);
        auto add = [&](const cublasLtMatmulAlgo_t& a) {
            if (!check(p, a) || !aligned(W, p.align_a) || !aligned(X, p.align_b) || !aligned(Y, p.align_c)) return;
            const Cfg c = cfg_of(a);
            for (const Cand& o : cs)
                if (o.lt && same(o.cfg, c)) return;
            cs.push_back({true, a, c, 1e30f});
        };
        {
            cublasLtMatmulPreference_t pref;
            lt_check(cublasLtMatmulPreferenceCreate(&pref), "preference");
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes_, sizeof(ws_bytes_));
            cublasLtMatmulHeuristicResult_t res[3];
            int n = 0;
            if (cublasLtMatmulAlgoGetHeuristic(lt_, op_, p.la, p.lb, p.lc, p.lc, pref, 3, res, &n) == CUBLAS_STATUS_SUCCESS)
                for (int i = 0; i < n; ++i)
                    if (res[i].state == CUBLAS_STATUS_SUCCESS) add(res[i].algo);
            cublasLtMatmulPreferenceDestroy(pref);
        }
        for (const Fixed& f : kFixed) {
            cublasLtMatmulAlgo_t a;
            const cudaDataType ab = (cudaDataType) sh.ab, c = (cudaDataType) sh.c;
            if (cublasLtMatmulAlgoInit(lt_, CUBLAS_COMPUTE_32F, CUDA_R_32F, ab, ab, c, c, f.id, &a) != CUBLAS_STATUS_SUCCESS)
                continue;
            const uint32_t one = 1, none = 0;
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &f.tile, sizeof(f.tile));
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &f.stages, sizeof(f.stages));
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, sizeof(one));
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, sizeof(none));
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &f.swz, sizeof(f.swz));
            cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &none, sizeof(none));
            add(a);
        }
        static const float onef = 1.f, zerof = 0.f;
        auto launch = [&](const Cand& c) {
            if (!c.lt)
                run_default(W, (cudaDataType) sh.ab, X, (cudaDataType) sh.ab, Y, (cudaDataType) sh.c, M, sh.N, sh.K, "tune");
            else
                lt_check(cublasLtMatmul(lt_, op_, &onef, W, p.la, X, p.lb, &zerof, Y, p.lc, Y, p.lc, &c.algo, ws_,
                                        ws_bytes_, s_), "tune");
        };
        // timed back to back (one sync a round, as the encoder runs them), an event pair around each launch, min over
        // the rounds, the order rotated each round: one untimed round loads the kernels, two rounds screen everything,
        // four decide between the best two and the default; then the winner and the default are re-timed on fresh
        // rounds (the minimum of many noisy candidates is biased low) and the winner kept only if `margin` faster
        if (evs_.size() < 2 * cs.size())
            for (size_t i = evs_.size(); i < 2 * cs.size(); ++i) {
                cudaEvent_t e;
                if (cudaEventCreate(&e) != cudaSuccess) throw std::runtime_error("vision gemm: events");
                evs_.push_back(e);
            }
        auto rounds = [&](const std::vector<int>& idx, int nr) {
            const int n = (int) idx.size();
            for (int r = 0; r < nr; ++r) {
                for (int j = 0; j < n; ++j) {
                    const int i = idx[(j + r) % n];
                    if (cudaEventRecord(evs_[2 * i], s_) != cudaSuccess) throw std::runtime_error("vision gemm: event");
                    launch(cs[i]);
                    cudaEventRecord(evs_[2 * i + 1], s_);
                }
                if (cudaStreamSynchronize(s_) != cudaSuccess) throw std::runtime_error("vision gemm: tune sync");
                for (int i : idx) {
                    float ms = 0;
                    cudaEventElapsedTime(&ms, evs_[2 * i], evs_[2 * i + 1]);
                    cs[i].best = std::min(cs[i].best, ms);
                }
            }
        };
        std::vector<int> all(cs.size());
        for (size_t i = 0; i < cs.size(); ++i) all[i] = (int) i;
        rounds(all, 1);
        for (Cand& c : cs) c.best = 1e30f;
        rounds(all, 2);
        std::vector<int> top(all.begin() + 1, all.end());
        std::sort(top.begin(), top.end(), [&](int a, int b) { return cs[a].best < cs[b].best; });
        if (top.size() > 2) top.resize(2);
        top.insert(top.begin(), 0);
        for (int i : top) cs[i].best = 1e30f;
        rounds(top, 4);
        int bi = 0;
        for (int i : top)
            if (cs[i].best < cs[bi].best) bi = i;
        if (bi != 0) {
            cs[0].best = cs[bi].best = 1e30f;
            rounds({0, bi}, 5);
            if (cs[bi].best > (1.f - margin) * cs[0].best) bi = 0;   // the default unless clearly beaten
        }
        free_layouts(p);
        tuned_[sh].push_back({M, cs[bi].lt, cs[bi].algo, cs[0].best, cs[bi].best});
        ++n_tuned;
        ms_default += cs[0].best;
        ms_chosen += cs[bi].best;
        tune_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    void release() {
        for (auto& [k, p] : plans_) free_layouts(p);
        plans_.clear();
        for (cudaEvent_t e : evs_) cudaEventDestroy(e);
        evs_.clear();
        if (op_) cublasLtMatmulDescDestroy(op_);
        op_ = nullptr;
        if (lt_) cublasLtDestroy(lt_);
        lt_ = nullptr;
        if (ws_) cudaFree(ws_);
        ws_ = nullptr;
    }
};

}  // namespace strata::gemma
