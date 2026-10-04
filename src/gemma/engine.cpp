// src/gemma/engine.cpp - see include/strata/gemma/engine.hpp.
#include "strata/gemma/engine.hpp"

#include "strata/gemma/kernels.hpp"
#include "strata/gemma/mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::gemma {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void cb(cublasStatus_t e, const char* what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("cuBLAS ") + what + ": status " + std::to_string((int) e));
}
size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// STRATA_DUMP=<dir>: write intermediate tensors of the next prompt (n > 1) as <dir>/<name>.bin, in the layout of
// tools/gemma/ref_logits --dump (i64 ne[4], then f32 rows), named like llama.cpp's graph callbacks
const char* dump_dir() {
    static const char* d = std::getenv("STRATA_DUMP");
    return d;
}
bool g_dumping = false;
void dump(const char* name, int il, const float* dev, int64_t ne0, int64_t ne1, cudaStream_t s) {
    if (!g_dumping) return;
    std::vector<float> h((size_t) ne0 * ne1);
    cudaMemcpyAsync(h.data(), dev, h.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    const std::string path = std::string(dump_dir()) + "/" + name + (il >= 0 ? "-" + std::to_string(il) : "") + ".bin";
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        const int64_t ne[4] = {ne0, ne1, 1, 1};
        std::fwrite(ne, 8, 4, f);
        std::fwrite(h.data(), 4, h.size(), f);
        std::fclose(f);
    }
}

}  // namespace

template <class T> T* Engine::carve(size_t count) {
    const size_t off = align256(carve_off_);
    carve_off_ = off + count * sizeof(T);
    if (!arena_) return nullptr;   // sizing pass
    return reinterpret_cast<T*>(static_cast<uint8_t*>(arena_) + off);
}

Engine::Engine(const Model& m, int ctx, int max_batch) : m_(m), ctx_(ctx), max_batch_(max_batch) {
    const Config& c = m.cfg;
    ck(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), "stream");
    cublasHandle_t h;
    cb(cublasCreate(&h), "create");
    cb(cublasSetStream(h, s_), "stream");
    cublas_ = h;
    mmq_ = new mmq::Context();

    const int64_t N = max_batch, D = c.n_embd, Q = c.max_q_dim, KV = c.max_kv_dim, F = c.n_ff;
    const int64_t E = std::max(c.n_expert, 1), K = std::max(c.n_expert_used, 1), FE = std::max(c.n_ff_exp, 32);
    int max_head = 0;
    for (const Layer& L : m.layers) max_head = std::max(max_head, L.n_head);
    size_t xq = 0;
    for (auto [rows, cols] : {std::pair<int64_t, int64_t>{N, D}, {N, Q}, {N, F}, {N * K, D}, {N * K, FE}})
        xq = std::max(xq, mmq::q8_bytes(rows, cols));
    // native_mmvq's q8_1: 36 bytes per 32 values, the columns back to back
    for (int64_t vals : {kSmall * Q, kSmall * F, kSmall * K * D, kSmall * K * FE}) xq = std::max(xq, (size_t) vals / 32 * 36);
    xq_bytes_ = xq;

    // two passes over the same carve list: the first sizes the arena, the second assigns the pointers
    for (int pass = 0; pass < 2; ++pass) {
        carve_off_ = 0;
        kc_.assign(c.n_layer, nullptr);
        vc_.assign(c.n_layer, nullptr);
        for (int il = 0; il < c.n_layer; ++il) {
            const Layer& L = m.layers[il];
            kc_[il] = carve<__half>((size_t) ctx * L.n_head_kv * L.head_dim);
            vc_[il] = carve<__half>((size_t) ctx * L.n_head_kv * L.head_dim);
        }
        x_ = carve<float>(N * D);
        x1_ = carve<float>(N * D);
        h_ = carve<float>(N * D);
        f_ = carve<float>(N * D);
        g_ = carve<float>(N * D);
        r_ = carve<float>(N * D);
        y_ = carve<float>(N * D);
        mlp_ = carve<float>(N * D);
        moe_ = carve<float>(N * D);
        q_ = carve<float>(N * Q);
        att_ = carve<float>(N * Q);
        k_ = carve<float>(N * KV);
        v_ = carve<float>(N * KV);
        gate_ = carve<float>(N * F);
        up_ = carve<float>(N * F);
        hid_ = carve<float>(N * F);
        rlog_ = carve<float>(N * E);
        ew_ = carve<float>(N * K);
        eid_ = carve<int32_t>(N * K);
        gu_ = carve<float>(N * K * 2 * FE);
        eh_ = carve<float>(N * K * FE);
        ey_ = carve<float>(N * K * D);
        bounds_ = carve<int32_t>(E + 1);
        src_ = carve<int32_t>(N * K);
        inv_ = carve<int32_t>(N * K);
        counts_ = carve<int32_t>(E);
        iota_ = carve<int32_t>(N * K + 2);
        S_ = carve<float>((size_t) max_head * N * (ctx + 8));
        P_ = carve<__half>((size_t) max_head * N * (ctx + 8));
        q16_ = carve<__half>(N * Q);
        xq_ = carve<uint8_t>(xq);
        att_scratch_ = carve<uint8_t>(k::attn_decode_scratch_bytes(kSmall, max_head, c.max_head_dim, ctx));
        tok_ = carve<int32_t>(N);
        pos_ = carve<int32_t>(N);
        lo_ = carve<int32_t>(2 * N);
        hi_ = carve<int32_t>(2 * N);
        spans_ = carve<int32_t>(256);
        logits_ = carve<float>((size_t) c.n_vocab * kSmall);
        amax_ = carve<int32_t>(kSmall);
        hnorm_ = carve<float>((size_t) kSmall * D);
        groups_ = carve<uint8_t>(strata::kernels::native_mmvq_groups_bytes());
        xqf_ = carve<uint8_t>((size_t) kSmall * D / 32 * 36);
        xqg_ = carve<uint8_t>((size_t) kSmall * D / 32 * 36);
        xqh_ = carve<uint8_t>((size_t) kSmall * F / 32 * 36);
        xqe_ = carve<uint8_t>((size_t) kSmall * K * FE / 32 * 36);
        ptrs_ = reinterpret_cast<const void**>(carve<int32_t>(4));   // dense bounds {0, n} live here
        if (pass == 0) {
            buf_bytes_ = align256(carve_off_);
            ck(cudaMalloc(&arena_, buf_bytes_), "cudaMalloc(engine buffers)");
            ck(cudaMemset(arena_, 0, buf_bytes_), "cudaMemset(engine buffers)");
        }
    }
    mmq::iota(iota_, N * K + 2, s_);
    h_logits_.resize(c.n_vocab);
    h_amax_.resize(kSmall);
    n_split_ = (ctx + 127) / 128;
    graph_exec_.assign((kSmall + 1) * 3, nullptr);
    if (const char* e = std::getenv("STRATA_NO_GRAPH")) graphs_ = std::string(e) != "1";
    ck(cudaStreamSynchronize(s_), "init");
}

Engine::~Engine() {
    for (void* g : graph_exec_)
        if (g) cudaGraphExecDestroy((cudaGraphExec_t) g);
    if (cublas_) cublasDestroy((cublasHandle_t) cublas_);
    delete (mmq::Context*) mmq_;
    if (arena_) cudaFree(arena_);
    if (s_) cudaStreamDestroy(s_);
}

std::vector<int32_t> Engine::debug_expert_ids(int rows) const {
    std::vector<int32_t> v((size_t) rows * m_.cfg.n_expert_used);
    cudaMemcpy(v.data(), eid_, v.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    return v;
}

const float* Engine::hidden_dev(int i) const {
    if (i < 0) i = n_scored_ - 1;
    return hnorm_ + (size_t) (hrow0_ + i) * m_.cfg.n_embd;
}

const float* Engine::logits_host(int i) {
    if (i < 0) i = n_scored_ - 1;
    if (i != logits_row_) {
        ck(cudaMemcpyAsync(h_logits_.data(), logits_ + (size_t) i * m_.cfg.n_vocab, (size_t) m_.cfg.n_vocab * sizeof(float),
                           cudaMemcpyDeviceToHost, s_), "logits");
        ck(cudaStreamSynchronize(s_), "logits");
        logits_row_ = i;
    }
    return h_logits_.data();
}

void Engine::forward(const Batch& b, Logits mode) {
    const Config& c = m_.cfg;
    const int n = (int) b.tokens.size();
    if (n <= 0) return;
    if (n > max_batch_) throw std::runtime_error("forward: " + std::to_string(n) + " rows > max batch " + std::to_string(max_batch_));
    if (n_past_ + n > ctx_) throw std::runtime_error("forward: context full (" + std::to_string(n_past_ + n) + " > " + std::to_string(ctx_) + ")");
    if (mode == Logits::All && n > kSmall) throw std::runtime_error("forward: Logits::All takes at most kSmall rows");
    logits_row_ = -1;
    bool has_embd = false;
    for (int32_t t : b.tokens) has_embd |= t < 0;
    const bool small = n <= kSmall && !has_embd && b.spans.empty();

    // inputs: token ids (image rows gather row 0, then get overwritten), positions, image spans
    std::vector<int32_t> tok(n), pos(n);
    for (int i = 0; i < n; ++i) {
        tok[i] = std::max(b.tokens[i], 0);
        pos[i] = n_past_ + i;
    }
    n_spans_ = (int) b.spans.size() / 2;
    if (n_spans_ > 128) throw std::runtime_error("forward: too many image spans");
    ck(cudaMemcpyAsync(tok_, tok.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, s_), "tokens");
    ck(cudaMemcpyAsync(pos_, pos.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, s_), "positions");
    if (n_spans_) ck(cudaMemcpyAsync(spans_, b.spans.data(), b.spans.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s_), "spans");
    const int32_t dense_bounds[2] = {0, n};
    ck(cudaMemcpyAsync(ptrs_, dense_bounds, sizeof dense_bounds, cudaMemcpyHostToDevice, s_), "bounds");
    if (small) {   // the decode path: a captured graph per (rows, logits mode)
        const int gi = n * 3 + (int) mode;
        if (graphs_) {
            if (!graph_exec_[gi]) {
                cudaGraph_t g;
                ck(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal), "capture");
                run_small(n, mode);
                ck(cudaStreamEndCapture(s_, &g), "capture end");
                cudaGraphExec_t ge;
                ck(cudaGraphInstantiate(&ge, g, 0), "graph instantiate");
                cudaGraphDestroy(g);
                graph_exec_[gi] = ge;
            }
            ck(cudaGraphLaunch((cudaGraphExec_t) graph_exec_[gi], s_), "graph launch");
        } else {
            run_small(n, mode);
        }
        n_past_ += n;
        n_scored_ = mode == Logits::None ? 0 : mode == Logits::All ? n : 1;
        hrow0_ = mode == Logits::All ? 0 : n - 1;
        if (n_scored_) {
            ck(cudaMemcpyAsync(h_amax_.data(), amax_, n_scored_ * sizeof(int32_t), cudaMemcpyDeviceToHost, s_), "argmax");
        }
        ck(cudaStreamSynchronize(s_), "forward");
        return;
    }
    k::key_ranges(pos_, n, c.n_swa, true, spans_, n_spans_, lo_, hi_, s_);
    k::key_ranges(pos_, n, c.n_swa, false, spans_, n_spans_, lo_ + max_batch_, hi_ + max_batch_, s_);

    k::embed_rows(m_.tok_embd.d, m_.tok_embd.type, c.n_embd, tok_, x_, n, sqrtf((float) c.n_embd), s_);
    for (int i = 0, e = 0; i < n;) {   // image rows enter unscaled, a contiguous run at a time
        if (b.tokens[i] >= 0) {
            ++i;
            continue;
        }
        int j = i;
        while (j < n && b.tokens[j] < 0) ++j;
        if ((size_t) (e + (j - i)) * c.n_embd > b.embd.size()) throw std::runtime_error("forward: missing embedding rows");
        ck(cudaMemcpyAsync(x_ + (size_t) i * c.n_embd, b.embd.data() + (size_t) e * c.n_embd,
                           (size_t) (j - i) * c.n_embd * sizeof(float), cudaMemcpyHostToDevice, s_), "embd rows");
        e += j - i;
        i = j;
    }

    g_dumping = dump_dir() && n > 1;
    dump("inp_scaled", -1, x_, c.n_embd, n, s_);
    for (int il = 0; il < c.n_layer; ++il) {
        layer_big(il, n);
        dump("l_out", il, x_, c.n_embd, n, s_);
    }
    g_dumping = false;
    n_past_ += n;
    n_scored_ = mode == Logits::None ? 0 : 1;
    if (n_scored_) head(n - 1);
    else ck(cudaStreamSynchronize(s_), "forward");
}

void Engine::run_small(int n, Logits mode) {
    using namespace strata::kernels;
    const Config& c = m_.cfg;
    k::key_ranges(pos_, n, c.n_swa, true, spans_, 0, lo_, hi_, s_);
    k::key_ranges(pos_, n, c.n_swa, false, spans_, 0, lo_ + max_batch_, hi_ + max_batch_, s_);
    k::embed_rows(m_.tok_embd.d, m_.tok_embd.type, c.n_embd, tok_, x_, n, sqrtf((float) c.n_embd), s_);
    k::norm_quant(x_, m_.layers[0].attn_norm.f32(), xq_, c.n_embd, n, c.eps, s_);
    for (int il = 0; il < c.n_layer; ++il) layer_small(il, n);
    if (mode == Logits::None) return;
    // xq_ holds q8_1(rms(x) * output_norm) of every row
    const int rows = mode == Logits::All ? n : 1;
    const void* xq = static_cast<const uint8_t*>(xq_) + (size_t) (mode == Logits::All ? 0 : n - 1) * (c.n_embd / 32) * 36;
    native_mmvq(m_.tok_embd.type, m_.tok_embd.d, xq, logits_, c.n_embd, c.n_vocab, rows, s_);
    k::softcap(logits_, (int64_t) c.n_vocab * rows, c.softcap, s_);
    k::argmax_rows(logits_, c.n_vocab, rows, amax_, s_);
}

// ---------------------------------------------------------------------------------------------- a few rows

void Engine::layer_small(int il, int n) {
    using namespace strata::kernels;
    const Config& c = m_.cfg;
    const Layer& L = m_.layers[il];
    const int D = c.n_embd, qd = L.n_head * L.head_dim, kvd = L.n_head_kv * L.head_dim;
    const int K = c.n_expert_used, FE = c.n_ff_exp;

    // xq_ = q8_1(rms(x) * attn_norm), left by the previous layer's last kernel
    native_mmvq(L.wq.type, L.wq.d, xq_, q_, D, qd, n, s_);
    native_mmvq(L.wk.type, L.wk.d, xq_, k_, D, kvd, n, s_);
    if (L.wv) native_mmvq(L.wv.type, L.wv.d, xq_, v_, D, kvd, n, s_);

    k::QkvArgs a;
    a.q = q_;
    a.k = k_;
    a.v = L.wv ? v_ : nullptr;
    a.q_norm = L.q_norm.f32();
    a.k_norm = L.k_norm.f32();
    a.freq_factors = L.swa ? nullptr : (m_.rope_freqs ? m_.rope_freqs.f32() : nullptr);
    a.pos = pos_;
    a.kc = kc_[il];
    a.vc = vc_[il];
    a.rows = n;
    a.n_head = L.n_head;
    a.n_kv = L.n_head_kv;
    a.hd = L.head_dim;
    a.n_rot = L.n_rot;
    a.base = L.rope_base;
    a.eps = c.eps;
    k::qkv_prep(a, s_);
    const int32_t* lo = L.swa ? lo_ : lo_ + max_batch_;
    const int32_t* hi = L.swa ? hi_ : hi_ + max_batch_;
    k::attn_partials(q_, kc_[il], vc_[il], lo, hi, att_scratch_, n, L.n_head, L.n_head_kv, L.head_dim, n_split_, s_);
    k::attn_combine_quant(att_scratch_, att_, xq_, n, L.n_head, L.head_dim, n_split_, s_);
    native_mmvq(L.wo.type, L.wo.d, xq_, y_, qd, D, n, s_);

    const bool moe = L.moe();
    k::post_attn_fused(x_, y_, L.post_attn_norm.f32(), x1_, L.ffn_norm.f32(), xqf_, moe ? L.pre_ffw_norm_2.f32() : nullptr,
                       moe ? xqg_ : nullptr, moe ? L.router_scale.f32() : nullptr, r_, D, n, c.eps, s_);
    native_mmvq(L.ffn_gate.type, L.ffn_gate.d, xqf_, gate_, D, c.n_ff, n, s_);
    native_mmvq(L.ffn_up.type, L.ffn_up.d, xqf_, up_, D, c.n_ff, n, s_);
    k::geglu_quant(gate_, up_, c.n_ff, xqh_, c.n_ff, n, s_);
    native_mmvq(L.ffn_down.type, L.ffn_down.d, xqh_, mlp_, c.n_ff, D, n, s_);
    if (moe) {
        k::router_gemv(L.router.f32(), r_, rlog_, c.n_expert, D, n, s_);
        k::router_topk(rlog_, c.n_expert, K, n, eid_, ew_, s_);
        static const bool no_group = std::getenv("STRATA_NO_GROUP") != nullptr;
        if (n == 1 || no_group) {   // one token: the 4-warps-per-row id kernel is faster
            native_mmvq_id(L.gate_up_exps.type, L.gate_up_exps.d, L.gate_up_exps.nb2, eid_, n * K, xqg_, K, gu_, D, 2 * FE, s_);
            k::geglu_quant(gu_, gu_ + FE, 2 * FE, xqe_, FE, n * K, s_);
            native_mmvq_id(L.down_exps.type, L.down_exps.d, L.down_exps.nb2, eid_, n * K, xqe_, 1, ey_, FE, D, s_);
        } else {   // a verify window: an expert several tokens chose is read once
            native_mmvq_group_ids(eid_, n * K, groups_, s_);
            native_mmvq_grouped(L.gate_up_exps.type, L.gate_up_exps.d, L.gate_up_exps.nb2, groups_, n * K, xqg_, K, gu_, D,
                                2 * FE, s_);
            k::geglu_quant(gu_, gu_ + FE, 2 * FE, xqe_, FE, n * K, s_);
            native_mmvq_grouped(L.down_exps.type, L.down_exps.d, L.down_exps.nb2, groups_, n * K, xqe_, 1, ey_, FE, D, s_);
        }
    }
    const bool last = il + 1 == c.n_layer;
    k::moe_post_fused(moe ? ey_ : nullptr, eid_, ew_, L.down_exps_scale ? L.down_exps_scale.f32() : nullptr, K, mlp_, x1_,
                      moe ? L.post_ffw_norm_1.f32() : nullptr, moe ? L.post_ffw_norm_2.f32() : nullptr,
                      L.post_ffw_norm.f32(), L.out_scale, x_,
                      last ? m_.output_norm.f32() : m_.layers[il + 1].attn_norm.f32(), xq_, D, n, c.eps, s_,
                      last ? hnorm_ : nullptr);
}

// ---------------------------------------------------------------------------------------------- a prompt chunk

void Engine::layer_big(int il, int n) {
    const Config& c = m_.cfg;
    const Layer& L = m_.layers[il];
    const int D = c.n_embd, hd = L.head_dim, qd = L.n_head * hd, kvd = L.n_head_kv * hd;
    const int K = c.n_expert_used, FE = c.n_ff_exp, E = c.n_expert;
    auto* mq = (mmq::Context*) mmq_;
    const int32_t* dense_bounds = reinterpret_cast<const int32_t*>(ptrs_);
    auto dense = [&](const Tensor& w, const void* xq, int64_t rows, float* dst) {
        mmq::Product p;
        p.w = w.d;
        p.type = w.type;
        p.w_rows = w.ne[1];
        p.w_cols = w.ne[0];
        p.expert_bytes = w.nb2;
        p.n = 1;
        p.xq = xq;
        p.bounds = dense_bounds;
        p.ids = iota_;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = dst;
        p.ld_dst = w.ne[1];
        mq->run(p, s_);
    };

    k::rms_norm(x_, L.attn_norm.f32(), h_, D, n, c.eps, 1.f, s_);
    mmq::quantize(h_, nullptr, xq_, L.wq.type, D, D, n, s_);
    dense(L.wq, xq_, n, q_);
    if (L.wk.type != L.wq.type) mmq::quantize(h_, nullptr, xq_, L.wk.type, D, D, n, s_);
    dense(L.wk, xq_, n, k_);
    if (L.wv) {
        if (L.wv.type != L.wk.type) mmq::quantize(h_, nullptr, xq_, L.wv.type, D, D, n, s_);
        dense(L.wv, xq_, n, v_);
    }

    k::QkvArgs a;
    a.q = q_;
    a.k = k_;
    a.v = L.wv ? v_ : nullptr;
    a.q_norm = L.q_norm.f32();
    a.k_norm = L.k_norm.f32();
    a.freq_factors = L.swa ? nullptr : (m_.rope_freqs ? m_.rope_freqs.f32() : nullptr);
    a.pos = pos_;
    a.q16 = q16_;
    a.kc = kc_[il];
    a.vc = vc_[il];
    a.rows = n;
    a.n_head = L.n_head;
    a.n_kv = L.n_head_kv;
    a.hd = hd;
    a.n_rot = L.n_rot;
    a.base = L.rope_base;
    a.eps = c.eps;
    k::qkv_prep(a, s_);

    // attention through cuBLAS: per kv head g, its rep query heads in one strided batch
    //   S_h [n x T] = Q_h K_g^T, P = masked softmax(S), O_h [n x hd] = P_h V_g
    // the key count is padded to a multiple of 8 so cuBLAS takes its aligned kernels: the extra cache rows are finite
    // (zeroed at start, then real keys) and get probability 0 in mask_softmax
    const int T = n_past_ + n, rep = L.n_head / L.n_head_kv, TP = std::min((T + 7) / 8 * 8, ctx_);
    const float one = 1.f, zero = 0.f;
    auto hb = (cublasHandle_t) cublas_;
    for (int g = 0; g < L.n_head_kv; ++g) {
        cb(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, TP, n, hd, &one,
                                      kc_[il] + (size_t) g * hd, CUDA_R_16F, kvd, 0,
                                      q16_ + (size_t) g * rep * hd, CUDA_R_16F, qd, hd, &zero,
                                      S_ + (size_t) g * rep * n * TP, CUDA_R_32F, TP, (long long) n * TP, rep,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "scores");
    }
    const int32_t* lo = L.swa ? lo_ : lo_ + max_batch_;
    const int32_t* hi = L.swa ? hi_ : hi_ + max_batch_;
    k::mask_softmax(S_, P_, lo, hi, L.n_head, n, T, TP, s_);
    for (int g = 0; g < L.n_head_kv; ++g) {
        cb(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, hd, n, TP, &one,
                                      vc_[il] + (size_t) g * hd, CUDA_R_16F, kvd, 0,
                                      P_ + (size_t) g * rep * n * TP, CUDA_R_16F, TP, (long long) n * TP, &zero,
                                      att_ + (size_t) g * rep * hd, CUDA_R_32F, qd, hd, rep,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "attention");
    }

    dump("Qcur_pos", il, q_, qd, n, s_);
    dump("kqv_out", il, att_, qd, n, s_);
    mmq::quantize(att_, nullptr, xq_, L.wo.type, qd, qd, n, s_);
    dense(L.wo, xq_, n, y_);
    k::add_rms(x_, y_, L.post_attn_norm.f32(), x1_, D, n, c.eps, s_);
    dump("attn_out", il, x1_, D, n, s_);

    const bool moe = L.moe();
    k::ffn_norms(x1_, L.ffn_norm.f32(), moe ? L.pre_ffw_norm_2.f32() : nullptr, moe ? L.router_scale.f32() : nullptr,
                 f_, moe ? g_ : nullptr, moe ? r_ : nullptr, D, n, c.eps, s_);
    mmq::quantize(f_, nullptr, xq_, L.ffn_gate.type, D, D, n, s_);
    dense(L.ffn_gate, xq_, n, gate_);
    if (L.ffn_up.type != L.ffn_gate.type) mmq::quantize(f_, nullptr, xq_, L.ffn_up.type, D, D, n, s_);
    dense(L.ffn_up, xq_, n, up_);
    k::geglu(gate_, up_, hid_, n, c.n_ff, c.n_ff, s_);
    mmq::quantize(hid_, nullptr, xq_, L.ffn_down.type, c.n_ff, c.n_ff, n, s_);
    dense(L.ffn_down, xq_, n, mlp_);

    if (moe) {
        cb(cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, E, n, D, &one, L.router.f32(), D, r_, D, &zero, rlog_, E), "router");
        dump("ffn_moe_logits", il, rlog_, E, n, s_);
        k::router_topk(rlog_, E, K, n, eid_, ew_, s_);
        dump("ffn_moe_weights_norm", il, ew_, K, n, s_);
        k::moe_sort(eid_, n, K, E, bounds_, src_, inv_, counts_, s_);
        const int64_t rows = (int64_t) n * K;
        // MMQ's grid spans max_rows tokens per expert; with every expert sized for all n tokens, its stream-k split
        // hands most blocks empty tiles (4-5x slower). The bounds come back to the host for the true maximum.
        h_bounds_.resize(E + 1);
        ck(cudaMemcpyAsync(h_bounds_.data(), bounds_, (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, s_), "bounds");
        mmq::quantize(g_, src_, xq_, L.gate_up_exps.type, D, D, rows, s_);
        ck(cudaStreamSynchronize(s_), "bounds");
        int max_rows = 1;
        for (int e = 0; e < E; ++e) max_rows = std::max(max_rows, h_bounds_[e + 1] - h_bounds_[e]);
        mmq::Product p;
        p.w = L.gate_up_exps.d;
        p.type = L.gate_up_exps.type;
        p.w_rows = 2 * FE;
        p.w_cols = D;
        p.expert_bytes = L.gate_up_exps.nb2;
        p.n = E;
        p.xq = xq_;
        p.bounds = bounds_;
        p.ids = iota_;
        p.total_rows = rows;
        p.max_rows = max_rows;
        p.dst = gu_;
        p.ld_dst = 2 * FE;
        mq->run(p, s_);
        k::geglu(gu_, gu_ + FE, eh_, rows, FE, 2 * FE, s_);
        mmq::quantize(eh_, nullptr, xq_, L.down_exps.type, FE, FE, rows, s_);
        p.w = L.down_exps.d;
        p.type = L.down_exps.type;
        p.w_rows = D;
        p.w_cols = FE;
        p.expert_bytes = L.down_exps.nb2;
        p.dst = ey_;
        p.ld_dst = D;
        mq->run(p, s_);
        k::moe_combine(ey_, inv_, eid_, ew_, L.down_exps_scale ? L.down_exps_scale.f32() : nullptr, moe_, D, n, K, s_);
        dump("moe_raw", il, moe_, D, n, s_);
    }
    dump("mlp_raw", il, mlp_, D, n, s_);
    k::ffn_post(x1_, mlp_, moe ? moe_ : nullptr, moe ? L.post_ffw_norm_1.f32() : nullptr,
                moe ? L.post_ffw_norm_2.f32() : nullptr, L.post_ffw_norm.f32(), L.out_scale, x_, D, n, c.eps, s_);
}

// ---------------------------------------------------------------------------------------------- the head

void Engine::head(int row) {
    using namespace strata::kernels;
    const Config& c = m_.cfg;
    k::rms_norm(x_ + (size_t) row * c.n_embd, m_.output_norm.f32(), hnorm_, c.n_embd, 1, c.eps, 1.f, s_);
    hrow0_ = 0;
    native_quantize_q8_1(hnorm_, xq_, c.n_embd, 1, s_);
    native_mmvq(m_.tok_embd.type, m_.tok_embd.d, xq_, logits_, c.n_embd, c.n_vocab, 1, s_);
    k::softcap(logits_, c.n_vocab, c.softcap, s_);
    k::argmax_rows(logits_, c.n_vocab, 1, amax_, s_);
    ck(cudaMemcpyAsync(h_amax_.data(), amax_, sizeof(int32_t), cudaMemcpyDeviceToHost, s_), "argmax");
    ck(cudaStreamSynchronize(s_), "forward");
}

}  // namespace strata::gemma
