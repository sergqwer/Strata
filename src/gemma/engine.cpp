// src/gemma/engine.cpp - see include/strata/gemma/engine.hpp.
#include "strata/gemma/engine.hpp"

#include "strata/gemma/decode.hpp"
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

namespace {
constexpr int kPhases = 8;
const char* const kPhaseNames[kPhases] = {"qkv", "attention", "attn_out", "dense_mlp", "router_sort", "moe_gate_up",
                                          "moe_down", "combine_post"};
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
    // the decode layer's FFN fork (decode.hpp): the dense MLP on side_ while the experts run on s_
    ck(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking), "stream");
    ck(cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming), "event");
    ck(cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming), "event");
    if (const char* e = std::getenv("STRATA_NO_FORK")) fork_ = std::string(e) != "1";
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
        head_w_ = carve<uint8_t>((size_t) kHeadRows * m.tok_embd.nb1);
        head_ids_dev_ = carve<int32_t>(kHeadRows);
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
    graph_exec_.assign((kSmall + 1) * 3 * 2, nullptr);
    if (const char* e = std::getenv("STRATA_NO_GRAPH")) graphs_ = std::string(e) != "1";
    ck(cudaStreamSynchronize(s_), "init");
}

Engine::~Engine() {
    for (void* g : graph_exec_)
        if (g) cudaGraphExecDestroy((cudaGraphExec_t) g);
    if (cublas_) cublasDestroy((cublasHandle_t) cublas_);
    delete (mmq::Context*) mmq_;
    if (arena_) cudaFree(arena_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    if (side_) cudaStreamDestroy(side_);
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
        ck(cudaMemcpyAsync(h_logits_.data(), logits_ + (size_t) i * head_rows(), (size_t) head_n() * sizeof(float),
                           cudaMemcpyDeviceToHost, s_), "logits");
        ck(cudaStreamSynchronize(s_), "logits");
        logits_row_ = i;
    }
    return h_logits_.data();
}

void Engine::set_head_rows(const std::vector<int32_t>& ids) {
    if (ids.size() > (size_t) kHeadRows) throw std::runtime_error("set_head_rows: more than kHeadRows rows");
    if (ids == head_ids_) return;
    head_ids_ = ids;
    logits_row_ = -1;
    if (ids.empty()) return;
    // padded with the last id: argmax takes the first of equal values, so a padding row never wins over its original
    std::vector<int32_t> pad(kHeadRows, ids.back());
    std::copy(ids.begin(), ids.end(), pad.begin());
    ck(cudaMemcpyAsync(head_ids_dev_, pad.data(), kHeadRows * sizeof(int32_t), cudaMemcpyHostToDevice, s_), "head ids");
    k::gather_rows(m_.tok_embd.d, m_.tok_embd.nb1, head_ids_dev_, kHeadRows, head_w_, s_);
    ck(cudaStreamSynchronize(s_), "head rows");   // `pad` is pageable host memory and goes out of scope
}

size_t Engine::kv_row_bytes() const {
    size_t b = 0;
    for (int il = 0; il < m_.cfg.n_layer; ++il)
        b += 2 * (size_t) m_.layers[il].n_head_kv * m_.layers[il].head_dim * sizeof(__half);
    return b;
}

// host layout: per layer, K rows [0, n) then V rows [0, n) (each layer's cache is [ctx][n_kv][hd], rows contiguous)
void Engine::kv_save(int n, void* host) const {
    if (n > n_past_) throw std::runtime_error("kv_save: " + std::to_string(n) + " positions > " + std::to_string(n_past_) + " computed");
    char* p = static_cast<char*>(host);
    for (int il = 0; il < m_.cfg.n_layer; ++il) {
        const size_t sz = (size_t) n * m_.layers[il].n_head_kv * m_.layers[il].head_dim * sizeof(__half);
        ck(cudaMemcpyAsync(p, kc_[il], sz, cudaMemcpyDeviceToHost, s_), "kv save");
        ck(cudaMemcpyAsync(p + sz, vc_[il], sz, cudaMemcpyDeviceToHost, s_), "kv save");
        p += 2 * sz;
    }
}

void Engine::kv_load(int n, const void* host) {
    if (n > ctx_) throw std::runtime_error("kv_load: " + std::to_string(n) + " positions > ctx");
    const char* p = static_cast<const char*>(host);
    for (int il = 0; il < m_.cfg.n_layer; ++il) {
        const size_t sz = (size_t) n * m_.layers[il].n_head_kv * m_.layers[il].head_dim * sizeof(__half);
        ck(cudaMemcpyAsync(kc_[il], p, sz, cudaMemcpyHostToDevice, s_), "kv load");
        ck(cudaMemcpyAsync(vc_[il], p + sz, sz, cudaMemcpyHostToDevice, s_), "kv load");
        p += 2 * sz;
    }
    n_past_ = n;
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
        const int gi = (n * 3 + (int) mode) * 2 + (head_ids_.empty() ? 0 : 1);
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
    if (prof_ && !pev_.empty()) {
        ck(cudaStreamSynchronize(s_), "profile");
        const int nb = kPhases + 1;
        for (int il = 0; il < c.n_layer; ++il)
            for (int k = 0; k < kPhases; ++k) {
                float ms = 0;
                if (cudaEventElapsedTime(&ms, (cudaEvent_t) pev_[(size_t) il * nb + k],
                                         (cudaEvent_t) pev_[(size_t) il * nb + k + 1]) == cudaSuccess)
                    ptot_[k] += ms;
            }
    }
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
    const int hv = head_rows();
    native_mmvq(m_.tok_embd.type, head_ids_.empty() ? m_.tok_embd.d : head_w_, xq, logits_, c.n_embd, hv, rows, s_);
    k::softcap(logits_, (int64_t) hv * rows, c.softcap, s_);
    k::argmax_rows(logits_, hv, rows, amax_, s_);
}

// ---------------------------------------------------------------------------------------------- a few rows

void Engine::layer_small(int il, int n) {
    const Config& c = m_.cfg;
    const Layer& L = m_.layers[il];
    DecodeBufs b;
    b.x = x_;
    b.x1 = x1_;
    b.y = y_;
    b.mlp = mlp_;
    b.q = q_;
    b.att = att_;
    b.k = k_;
    b.v = v_;
    b.gate = gate_;
    b.up = up_;
    b.r = r_;
    b.rlog = rlog_;
    b.ew = ew_;
    b.gu = gu_;
    b.ey = ey_;
    b.eid = eid_;
    b.xq = xq_;
    b.xqf = xqf_;
    b.xqg = xqg_;
    b.xqh = xqh_;
    b.xqe = xqe_;
    b.att_scratch = att_scratch_;
    b.groups = groups_;
    b.pos = pos_;
    b.lo = lo_;
    b.hi = hi_;
    b.max_batch = max_batch_;
    b.n_split = n_split_;
    DecodeFork f;
    f.side = side_;
    f.fork = ev_fork_;
    f.join = ev_join_;
    const bool last = il + 1 == c.n_layer;
    // xq_ = q8_1(rms(x) * attn_norm), left by the previous layer's last kernel
    decode_layer(c, L, L.swa ? nullptr : (m_.rope_freqs ? m_.rope_freqs.f32() : nullptr), kc_[il], vc_[il],
                 last ? m_.output_norm.f32() : m_.layers[il + 1].attn_norm.f32(), last ? hnorm_ : nullptr, b, n, s_,
                 fork_ ? &f : nullptr);
}

// ---------------------------------------------------------------------------------------------- a prompt chunk

void Engine::mark(int il, int k) {
    if (!prof_) return;
    const int nb = kPhases + 1;
    if (pev_.empty()) {
        pev_.resize((size_t) m_.cfg.n_layer * nb);
        for (auto& e : pev_) {
            cudaEvent_t ev;
            ck(cudaEventCreate(&ev), "event");
            e = ev;
        }
        ptot_.assign(kPhases, 0.0);
    }
    ck(cudaEventRecord((cudaEvent_t) pev_[(size_t) il * nb + k], s_), "event record");
}

std::vector<std::pair<const char*, double>> Engine::profile_take() {
    std::vector<std::pair<const char*, double>> out;
    if (ptot_.empty()) return out;
    for (int k = 0; k < kPhases; ++k) out.emplace_back(kPhaseNames[k], ptot_[k]);
    ptot_.assign(kPhases, 0.0);
    return out;
}

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

    mark(il, 0);
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
    mark(il, 1);

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

    mark(il, 2);
    dump("Qcur_pos", il, q_, qd, n, s_);
    dump("kqv_out", il, att_, qd, n, s_);
    mmq::quantize(att_, nullptr, xq_, L.wo.type, qd, qd, n, s_);
    dense(L.wo, xq_, n, y_);
    const bool moe = L.moe();
    // x1 = x + rms(y) * post_attn_norm, then the FFN's three norms of x1 (one pass; moe-glue)
    k::add_rms_ffn_norms(x_, y_, L.post_attn_norm.f32(), x1_, L.ffn_norm.f32(), moe ? L.pre_ffw_norm_2.f32() : nullptr,
                         moe ? L.router_scale.f32() : nullptr, f_, moe ? g_ : nullptr, moe ? r_ : nullptr, D, n, c.eps, s_);
    dump("attn_out", il, x1_, D, n, s_);
    mark(il, 3);

    mmq::quantize(f_, nullptr, xq_, L.ffn_gate.type, D, D, n, s_);
    dense(L.ffn_gate, xq_, n, gate_);
    if (L.ffn_up.type != L.ffn_gate.type) mmq::quantize(f_, nullptr, xq_, L.ffn_up.type, D, D, n, s_);
    dense(L.ffn_up, xq_, n, up_);
    k::geglu(gate_, up_, hid_, n, c.n_ff, c.n_ff, s_);
    mmq::quantize(hid_, nullptr, xq_, L.ffn_down.type, c.n_ff, c.n_ff, n, s_);
    dense(L.ffn_down, xq_, n, mlp_);
    mark(il, 4);

    if (moe) {
        cb(cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, E, n, D, &one, L.router.f32(), D, r_, D, &zero, rlog_, E), "router");
        dump("ffn_moe_logits", il, rlog_, E, n, s_);
        k::router_topk2(rlog_, E, K, n, eid_, ew_, s_);
        dump("ffn_moe_weights_norm", il, ew_, K, n, s_);
        k::moe_sort2(eid_, n, K, E, bounds_, src_, inv_, counts_, s_);
        const int64_t rows = (int64_t) n * K;
        // MMQ's grid spans max_rows tokens per expert; with every expert sized for all n tokens, its stream-k split
        // hands most blocks empty tiles (4-5x slower). The bounds come back to the host for the true maximum.
        h_bounds_.resize(E + 1);
        ck(cudaMemcpyAsync(h_bounds_.data(), bounds_, (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, s_), "bounds");
        mmq::quantize(g_, src_, xq_, L.gate_up_exps.type, D, D, rows, s_);
        ck(cudaStreamSynchronize(s_), "bounds");
        int max_rows = 1;
        for (int e = 0; e < E; ++e) max_rows = std::max(max_rows, h_bounds_[e + 1] - h_bounds_[e]);
        mark(il, 5);
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
        mark(il, 6);
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
        mark(il, 7);
        if (!g_dumping) {   // combine + post in one pass (moe-glue)
            k::moe_combine_post(ey_, inv_, eid_, ew_, L.down_exps_scale ? L.down_exps_scale.f32() : nullptr, moe_, x1_,
                                mlp_, L.post_ffw_norm_1.f32(), L.post_ffw_norm_2.f32(), L.post_ffw_norm.f32(),
                                L.out_scale, x_, D, n, K, c.eps, s_);
        } else {
            k::moe_combine(ey_, inv_, eid_, ew_, L.down_exps_scale ? L.down_exps_scale.f32() : nullptr, moe_, D, n, K, s_);
            dump("moe_raw", il, moe_, D, n, s_);
        }
    }
    if (!moe || g_dumping) {
        dump("mlp_raw", il, mlp_, D, n, s_);
        k::ffn_post(x1_, mlp_, moe ? moe_ : nullptr, moe ? L.post_ffw_norm_1.f32() : nullptr,
                    moe ? L.post_ffw_norm_2.f32() : nullptr, L.post_ffw_norm.f32(), L.out_scale, x_, D, n, c.eps, s_);
    }
    if (!moe) {  // a dense layer: its MoE phases take no time
        mark(il, 5);
        mark(il, 6);
        mark(il, 7);
    }
    mark(il, 8);
}

// ---------------------------------------------------------------------------------------------- the head

void Engine::head(int row) {
    using namespace strata::kernels;
    const Config& c = m_.cfg;
    k::rms_norm(x_ + (size_t) row * c.n_embd, m_.output_norm.f32(), hnorm_, c.n_embd, 1, c.eps, 1.f, s_);
    hrow0_ = 0;
    native_quantize_q8_1(hnorm_, xq_, c.n_embd, 1, s_);
    const int hv = head_rows();
    native_mmvq(m_.tok_embd.type, head_ids_.empty() ? m_.tok_embd.d : head_w_, xq_, logits_, c.n_embd, hv, 1, s_);
    k::softcap(logits_, hv, c.softcap, s_);
    k::argmax_rows(logits_, hv, 1, amax_, s_);
    ck(cudaMemcpyAsync(h_amax_.data(), amax_, sizeof(int32_t), cudaMemcpyDeviceToHost, s_), "argmax");
    ck(cudaStreamSynchronize(s_), "forward");
}

}  // namespace strata::gemma
