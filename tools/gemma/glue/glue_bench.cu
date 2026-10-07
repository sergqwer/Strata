// tools/gemma/glue/glue_bench.cu - old vs new glue around the MoE (see README.md).
#include "common.hpp"
#include "old.hpp"

#include "strata/gemma/decode.hpp"
#include "strata/gemma/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

using namespace glue;
namespace ok = strata::old_kernels;
namespace ogk = strata::gemma::old_k;

// ============================================================================================== decode (layer_small)

struct DecCtx {
    Config c;
    Layer S, G;                 // a sliding and a global layer of the model; the step repeats them in the model's pattern
    Tensor output_norm, rope_freqs;
    std::vector<char> swa;      // per layer of the step
    int ctx = 2048, n_split = 16, max_batch = 8, pos = 1500;
    float *x = nullptr, *x0 = nullptr, *x1 = nullptr, *y = nullptr, *mlp = nullptr, *q = nullptr, *att = nullptr,
          *k = nullptr, *v = nullptr, *gate = nullptr, *up = nullptr, *r = nullptr, *rlog = nullptr, *ew = nullptr,
          *gu = nullptr, *ey = nullptr, *hnorm = nullptr;
    int32_t *eid = nullptr, *posd = nullptr, *lo = nullptr, *hi = nullptr;
    void *xq = nullptr, *xqf = nullptr, *xqg = nullptr, *xqh = nullptr, *xqe = nullptr, *att_scratch = nullptr,
         *groups = nullptr;
    __half *kc[2] = {}, *vc[2] = {};
    __half *kc2[2] = {}, *vc2[2] = {};   // the new path's caches (same initial contents)
    const Layer& layer(int il) const { return swa[il] ? S : G; }
    __half* kcache(int il) const { return kc[swa[il] ? 0 : 1]; }
    __half* vcache(int il) const { return vc[swa[il] ? 0 : 1]; }
};

// the baseline Engine::layer_small (engine.cpp of gemma 8c7651c), verbatim but for the buffer names
static void old_layer_small(const DecCtx& C, int il, int n, cudaStream_t s_) {
    using namespace strata::old_kernels;
    namespace k = strata::gemma::old_k;
    const Config& c = C.c;
    const Layer& L = C.layer(il);
    const int D = c.n_embd, qd = L.n_head * L.head_dim, kvd = L.n_head_kv * L.head_dim;
    const int K = c.n_expert_used, FE = c.n_ff_exp;
    const int max_batch_ = C.max_batch, n_split_ = C.n_split;

    native_mmvq(L.wq.type, L.wq.d, C.xq, C.q, D, qd, n, s_);
    native_mmvq(L.wk.type, L.wk.d, C.xq, C.k, D, kvd, n, s_);
    if (L.wv) native_mmvq(L.wv.type, L.wv.d, C.xq, C.v, D, kvd, n, s_);

    k::QkvArgs a;
    a.q = C.q;
    a.old_k = C.k;   // (the member `k` of the baseline header, renamed with its namespace)
    a.v = L.wv ? C.v : nullptr;
    a.q_norm = L.q_norm.f32();
    a.k_norm = L.k_norm.f32();
    a.freq_factors = L.swa ? nullptr : (C.rope_freqs ? C.rope_freqs.f32() : nullptr);
    a.pos = C.posd;
    a.kc = C.kcache(il);
    a.vc = C.vcache(il);
    a.rows = n;
    a.n_head = L.n_head;
    a.n_kv = L.n_head_kv;
    a.hd = L.head_dim;
    a.n_rot = L.n_rot;
    a.base = L.rope_base;
    a.eps = c.eps;
    k::qkv_prep(a, s_);
    const int32_t* lo = L.swa ? C.lo : C.lo + max_batch_;
    const int32_t* hi = L.swa ? C.hi : C.hi + max_batch_;
    k::attn_partials(C.q, C.kcache(il), C.vcache(il), lo, hi, C.att_scratch, n, L.n_head, L.n_head_kv, L.head_dim, n_split_, s_);
    k::attn_combine_quant(C.att_scratch, C.att, C.xq, n, L.n_head, L.head_dim, n_split_, s_);
    native_mmvq(L.wo.type, L.wo.d, C.xq, C.y, qd, D, n, s_);

    const bool moe = L.moe();
    k::post_attn_fused(C.x, C.y, L.post_attn_norm.f32(), C.x1, L.ffn_norm.f32(), C.xqf, moe ? L.pre_ffw_norm_2.f32() : nullptr,
                       moe ? C.xqg : nullptr, moe ? L.router_scale.f32() : nullptr, C.r, D, n, c.eps, s_);
    native_mmvq(L.ffn_gate.type, L.ffn_gate.d, C.xqf, C.gate, D, c.n_ff, n, s_);
    native_mmvq(L.ffn_up.type, L.ffn_up.d, C.xqf, C.up, D, c.n_ff, n, s_);
    k::geglu_quant(C.gate, C.up, c.n_ff, C.xqh, c.n_ff, n, s_);
    native_mmvq(L.ffn_down.type, L.ffn_down.d, C.xqh, C.mlp, c.n_ff, D, n, s_);
    if (moe) {
        k::router_gemv(L.router.f32(), C.r, C.rlog, c.n_expert, D, n, s_);
        k::router_topk(C.rlog, c.n_expert, K, n, C.eid, C.ew, s_);
        if (n == 1) {
            native_mmvq_id(L.gate_up_exps.type, L.gate_up_exps.d, L.gate_up_exps.nb2, C.eid, n * K, C.xqg, K, C.gu, D, 2 * FE, s_);
            k::geglu_quant(C.gu, C.gu + FE, 2 * FE, C.xqe, FE, n * K, s_);
            native_mmvq_id(L.down_exps.type, L.down_exps.d, L.down_exps.nb2, C.eid, n * K, C.xqe, 1, C.ey, FE, D, s_);
        } else {
            native_mmvq_group_ids(C.eid, n * K, C.groups, s_);
            native_mmvq_grouped(L.gate_up_exps.type, L.gate_up_exps.d, L.gate_up_exps.nb2, C.groups, n * K, C.xqg, K, C.gu, D,
                                2 * FE, s_);
            k::geglu_quant(C.gu, C.gu + FE, 2 * FE, C.xqe, FE, n * K, s_);
            native_mmvq_grouped(L.down_exps.type, L.down_exps.d, L.down_exps.nb2, C.groups, n * K, C.xqe, 1, C.ey, FE, D, s_);
        }
    }
    const bool last = il + 1 == c.n_layer;
    k::moe_post_fused(moe ? C.ey : nullptr, C.eid, C.ew, L.down_exps_scale ? L.down_exps_scale.f32() : nullptr, K, C.mlp, C.x1,
                      moe ? L.post_ffw_norm_1.f32() : nullptr, moe ? L.post_ffw_norm_2.f32() : nullptr,
                      L.post_ffw_norm.f32(), L.out_scale, C.x,
                      last ? C.output_norm.f32() : C.layer(il + 1).attn_norm.f32(), C.xq, D, n, c.eps, s_,
                      last ? C.hnorm : nullptr);
}

static void old_step(const DecCtx& C, int n, cudaStream_t s) {
    ck(cudaMemcpyAsync(C.x, C.x0, (size_t) n * C.c.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, s), "x0");
    ogk::norm_quant(C.x, C.layer(0).attn_norm.f32(), C.xq, C.c.n_embd, n, C.c.eps, s);
    for (int il = 0; il < C.c.n_layer; ++il) old_layer_small(C, il, n, s);
}

static strata::gemma::DecodeBufs dec_bufs(const DecCtx& C) {
    strata::gemma::DecodeBufs b;
    b.x = C.x; b.x1 = C.x1; b.y = C.y; b.mlp = C.mlp; b.q = C.q; b.att = C.att; b.k = C.k; b.v = C.v;
    b.gate = C.gate; b.up = C.up; b.r = C.r; b.rlog = C.rlog; b.ew = C.ew; b.gu = C.gu; b.ey = C.ey; b.eid = C.eid;
    b.xq = C.xq; b.xqf = C.xqf; b.xqg = C.xqg; b.xqh = C.xqh; b.xqe = C.xqe; b.att_scratch = C.att_scratch;
    b.groups = C.groups; b.pos = C.posd; b.lo = C.lo; b.hi = C.hi; b.max_batch = C.max_batch; b.n_split = C.n_split;
    return b;
}

// the engine's decode step (Engine::run_small's layers) through strata::gemma::decode_layer, on the second caches
static void new_step(const DecCtx& C, int n, cudaStream_t s, const strata::gemma::DecodeFork* fork) {
    ck(cudaMemcpyAsync(C.x, C.x0, (size_t) n * C.c.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, s), "x0");
    strata::gemma::k::norm_quant(C.x, C.layer(0).attn_norm.f32(), C.xq, C.c.n_embd, n, C.c.eps, s);
    const strata::gemma::DecodeBufs b = dec_bufs(C);
    for (int il = 0; il < C.c.n_layer; ++il) {
        const Layer& L = C.layer(il);
        const bool last = il + 1 == C.c.n_layer;
        strata::gemma::decode_layer(C.c, L, L.swa ? nullptr : (C.rope_freqs ? C.rope_freqs.f32() : nullptr),
                                    C.kc2[L.swa ? 0 : 1], C.vc2[L.swa ? 0 : 1],
                                    last ? C.output_norm.f32() : C.layer(il + 1).attn_norm.f32(), last ? C.hnorm : nullptr,
                                    b, n, s, fork);
    }
}

static void setup_decode(DecCtx& C, Dev& dev, const strata::GgufFile& g, int keep_experts) {
    C.c = read_config(g);
    int il_s = -1, il_g = -1;
    for (int il = 0; il < C.c.n_layer; ++il) {
        const bool sw = layer_is_swa(g, il);
        C.swa.push_back(sw ? 1 : 0);
        if (sw && il_s < 0) il_s = il;
        if (!sw && il_g < 0) il_g = il;
    }
    C.S = load_layer(g, dev, il_s, keep_experts);
    if (keep_experts > 32) {   // VRAM: the global layer reuses the sliding layer's experts and router (same shapes)
        C.G = load_layer(g, dev, il_g, 0);
        C.G.router = C.S.router;
        C.G.gate_up_exps = C.S.gate_up_exps;
        C.G.down_exps = C.S.down_exps;
        C.G.down_exps_scale = C.S.down_exps_scale;
    } else {
        C.G = load_layer(g, dev, il_g, keep_experts);
    }
    C.output_norm = upload(g, dev, "output_norm.weight");
    C.rope_freqs = upload(g, dev, "rope_freqs.weight", -1, -1, false);
    if (keep_experts > 0) C.c.n_expert = keep_experts;
    std::printf("decode: layers %d (sliding = blk.%d, global = blk.%d), %d experts kept, top %d, ctx %d, pos %d; weights %.1f MiB\n",
                C.c.n_layer, il_s, il_g, C.c.n_expert, C.c.n_expert_used, C.ctx, C.pos, dev.bytes / 1048576.0);
    const int N = C.max_batch, D = C.c.n_embd, K = C.c.n_expert_used, FE = C.c.n_ff_exp, E = C.c.n_expert;
    const int Q = std::max(C.S.n_head * C.S.head_dim, C.G.n_head * C.G.head_dim);
    const int KV = std::max(C.S.n_head_kv * C.S.head_dim, C.G.n_head_kv * C.G.head_dim);
    const int F = C.c.n_ff;
    C.n_split = (C.ctx + 127) / 128;
    for (float** p : {&C.x, &C.x0, &C.x1, &C.y, &C.mlp, &C.hnorm}) *p = dev.alloc<float>((size_t) N * D);
    C.q = dev.alloc<float>((size_t) N * Q);
    C.att = dev.alloc<float>((size_t) N * Q);
    C.k = dev.alloc<float>((size_t) N * KV);
    C.v = dev.alloc<float>((size_t) N * KV);
    C.gate = dev.alloc<float>((size_t) N * F);
    C.up = dev.alloc<float>((size_t) N * F);
    C.r = dev.alloc<float>((size_t) N * D);
    C.rlog = dev.alloc<float>((size_t) N * 256);
    C.ew = dev.alloc<float>((size_t) N * K);
    C.eid = dev.alloc<int32_t>((size_t) N * K);
    C.gu = dev.alloc<float>((size_t) N * K * 2 * FE);
    C.ey = dev.alloc<float>((size_t) N * K * D);
    size_t xqb = 0;
    for (int64_t vals : {(int64_t) N * Q, (int64_t) N * F, (int64_t) N * K * D, (int64_t) N * K * FE, (int64_t) N * D})
        xqb = std::max(xqb, (size_t) vals / 32 * 36);
    C.xq = dev.alloc<uint8_t>(xqb);
    C.xqf = dev.alloc<uint8_t>((size_t) N * D / 32 * 36);
    C.xqg = dev.alloc<uint8_t>((size_t) N * D / 32 * 36);
    C.xqh = dev.alloc<uint8_t>((size_t) N * F / 32 * 36);
    C.xqe = dev.alloc<uint8_t>((size_t) N * K * FE / 32 * 36);
    C.att_scratch = dev.alloc<uint8_t>(ogk::attn_decode_scratch_bytes(N, std::max(C.S.n_head, C.G.n_head),
                                                                        std::max(C.S.head_dim, C.G.head_dim), C.ctx));
    C.groups = dev.alloc<uint8_t>(ok::native_mmvq_groups_bytes());
    C.posd = dev.alloc<int32_t>(N);
    C.lo = dev.alloc<int32_t>(2 * N);
    C.hi = dev.alloc<int32_t>(2 * N);
    (void) E;
    for (int t = 0; t < 2; ++t) {
        const Layer& L = t == 0 ? C.S : C.G;
        const size_t n = (size_t) C.ctx * L.n_head_kv * L.head_dim;
        C.kc[t] = dev.alloc<__half>(n);
        C.vc[t] = dev.alloc<__half>(n);
        std::vector<float> f = randn(n, 100 + t, 0.5f);
        std::vector<__half> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = __float2half(f[i]);
        to_dev(C.kc[t], h);
        C.kc2[t] = dev.alloc<__half>(n);
        C.vc2[t] = dev.alloc<__half>(n);
        ck(cudaMemcpy(C.kc2[t], C.kc[t], n * 2, cudaMemcpyDeviceToDevice), "kv copy");
        for (size_t i = 0; i < n; ++i) h[i] = __float2half(f[(i * 7919) % n]);
        to_dev(C.vc[t], h);
        to_dev(C.vc2[t], h);
    }
    std::printf("decode: buffers + KV -> %.1f MiB of VRAM in all\n", dev.bytes / 1048576.0);
}

static int C_pos_override = 0, C_seed = 7;   // GLUE_POS / GLUE_SEED
static void set_decode_inputs(DecCtx& C, int n, cudaStream_t s) {
    std::vector<int32_t> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = C.pos + i;
    to_dev(C.posd, pos);
    ogk::key_ranges(C.posd, n, C.c.n_swa, true, nullptr, 0, C.lo, C.hi, s);
    ogk::key_ranges(C.posd, n, C.c.n_swa, false, nullptr, 0, C.lo + C.max_batch, C.hi + C.max_batch, s);
    std::vector<float> x = randn((size_t) n * C.c.n_embd, C_seed, 1.0f);
    for (int i = 0; i < n; ++i) x[(size_t) i * C.c.n_embd + 443] = 60.f;   // a massive-activation dimension
    to_dev(C.x0, x);
    ck(cudaStreamSynchronize(s), "inputs");
}

static int run_decode(const strata::GgufFile& g, int argc, char** argv) {
    const std::string mode = argc > 0 ? argv[0] : "time";
    const int n = argc > 1 ? std::atoi(argv[1]) : 1;
    const int keep = argc > 2 ? std::atoi(argv[2]) : 32;
    if (const char* e = std::getenv("GLUE_ATTN_PF")) strata::gemma::k::set_attn_prefetch(std::atoi(e));
    if (const char* e = std::getenv("GLUE_POS")) C_pos_override = std::atoi(e);
    Dev dev;
    DecCtx C;
    if (C_pos_override > 0) C.pos = C_pos_override;
    if (const char* e = std::getenv("GLUE_SEED")) C_seed = std::atoi(e);
    setup_decode(C, dev, g, keep);
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    set_decode_inputs(C, n, s);
    if (mode == "prof") {   // the old step only, for nsys: 20 replays
        cudaGraphExec_t e = capture(s, [&] { old_step(C, n, s); });
        for (int i = 0; i < 20; ++i) ck(cudaGraphLaunch(e, s), "launch");
        ck(cudaStreamSynchronize(s), "sync");
        std::printf("prof: 20 old steps done\n");
        return 0;
    }
    cudaStream_t side;
    ck(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "stream");
    strata::gemma::DecodeFork fork;
    fork.side = side;
    ck(cudaEventCreateWithFlags(&fork.fork, cudaEventDisableTiming), "event");
    ck(cudaEventCreateWithFlags(&fork.join, cudaEventDisableTiming), "event");
    const int D = C.c.n_embd;
    float* snap = dev.alloc<float>((size_t) 3 * 8 * D);
    // bitwise: old step, then the new step (one stream, then forked), each from the same inputs and initial caches
    bool same = true;
    for (int variant = 0; variant < 2; ++variant) {
        // reset the caches to the initial contents: both paths write row pos of every layer's cache
        old_step(C, n, s);
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpyAsync(snap, C.x, (size_t) n * D * 4, cudaMemcpyDeviceToDevice, s), "snap");
        ck(cudaMemcpyAsync(snap + 8 * D, C.hnorm, (size_t) n * D * 4, cudaMemcpyDeviceToDevice, s), "snap");
        ck(cudaMemcpyAsync(snap + 16 * D, C.xq, (size_t) n * D / 32 * 36, cudaMemcpyDeviceToDevice, s), "snap");
        new_step(C, n, s, variant ? &fork : nullptr);
        ck(cudaStreamSynchronize(s), "sync");
        std::printf("decode step, n = %d, %s:\n", n, variant ? "new, forked FFN" : "new, one stream");
        same &= same_bits(snap, C.x, (size_t) n * D * 4, "x after 30 layers");
        same &= same_bits(snap + 8 * D, C.hnorm, (size_t) n * D * 4, "h (normed, the head's input)");
        same &= same_bits(snap + 16 * D, C.xq, (size_t) n * D / 32 * 36, "xq (q8_1 for the head)", false);
        for (int t = 0; t < 2; ++t) {
            const Layer& L = t == 0 ? C.S : C.G;
            const size_t kvn = (size_t) C.ctx * L.n_head_kv * L.head_dim * 2;
            same &= same_bits(C.kc[t], C.kc2[t], kvn, t == 0 ? "K cache (sliding)" : "K cache (global)", false);
            same &= same_bits(C.vc[t], C.vc2[t], kvn, t == 0 ? "V cache (sliding)" : "V cache (global)", false);
        }
    }
    std::printf("decode: %s\n", same ? "ALL IDENTICAL" : "DIFFERENCES");
    if (mode == "check") return same ? 0 : 3;
    std::vector<Timed> v(3);
    v[0].name = "old step (30 layers, n=" + std::to_string(n) + ")";
    v[0].exec = capture(s, [&] { old_step(C, n, s); });
    v[1].name = "new step, one stream";
    v[1].exec = capture(s, [&] { new_step(C, n, s, nullptr); });
    v[2].name = "new step, forked FFN";
    v[2].exec = capture(s, [&] { new_step(C, n, s, &fork); });
    if (mode == "nsys") {   // for nsys: interleaved replays, no event timing
        const int reps = argc > 3 ? std::atoi(argv[3]) : 40;
        for (int r = 0; r < reps; ++r)
            for (auto& t : v) {
                ck(cudaGraphLaunch(t.exec, s), "launch");
                ck(cudaStreamSynchronize(s), "sync");   // one replay at a time: a time slice of the other process
            }                                           // then delays one replay, not the queue behind it
        std::printf("graphs: old,new1,new2\n");
        return same ? 0 : 3;
    }
    time_interleaved(v, s, 5, 100.0);
    report(v, "step");
    return same ? 0 : 3;
}

// ============================================================================================== units: one kernel, old vs new

// a single-column product: old 128-thread kernels vs the warp-per-row layout (same per-row arithmetic), bitwise and timed
static int run_mmvq_units(const strata::GgufFile& g, int argc, char** argv) {
    const int keep = 32;
    Dev dev;
    DecCtx C;
    setup_decode(C, dev, g, keep);
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    const int D = C.c.n_embd, FE = C.c.n_ff_exp, K = C.c.n_expert_used;
    const int rows_force = argc > 0 ? std::atoi(argv[0]) : 0;
    const bool prof = argc > 1 && std::string(argv[1]) == "prof";   // for nsys: graphs replayed, no event timing
    std::string graph_names;
    strata::kernels::native_mmvq_set_vt_rows(rows_force);
    // activations: q8_1 of random rows (up to 8192 values), with a zero block and a massive value
    const int XMAX = 8192;
    float* xf = dev.alloc<float>(8 * XMAX);
    std::vector<float> h = randn(8 * XMAX, 11, 1.f);
    for (int i = 0; i < 32; ++i) h[64 + i] = 0.f;   // an all-zero block
    h[700] = 3000.f;
    to_dev(xf, h);
    void* xq = dev.alloc<uint8_t>((size_t) 8 * XMAX / 32 * 36);
    ok::native_quantize_q8_1(xf, xq, XMAX, 8, s);
    int32_t* ids = dev.alloc<int32_t>(64);
    to_dev(ids, std::vector<int32_t>{3, 17, 9, 30, 0, 22, 12, 5});
    float* y0 = dev.alloc<float>((size_t) 8 * 16384);
    float* y1 = dev.alloc<float>((size_t) 8 * 16384);
    struct Case { std::string name; const Tensor* w; int n_in, n_out; bool id; int x_div; };
    std::vector<Case> cases = {
        {"wq S (Q8_0 2816->4096)", &C.S.wq, D, (int) C.S.wq.ne[1], false, 1},
        {"wk S (2816->2048)", &C.S.wk, D, (int) C.S.wk.ne[1], false, 1},
        {"wo S (4096->2816)", &C.S.wo, (int) C.S.wo.ne[0], D, false, 1},
        {"ffn_gate (2816->2112)", &C.S.ffn_gate, D, C.c.n_ff, false, 1},
        {"ffn_down (2112->2816)", &C.S.ffn_down, C.c.n_ff, D, false, 1},
        {"wq G (2816->8192)", &C.G.wq, D, (int) C.G.wq.ne[1], false, 1},
        {"wo G (8192->2816)", &C.G.wo, (int) C.G.wo.ne[0], D, false, 1},
        {"gate_up id (Q4_0 8 x 2816->1408)", &C.S.gate_up_exps, D, 2 * FE, true, K},
        {"down id (Q4_0 8 x 704->2816)", &C.S.down_exps, FE, D, true, 1},
    };
    bool all_same = true;
    for (const Case& cs : cases) {
        const Tensor& w = *cs.w;
        auto run_old = [&](float* y) {
            if (cs.id) ok::native_mmvq_id(w.type, w.d, w.nb2, ids, K, xq, cs.x_div, y, cs.n_in, cs.n_out, s);
            else ok::native_mmvq(w.type, w.d, xq, y, cs.n_in, cs.n_out, 1, s);
        };
        auto run_new = [&](float* y) {
            if (cs.id) strata::kernels::native_mmvq_id(w.type, w.d, w.nb2, ids, K, xq, cs.x_div, y, cs.n_in, cs.n_out, s);
            else strata::kernels::native_mmvq(w.type, w.d, xq, y, cs.n_in, cs.n_out, 1, s);
        };
        ck(cudaMemsetAsync(y0, 0xff, (size_t) 8 * 16384 * 4, s), "memset");   // on s: the stream does not sync with
        ck(cudaMemsetAsync(y1, 0xee, (size_t) 8 * 16384 * 4, s), "memset");   // the legacy default stream
        run_old(y0);
        run_new(y1);
        ck(cudaStreamSynchronize(s), "sync");
        const size_t nout = (size_t) (cs.id ? K : 1) * cs.n_out;
        std::printf("%s\n", cs.name.c_str());
        all_same &= same_bits(y0, y1, nout * 4, "output");
        std::vector<Timed> v(2);
        v[0].name = "old";
        v[1].name = "new (vt)";
        v[0].exec = capture(s, [&] { for (int i = 0; i < 10; ++i) run_old(y0); });
        v[1].exec = capture(s, [&] { for (int i = 0; i < 10; ++i) run_new(y1); });
        if (prof) {
            for (int r = 0; r < 15; ++r)
                for (auto& t : v) ck(cudaGraphLaunch(t.exec, s), "launch");
            ck(cudaStreamSynchronize(s), "sync");
            for (const char* nm : {"old", "new"}) {
                std::string c = cs.name.substr(0, cs.name.find(' ')) + "_" + nm;
                graph_names += (graph_names.empty() ? "" : ",") + c;
            }
            continue;
        }
        time_interleaved(v, s, 7, 60.0);
        const double bytes = (double) w.nb1 * cs.n_out * (cs.id ? K : 1);
        for (auto& t : v) {
            std::vector<float> m = t.ms;
            std::sort(m.begin(), m.end());
            std::printf("  %-10s min %8.2f us  median %8.2f us per call  (%.0f GB/s at the median)\n", t.name.c_str(),
                        m.front() * 100.0, m[m.size() / 2] * 100.0, bytes / (m[m.size() / 2] / 10 * 1e-3) / 1e9);
        }
    }
    // merged launches: q / k / v and gate / up as one native_mmvq_multi_w each vs the separate old calls
    for (int which = 0; which < 3; ++which) {
        const Layer& L = which == 1 ? C.G : C.S;
        std::vector<const Tensor*> ws;
        if (which < 2) {
            ws = {&L.wq, &L.wk};
            if (L.wv) ws.push_back(&L.wv);
        } else {
            ws = {&L.ffn_gate, &L.ffn_up};
        }
        const char* nm = which == 0 ? "qkv S" : which == 1 ? "qk G" : "gate+up";
        std::vector<const void*> wp;
        std::vector<float*> y0p, y1p;
        std::vector<int> no;
        size_t off = 0;
        for (const Tensor* t : ws) {
            wp.push_back(t->d);
            no.push_back((int) t->ne[1]);
            y0p.push_back(y0 + off);
            y1p.push_back(y1 + off);
            off += t->ne[1];
        }
        auto run_old = [&] { for (size_t j = 0; j < ws.size(); ++j) ok::native_mmvq(ws[j]->type, ws[j]->d, xq, y0p[j], D, no[j], 1, s); };
        auto run_new = [&] { strata::kernels::native_mmvq_multi_w(ws[0]->type, (int) ws.size(), wp.data(), y1p.data(), no.data(), xq, D, s); };
        ck(cudaMemsetAsync(y0, 0xff, (size_t) 8 * 16384 * 4, s), "memset");
        ck(cudaMemsetAsync(y1, 0xee, (size_t) 8 * 16384 * 4, s), "memset");
        run_old();
        run_new();
        ck(cudaStreamSynchronize(s), "sync");
        std::printf("%s (merged)\n", nm);
        all_same &= same_bits(y0, y1, off * 4, "outputs");
        std::vector<Timed> v(2);
        v[0].exec = capture(s, [&] { for (int i = 0; i < 10; ++i) run_old(); });
        v[1].exec = capture(s, [&] { for (int i = 0; i < 10; ++i) run_new(); });
        if (prof) {
            for (int r = 0; r < 15; ++r)
                for (auto& t : v) ck(cudaGraphLaunch(t.exec, s), "launch");
            ck(cudaStreamSynchronize(s), "sync");
            for (const char* suf : {"_sep_old", "_merged_new"}) {
                std::string c = std::string(nm) + suf;
                for (auto& ch : c) if (ch == ' ' || ch == '+') ch = '_';
                graph_names += (graph_names.empty() ? "" : ",") + c;
            }
        }
    }
    if (prof) std::printf("graphs: %s\n", graph_names.c_str());
    std::printf("mmvq units: %s\n", all_same ? "ALL IDENTICAL" : "DIFFERENCES");
    return all_same ? 0 : 3;
}

// ============================================================================================== prompt glue (layer_big)

// the glue around the prompt chunk's expert GEMMs: add_rms + ffn_norms, router_topk, moe_sort, moe_combine + ffn_post,
// old vs new on the same inputs (one MoE layer's norms and scales; logits with the skewed routing of routing-26b.json;
// synthetic activations), bitwise; then both sequences as graphs for nsys
static int run_prompt(const strata::GgufFile& g, int argc, char** argv) {
    const int n = argc > 0 ? std::atoi(argv[0]) : 692;
    const std::string masses_file = argc > 1 ? argv[1] : "masses_l0.txt";
    const bool prof = argc > 2 && std::string(argv[2]) == "prof";
    const int il = argc > 3 ? std::atoi(argv[3]) : 0;
    const Config c = read_config(g);
    const int D = c.n_embd, E = c.n_expert, K = c.n_expert_used;
    Dev dev;
    const std::string p = "blk." + std::to_string(il) + ".";
    Tensor pan = upload(g, dev, p + "post_attention_norm.weight"), fn = upload(g, dev, p + "ffn_norm.weight"),
           pn2in = upload(g, dev, p + "pre_ffw_norm_2.weight"), rsc = upload(g, dev, p + "ffn_gate_inp.scale"),
           pn1 = upload(g, dev, p + "post_ffw_norm_1.weight"), pn2 = upload(g, dev, p + "post_ffw_norm_2.weight"),
           pfn = upload(g, dev, p + "post_ffw_norm.weight"),
           dsc = upload(g, dev, p + "ffn_down_exps.scale", -1, -1, false);
    float out_scale = 1.f;
    if (const strata::TensorInfo* t = g.find(p + "layer_output_scale.weight")) std::memcpy(&out_scale, g.tensor_data(*t), 4);
    std::vector<double> mass;
    if (FILE* f = std::fopen(masses_file.c_str(), "r")) {
        double v;
        while (std::fscanf(f, "%lf", &v) == 1) mass.push_back(v);
        std::fclose(f);
    }
    if ((int) mass.size() != E) mass.assign(E, 1.0 / E);   // uniform routing
    std::printf("prompt: %d rows, %d experts (top %d), %s routing, layer %d, down_exps.scale %s\n", n, E, K,
                (int) mass.size() == E && masses_file != "uniform" ? masses_file.c_str() : "uniform", il, dsc ? "yes" : "no");
    const size_t ND = (size_t) n * D;
    float *x = dev.alloc<float>(ND), *y = dev.alloc<float>(ND), *mlp = dev.alloc<float>(ND), *moe = dev.alloc<float>(ND);
    float *x1a = dev.alloc<float>(ND), *fa = dev.alloc<float>(ND), *ga = dev.alloc<float>(ND), *ra = dev.alloc<float>(ND);
    float *x1b = dev.alloc<float>(ND), *fb = dev.alloc<float>(ND), *gb = dev.alloc<float>(ND), *rb = dev.alloc<float>(ND);
    float *outa = dev.alloc<float>(ND), *outb = dev.alloc<float>(ND);
    float* ey = dev.alloc<float>(ND * K, false);
    float* rlog = dev.alloc<float>((size_t) n * E);
    int32_t *ida = dev.alloc<int32_t>((size_t) n * K), *idb = dev.alloc<int32_t>((size_t) n * K);
    float *wa = dev.alloc<float>((size_t) n * K), *wb = dev.alloc<float>((size_t) n * K);
    int32_t *bnda = dev.alloc<int32_t>(E + 1), *bndb = dev.alloc<int32_t>(E + 1);
    int32_t *srca = dev.alloc<int32_t>((size_t) n * K), *srcb = dev.alloc<int32_t>((size_t) n * K);
    int32_t *inva = dev.alloc<int32_t>((size_t) n * K), *invb = dev.alloc<int32_t>((size_t) n * K);
    int32_t *cnta = dev.alloc<int32_t>(E), *cntb = dev.alloc<int32_t>(E);
    std::printf("prompt: %.1f MiB of VRAM\n", dev.bytes / 1048576.0);
    {
        std::vector<float> h = randn(ND, 21, 3.f);
        to_dev(x, h);
        h = randn(ND, 22, 1.f);
        to_dev(y, h);
        h = randn(ND, 23, 0.5f);
        to_dev(mlp, h);
        std::mt19937_64 rng(24);
        std::normal_distribution<float> nd(0.f, 0.05f);
        std::vector<float> e(ND * K);
        for (auto& v : e) v = nd(rng);
        to_dev(ey, e);
        // logits: log(mass) + Gumbel noise, so the top-8 of each row is drawn without replacement by routing mass
        std::uniform_real_distribution<double> u(1e-12, 1.0);
        std::vector<float> lg((size_t) n * E);
        for (int t = 0; t < n; ++t)
            for (int ex = 0; ex < E; ++ex) lg[(size_t) t * E + ex] = (float) (std::log(mass[ex]) - std::log(-std::log(u(rng))));
        to_dev(rlog, lg);
    }
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    namespace nk = strata::gemma::k;
    auto run_old = [&] {
        ogk::add_rms(x, y, pan.f32(), x1a, D, n, c.eps, s);
        ogk::ffn_norms(x1a, fn.f32(), pn2in.f32(), rsc.f32(), fa, ga, ra, D, n, c.eps, s);
        ogk::router_topk(rlog, E, K, n, ida, wa, s);
        ogk::moe_sort(ida, n, K, E, bnda, srca, inva, cnta, s);
        ogk::moe_combine(ey, inva, ida, wa, dsc ? dsc.f32() : nullptr, moe, D, n, K, s);
        ogk::ffn_post(x1a, mlp, moe, pn1.f32(), pn2.f32(), pfn.f32(), out_scale, outa, D, n, c.eps, s);
    };
    // the new sequence; the combine reads the OLD sort's inv (the scatter's order within an expert comes from atomics,
    // so two sorts need not agree on it - the sorts are compared on their own below)
    auto run_new = [&] {
        nk::add_rms_ffn_norms(x, y, pan.f32(), x1b, fn.f32(), pn2in.f32(), rsc.f32(), fb, gb, rb, D, n, c.eps, s);
        nk::router_topk2(rlog, E, K, n, idb, wb, s);
        nk::moe_sort2(idb, n, K, E, bndb, srcb, invb, cntb, s);
        nk::moe_combine_post(ey, inva, idb, wb, dsc ? dsc.f32() : nullptr, moe, x1b, mlp, pn1.f32(), pn2.f32(), pfn.f32(),
                             out_scale, outb, D, n, K, c.eps, s);
    };
    run_old();
    run_new();
    ck(cudaStreamSynchronize(s), "sync");
    bool same = true;
    same &= same_bits(x1a, x1b, ND * 4, "x1");
    same &= same_bits(fa, fb, ND * 4, "f (dense MLP input)");
    same &= same_bits(ga, gb, ND * 4, "g (experts' input)");
    same &= same_bits(ra, rb, ND * 4, "r (router input)");
    same &= same_bits(ida, idb, (size_t) n * K * 4, "expert ids", false);
    same &= same_bits(wa, wb, (size_t) n * K * 4, "expert weights");
    same &= same_bits(bnda, bndb, (size_t) (E + 1) * 4, "sort bounds", false);
    {   // src / inv: a permutation within each expert's rows, consistent with each other
        std::vector<int32_t> ids = from_dev(idb, (size_t) n * K), bnd = from_dev(bndb, E + 1), src = from_dev(srcb, (size_t) n * K),
                             inv = from_dev(invb, (size_t) n * K), inv_old = from_dev(inva, (size_t) n * K);
        bool ok = true;
        for (int pp = 0; pp < n * K; ++pp) {
            const int e = ids[pp], r = inv[pp];
            ok &= r >= bnd[e] && r < bnd[e + 1] && src[r] == pp / K;
        }
        size_t moved = 0;
        for (int pp = 0; pp < n * K; ++pp) moved += inv[pp] != inv_old[pp];
        std::printf("  %-34s %s (rows placed differently from the old sort's run: %zu of %d)\n", "sort src / inv",
                    ok ? "CONSISTENT" : "INCONSISTENT", moved, n * K);
        same &= ok;
        std::vector<int32_t> cnt(E, 0);
        for (int pp = 0; pp < n * K; ++pp) cnt[ids[pp]]++;
        std::sort(cnt.begin(), cnt.end());
        std::printf("  rows per expert: max %d, median %d, min %d (mean %.1f)\n", cnt.back(), cnt[E / 2], cnt.front(), (double) n * K / E);
    }
    same &= same_bits(outa, outb, ND * 4, "layer output (combine + post)");
    std::printf("prompt glue: %s\n", same ? "ALL IDENTICAL" : "DIFFERENCES");
    if (prof) {
        cudaGraphExec_t eo = capture(s, run_old), en = capture(s, run_new);
        for (int r = 0; r < 40; ++r) {
            ck(cudaGraphLaunch(eo, s), "launch");
            ck(cudaStreamSynchronize(s), "sync");
            ck(cudaGraphLaunch(en, s), "launch");
            ck(cudaStreamSynchronize(s), "sync");
        }
        std::printf("graphs: old,new\n");
    }
    return same ? 0 : 3;
}

// ============================================================================================== router units (96 experts)

// the decode router (gemv + top-k) on the real 96-row router of one layer: many single rows, old vs new, bitwise
static int run_router(const strata::GgufFile& g, int argc, char** argv) {
    const int trials = argc > 0 ? std::atoi(argv[0]) : 2000;
    const int il = argc > 1 ? std::atoi(argv[1]) : 0;
    const Config c = read_config(g);
    Dev dev;
    const std::string p = "blk." + std::to_string(il) + ".";
    Tensor W = upload(g, dev, p + "ffn_gate_inp.weight");
    const int D = c.n_embd, E = c.n_expert, K = c.n_expert_used;
    float* r = dev.alloc<float>((size_t) trials * D);
    std::vector<float> h = randn((size_t) trials * D, 31, 0.02f);
    for (int t = 0; t < trials; t += 7) h[(size_t) t * D + 100] = 0.f;   // a few exact zeros
    to_dev(r, h);
    float *la = dev.alloc<float>((size_t) trials * E), *lb = dev.alloc<float>((size_t) trials * E);
    int32_t *ia = dev.alloc<int32_t>((size_t) trials * K), *ib = dev.alloc<int32_t>((size_t) trials * K);
    float *wa = dev.alloc<float>((size_t) trials * K), *wb = dev.alloc<float>((size_t) trials * K);
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    for (int t = 0; t < trials; ++t) {   // one row a call, as the decode step runs them
        ogk::router_gemv(W.f32(), r + (size_t) t * D, la + (size_t) t * E, E, D, 1, s);
        ogk::router_topk(la + (size_t) t * E, E, K, 1, ia + (size_t) t * K, wa + (size_t) t * K, s);
        strata::gemma::k::router_gemv2(W.f32(), r + (size_t) t * D, lb + (size_t) t * E, E, D, 1, s);
        strata::gemma::k::router_topk2(lb + (size_t) t * E, E, K, 1, ib + (size_t) t * K, wb + (size_t) t * K, s);
    }
    ck(cudaStreamSynchronize(s), "sync");
    std::printf("router: %d single rows, %d experts, top %d\n", trials, E, K);
    bool same = true;
    same &= same_bits(la, lb, (size_t) trials * E * 4, "logits");
    same &= same_bits(ia, ib, (size_t) trials * K * 4, "expert ids", false);
    same &= same_bits(wa, wb, (size_t) trials * K * 4, "expert weights");
    // ties: logits with repeated values (rounded to a coarse grid) through both top-k kernels
    std::vector<float> lg((size_t) trials * E);
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<int> ud(0, 6);
    for (auto& v : lg) v = 0.5f * ud(rng);
    to_dev(la, lg);
    ogk::router_topk(la, E, K, trials, ia, wa, s);
    strata::gemma::k::router_topk2(la, E, K, trials, ib, wb, s);
    ck(cudaStreamSynchronize(s), "sync");
    same &= same_bits(ia, ib, (size_t) trials * K * 4, "expert ids (tied logits)", false);
    same &= same_bits(wa, wb, (size_t) trials * K * 4, "expert weights (tied logits)");
    // timing graphs: 30 single-row router calls each
    if (argc > 2 && std::string(argv[2]) == "prof") {
        cudaGraphExec_t eo = capture(s, [&] { for (int i = 0; i < 30; ++i) { ogk::router_gemv(W.f32(), r, la, E, D, 1, s); ogk::router_topk(la, E, K, 1, ia, wa, s); } });
        cudaGraphExec_t en = capture(s, [&] { for (int i = 0; i < 30; ++i) { strata::gemma::k::router_gemv2(W.f32(), r, lb, E, D, 1, s); strata::gemma::k::router_topk2(lb, E, K, 1, ib, wb, s); } });
        for (int i = 0; i < 30; ++i) {
            ck(cudaGraphLaunch(eo, s), "launch");
            ck(cudaStreamSynchronize(s), "sync");
            ck(cudaGraphLaunch(en, s), "launch");
            ck(cudaStreamSynchronize(s), "sync");
        }
        std::printf("graphs: old,new\n");
    }
    std::printf("router: %s\n", same ? "ALL IDENTICAL" : "DIFFERENCES");
    return same ? 0 : 3;
}

// ============================================================================================== main

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: glue_bench <model.gguf> decode [time|prof] [n] [experts kept]\n");
        return 2;
    }
    try {
        strata::GgufFile g(argv[1]);
        const std::string what = argv[2];
        if (what == "decode") return run_decode(g, argc - 3, argv + 3);
        if (what == "mmvq") return run_mmvq_units(g, argc - 3, argv + 3);
        if (what == "prompt") return run_prompt(g, argc - 3, argv + 3);
        if (what == "router") return run_router(g, argc - 3, argv + 3);
        std::fprintf(stderr, "unknown mode %s\n", what.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
