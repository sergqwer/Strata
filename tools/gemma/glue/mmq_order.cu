// tools/gemma/glue/mmq_order.cu - does a prompt row's expert GEMM result depend on where the expert sort put it?
//
// moe_sort's scatter places each (token, slot) row with an atomicAdd, so the order of rows inside one expert's
// segment changes from run to run. llama.cpp's MMQ (stream-k) splits some tiles' K range across blocks and adds the
// partial sums afterwards; if a row's tile (its position) decides whether its sum is split, the same row gives
// different bits in different runs. This runs the old moe_sort twice on the same expert ids, then the real gate_up
// GEMM of one layer (all 96 experts, MMQ, as layer_big runs it) on both orders and compares every (token, slot)
// row's outputs bitwise.
//
//   build: see build.sh (mmq_order target); run: flock /root/sg-tools/gpu.lock ./mmq_order <gguf> [rows] [layer]
#include "common.hpp"
#include "old.hpp"

#include "strata/gemma/mmq.hpp"

using namespace glue;
namespace ogk = strata::gemma::old_k;
namespace mmq = strata::gemma::mmq;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: mmq_order <model.gguf> [rows=692] [layer=0] [masses file]\n");
        return 2;
    }
    try {
        strata::GgufFile g(argv[1]);
        const int n = argc > 2 ? std::atoi(argv[2]) : 692;
        const int il = argc > 3 ? std::atoi(argv[3]) : 0;
        const std::string mfile = argc > 4 ? argv[4] : "masses_l0.txt";
        const Config c = read_config(g);
        const int D = c.n_embd, E = c.n_expert, K = c.n_expert_used, FE = c.n_ff_exp;
        Dev dev;
        const std::string p = "blk." + std::to_string(il) + ".";
        Tensor gu = upload(g, dev, p + "ffn_gate_up_exps.weight");
        std::printf("mmq_order: %d rows, layer %d, gate_up %d x %lld x %lld (%s), %.1f MiB\n", n, il, (int) gu.ne[2],
                    (long long) gu.ne[1], (long long) gu.ne[0], gu.type == 2 ? "Q4_0" : "?", dev.bytes / 1048576.0);
        std::vector<double> mass;
        if (FILE* f = std::fopen(mfile.c_str(), "r")) {
            double v;
            while (std::fscanf(f, "%lf", &v) == 1) mass.push_back(v);
            std::fclose(f);
        }
        if ((int) mass.size() != E) mass.assign(E, 1.0 / E);
        const int64_t rows = (int64_t) n * K;
        float* x = dev.alloc<float>((size_t) n * D);
        to_dev(x, randn((size_t) n * D, 41, 1.f));
        float* rlog = dev.alloc<float>((size_t) n * E);
        {
            std::mt19937_64 rng(42);
            std::uniform_real_distribution<double> u(1e-12, 1.0);
            std::vector<float> lg((size_t) n * E);
            for (int t = 0; t < n; ++t)
                for (int e = 0; e < E; ++e) lg[(size_t) t * E + e] = (float) (std::log(mass[e]) - std::log(-std::log(u(rng))));
            to_dev(rlog, lg);
        }
        int32_t* ids = dev.alloc<int32_t>(rows);
        float* w = dev.alloc<float>(rows);
        int32_t *bnd[2], *src[2], *inv[2], *cnt[2];
        for (int r = 0; r < 2; ++r) {
            bnd[r] = dev.alloc<int32_t>(E + 1);
            src[r] = dev.alloc<int32_t>(rows);
            inv[r] = dev.alloc<int32_t>(rows);
            cnt[r] = dev.alloc<int32_t>(E);
        }
        void* xq = dev.alloc<uint8_t>(mmq::q8_bytes(rows, D));
        float* out[2] = {dev.alloc<float>((size_t) rows * 2 * FE), dev.alloc<float>((size_t) rows * 2 * FE)};
        int32_t* iota = dev.alloc<int32_t>(rows + 2);
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        mmq::iota(iota, rows + 2, s);
        mmq::Context ctx;
        ogk::router_topk(rlog, E, K, n, ids, w, s);
        std::vector<float> first;      // the process's first MMQ product, kept to compare with the later ones
        std::vector<int32_t> first_inv;
        int tries = 0, differ_rows = 0, same_order_runs = 0;
        size_t moved_total = 0;
        for (int trial = 0; trial < 6; ++trial) {
            for (int r = 0; r < 2; ++r) {
                ogk::moe_sort(ids, n, K, E, bnd[r], src[r], inv[r], cnt[r], s);
                ck(cudaStreamSynchronize(s), "sync");   // s does not sync with the legacy stream from_dev copies on
                std::vector<int32_t> hb = from_dev(bnd[r], E + 1);
                int max_rows = 1;
                for (int e = 0; e < E; ++e) max_rows = std::max(max_rows, hb[e + 1] - hb[e]);
                mmq::quantize(x, src[r], xq, gu.type, D, D, rows, s);
                mmq::Product pr;
                pr.w = gu.d;
                pr.type = gu.type;
                pr.w_rows = 2 * FE;
                pr.w_cols = D;
                pr.expert_bytes = gu.nb2;
                pr.n = E;
                pr.xq = xq;
                pr.bounds = bnd[r];
                pr.ids = iota;
                pr.total_rows = rows;
                pr.max_rows = max_rows;
                pr.dst = out[r];
                pr.ld_dst = 2 * FE;
                ctx.run(pr, s);
            }
            ck(cudaStreamSynchronize(s), "sync");
            std::vector<int32_t> i0 = from_dev(inv[0], rows), i1 = from_dev(inv[1], rows);
            std::vector<float> o0 = from_dev(out[0], (size_t) rows * 2 * FE), o1 = from_dev(out[1], (size_t) rows * 2 * FE);
            if (trial == 0) {
                first = o0;
                first_inv = i0;
            } else {
                int d1 = 0;
                for (int64_t pp = 0; pp < rows; ++pp)
                    if (std::memcmp(&first[(size_t) first_inv[pp] * 2 * FE], &o1[(size_t) i1[pp] * 2 * FE], 2 * FE * 4)) ++d1;
                std::printf("  trial %d vs the first product of the process: %d rows differ\n", trial, d1);
            }
            size_t moved = 0;
            int diff = 0;
            for (int64_t pp = 0; pp < rows; ++pp) {
                moved += i0[pp] != i1[pp];
                if (std::memcmp(&o0[(size_t) i0[pp] * 2 * FE], &o1[(size_t) i1[pp] * 2 * FE], 2 * FE * 4)) ++diff;
            }
            ++tries;
            moved_total += moved;
            differ_rows += diff;
            same_order_runs += moved == 0;
            std::printf("  trial %d: rows placed differently %zu of %lld; (token, slot) rows whose gate_up outputs differ: %d\n",
                        trial, moved, (long long) rows, diff);
        }
        std::printf("mmq_order: %d of %d trials had differing rows\n", differ_rows ? 1 : 0, tries);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
