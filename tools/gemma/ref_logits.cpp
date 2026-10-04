// ref_logits - the parity reference for strata-gemma: run a prompt through llama.cpp (libllama + libmtmd) and dump
// the logits the engine has to reproduce.
//
//   ref_logits -m model.gguf [--mmproj mm.gguf --image a.png ...] -p prompt.txt [-n 32] [--last 8]
//              [--ngl 99] [--n-cpu-moe 0] [--marker <__media__>] -o ref.bin
//
// The prompt file is raw text with special tokens parsed (exactly what hcapsolv's core.gemma_prompt builds); each
// media marker is replaced by the next --image. Output (little-endian):
//   "SGREF1\0\0", i32 n_vocab, i32 n_prompt, i32 n_last, i32 n_gen,
//   i32 prompt_tokens[n_prompt] (image positions are -1),
//   f32 logits[n_last][n_vocab]  (the last n_last prompt positions; text-only prompts),
//   i32 gen_tokens[n_gen], f32 gen_logits[n_gen][n_vocab]  (greedy: logits that chose gen_tokens[i])
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// --dump DIR: every tensor whose name starts with one of the dump prefixes is written to DIR/<name>.bin as f32
static std::string g_dump;
static bool dump_cb(struct ggml_tensor* t, bool ask, void*) {
    static const char* pre[] = {"l_out-", "attn_out-", "ffn_moe_logits-", "ffn_mlp-", "ffn_moe-", "Qcur_pos-", "Kcur_pos-",
                                "inp_scaled", "result_norm", "attn_post_norm-", "ffn_moe_weights_norm-", "ffn_moe_topk-",
                                "kqv_out-", "Vcur_normed-", "ffn_moe_combined-", "ffn_post_norm-", "ffn_norm_1-", "ffn_norm_2-"};
    bool want = false;
    for (const char* p : pre) if (strncmp(t->name, p, strlen(p)) == 0) want = true;
    if (ask) return want;
    if (!want || g_dump.empty()) return true;
    std::vector<float> buf(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * 4);
    else if (t->type == GGML_TYPE_I32 && ggml_is_contiguous(t)) {
        std::vector<int32_t> ib(buf.size()); ggml_backend_tensor_get(t, ib.data(), 0, ib.size() * 4);
        for (size_t i = 0; i < ib.size(); ++i) buf[i] = (float) ib[i];
    } else return true;
    std::string name = t->name;
    for (char& ch : name) if (ch == ' ' || ch == '/') ch = '_';
    FILE* f = fopen((g_dump + "/" + name + ".bin").c_str(), "wb");
    if (f) { int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}; fwrite(ne, 8, 4, f); fwrite(buf.data(), 4, buf.size(), f); fclose(f); }
    return true;
}

static void die(const char* m) { fprintf(stderr, "ref_logits: %s\n", m); exit(1); }

int main(int argc, char** argv) {
    std::string model, mmproj, prompt_file, out, marker;
    std::vector<std::string> images;
    bool no_fa = false;
    int n_gen = 16, n_last = 1, ngl = 99, n_cpu_moe = 0, n_ctx = 4096, img_tokens = 280;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return std::string(argv[++i]); };
        if (a == "-m") model = next(); else if (a == "--mmproj") mmproj = next(); else if (a == "--image") images.push_back(next());
        else if (a == "-p") prompt_file = next(); else if (a == "-o") out = next(); else if (a == "-n") n_gen = std::stoi(next());
        else if (a == "--last") n_last = std::stoi(next()); else if (a == "--ngl") ngl = std::stoi(next());
        else if (a == "--n-cpu-moe") n_cpu_moe = std::stoi(next()); else if (a == "-c") n_ctx = std::stoi(next());
        else if (a == "--marker") marker = next();
        else if (a == "--image-tokens") img_tokens = std::stoi(next());
        else if (a == "--dump") g_dump = next();
        else if (a == "--no-fa") no_fa = true;
        else die(("unknown arg " + a).c_str());
    }
    if (model.empty() || prompt_file.empty() || out.empty()) die("need -m, -p, -o");
    std::ifstream pf(prompt_file); std::stringstream ss; ss << pf.rdbuf(); std::string prompt = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    std::vector<llama_model_tensor_buft_override> ov;
    std::vector<std::string> pats;
    if (n_cpu_moe > 0) {
        for (int l = 0; l < n_cpu_moe; ++l) pats.push_back("blk\\." + std::to_string(l) + "\\.ffn_(up|down|gate|gate_up)_exps");
        for (auto& p : pats) ov.push_back({p.c_str(), ggml_backend_cpu_buffer_type()});
        ov.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = ov.data();
    }
    llama_model* m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) die("model load failed");
    const llama_vocab* vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx; cp.no_perf = true;
    if (!g_dump.empty()) { cp.cb_eval = dump_cb; cp.cb_eval_user_data = nullptr; }
    if (no_fa) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context* ctx = llama_init_from_model(m, cp);
    if (!ctx) die("context failed");

    std::vector<int32_t> ptoks;
    std::vector<float> last_logits;
    llama_pos n_past = 0;
    if (mmproj.empty()) {
        std::vector<llama_token> t(prompt.size() + 16);
        int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), t.data(), (int)t.size(), true, true);
        if (n < 0) die("tokenize failed");
        t.resize(n);
        ptoks.assign(t.begin(), t.end());
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = t[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i >= n - n_last;
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b)) die("decode failed");
        for (int i = n - n_last; i < n; ++i) {
            const float* lg = llama_get_logits_ith(ctx, i);
            last_logits.insert(last_logits.end(), lg, lg + n_vocab);
        }
        llama_batch_free(b);
        n_past = n;
    } else {
        mtmd_context_params mcp = mtmd_context_params_default();
        mcp.use_gpu = true; mcp.n_threads = 8;
        mcp.image_min_tokens = img_tokens; mcp.image_max_tokens = img_tokens;  // = llama-server --image-{min,max}-tokens
        if (!marker.empty()) mcp.media_marker = marker.c_str();
        mtmd_context* mctx = mtmd_init_from_file(mmproj.c_str(), m, mcp);
        if (!mctx) die("mmproj load failed");
        std::vector<mtmd_bitmap*> bms;
        for (auto& f : images) {
            mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(mctx, f.c_str(), false, mtmd_helper_init_opt_default());
            if (!w.bitmap) die(("image load failed: " + f).c_str());
            bms.push_back(w.bitmap);
        }
        mtmd_input_text txt{prompt.c_str(), prompt.size(), true, true};
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        if (mtmd_tokenize(mctx, chunks, &txt, (const mtmd_bitmap**)bms.data(), bms.size())) die("mtmd_tokenize failed");
        for (size_t c = 0; c < mtmd_input_chunks_size(chunks); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t nt; const llama_token* tk = mtmd_input_chunk_get_tokens_text(ch, &nt);
                ptoks.insert(ptoks.end(), tk, tk + nt);
            } else {
                ptoks.insert(ptoks.end(), mtmd_input_chunk_get_n_tokens(ch), -1);
            }
        }
        llama_pos new_past = 0;
        if (mtmd_helper_eval_chunks(mctx, ctx, chunks, 0, 0, 512, true, &new_past)) die("eval chunks failed");
        n_past = new_past;
        n_last = 1;
        const float* lg = llama_get_logits_ith(ctx, -1);
        last_logits.assign(lg, lg + n_vocab);
    }

    g_dump.clear();
    std::vector<int32_t> gen;
    std::vector<float> gen_logits;
    const float* lg = last_logits.data() + (size_t)(n_last - 1) * n_vocab;
    std::vector<float> cur(lg, lg + n_vocab);
    for (int s = 0; s < n_gen; ++s) {
        int best = 0;
        for (int v = 1; v < n_vocab; ++v) if (cur[v] > cur[best]) best = v;
        gen.push_back(best);
        gen_logits.insert(gen_logits.end(), cur.begin(), cur.end());
        if (llama_vocab_is_eog(vocab, best)) break;
        llama_batch b = llama_batch_get_one(&gen.back(), 1);
        if (llama_decode(ctx, b)) die("decode (gen) failed");
        const float* l2 = llama_get_logits_ith(ctx, -1);
        cur.assign(l2, l2 + n_vocab);
        ++n_past;
    }
    std::string text;
    for (int t : gen) { char buf[256]; int k = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true); if (k > 0) text.append(buf, k); }
    fprintf(stderr, "prompt %zu tokens, generated %zu: %s\n", ptoks.size(), gen.size(), text.c_str());

    FILE* f = fopen(out.c_str(), "wb");
    if (!f) die("cannot write output");
    fwrite("SGREF1\0\0", 1, 8, f);
    int32_t hdr[4] = {n_vocab, (int32_t)ptoks.size(), n_last, (int32_t)gen.size()};
    fwrite(hdr, 4, 4, f);
    fwrite(ptoks.data(), 4, ptoks.size(), f);
    fwrite(last_logits.data(), 4, last_logits.size(), f);
    fwrite(gen.data(), 4, gen.size(), f);
    fwrite(gen_logits.data(), 4, gen_logits.size(), f);
    fclose(f);
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
