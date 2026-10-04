// vis_bench - where does llama.cpp's Gemma 4 image time go: decode, preprocess (mtmd_tokenize), encode (GPU)?
//   vis_bench -m model.gguf --mmproj mm.gguf --image a.jpg [--image-tokens 280] [-r 5]
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
    std::string model, mmproj, image, dump; int tokens = 280, reps = 5;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i];
        if (a == "-m") model = argv[++i]; else if (a == "--mmproj") mmproj = argv[++i]; else if (a == "--image") image = argv[++i];
        else if (a == "--image-tokens") tokens = atoi(argv[++i]); else if (a == "-r") reps = atoi(argv[++i]); else if (a == "--dump") dump = argv[++i]; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.vocab_only = true;
    llama_model* m = llama_model_load_from_file(model.c_str(), mp);
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = true; cp.n_threads = 8; cp.image_min_tokens = tokens; cp.image_max_tokens = tokens;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.print_timings = true;
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), m, cp);
    std::ifstream f(image, std::ios::binary); std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), {});
    for (int r = 0; r < reps; ++r) {
        double t0 = now_ms();
        auto w = mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size(), false, mtmd_helper_init_opt_default());
        double t1 = now_ms();
        std::string p = std::string("x ") + mtmd_get_marker(ctx) + " y";
        mtmd_input_text txt{p.c_str(), p.size(), true, true};
        mtmd_input_chunks* ch = mtmd_input_chunks_init();
        const mtmd_bitmap* bm = w.bitmap;
        mtmd_tokenize(ctx, ch, &txt, &bm, 1);
        double t2 = now_ms();
        const mtmd_input_chunk* img = nullptr; size_t n_tok = 0;
        for (size_t c = 0; c < mtmd_input_chunks_size(ch); ++c)
            if (mtmd_input_chunk_get_type(mtmd_input_chunks_get(ch, c)) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { img = mtmd_input_chunks_get(ch, c); n_tok = mtmd_input_chunk_get_n_tokens(img); }
        mtmd_encode_chunk(ctx, img);
        double t3 = now_ms();
        if (!dump.empty() && r == 0) {
            const float* e = mtmd_get_output_embd(ctx);
            const int n_embd = 2816;
            FILE* f = fopen(dump.c_str(), "wb");
            int32_t hdr[2] = {(int32_t) n_tok, n_embd};
            fwrite(hdr, 4, 2, f); fwrite(e, 4, (size_t) n_tok * n_embd, f); fclose(f);
        }
        printf("rep %d: decode %.1f ms, preprocess %.1f ms, encode %.1f ms, %zu tokens (bitmap %ux%u)\n", r, t1 - t0, t2 - t1, t3 - t2, n_tok,
               mtmd_bitmap_get_nx(w.bitmap), mtmd_bitmap_get_ny(w.bitmap));
        mtmd_input_chunks_free(ch); mtmd_bitmap_free(w.bitmap);
    }
    return 0;
}
