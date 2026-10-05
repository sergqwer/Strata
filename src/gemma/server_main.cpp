// strata-gemma-server - an HTTP server for Gemma 4 on strata-gemma, speaking the subset of llama-server's API that
// hcap uses, so a client switches by URL:
//
//   POST /completion  {"prompt": "<raw prompt>" | {"prompt_string": "...<__media__>...", "multimodal_data": [b64...]},
//                      "n_predict": 300, "temperature": 0, "json_schema": {...} | "grammar": "<gbnf>",
//                      "image_tokens": 70,   (optional: soft tokens per picture, transformers' resize; video frames)
//                      "max_queue": 2}       (optional: 503 "busy" if this many requests are already in; low priority)
//                  -> {"content", "tokens_evaluated", "tokens_predicted", "stop", "stop_type", "timings": {...}}
//   GET  /props    -> {"media_marker", "n_ctx", "model"}
//   GET  /health   -> {"status": "ok"}
//
// Decoding is greedy (temperature 0; anything else is refused). A JSON schema or GBNF grammar constrains it with
// llama.cpp's grammar sampler, exactly as llama-server's common_sampler does for greedy sampling: the argmax is taken
// when the grammar accepts it, otherwise the best token the grammar allows. The prompt is tokenized as llama.cpp's
// mtmd does it: BOS, then every text part on its own, each image as "<|image>" + its soft tokens + "<image|>".
//
//   strata-gemma-server -m model.gguf --mmproj mmproj.gguf [--host 0.0.0.0] [--port 8091] [--api-key-file F]
//                       [--ctx 4096] [--batch 2048] [--image-tokens 280] [--max-image-tokens 560]
//                       [--mtp mtp.gguf --draft 3] [--max-queue N]
// --max-image-tokens sizes the vision work buffers for the largest per-request "image_tokens" (~90 KB a patch).
#include "strata/gemma/engine.hpp"
#include "strata/gemma/image.hpp"
#include "strata/gemma/model.hpp"
#include "strata/gemma/mtp.hpp"
#include "strata/gemma/vision.hpp"

#include "json-schema-to-grammar.h"
#include "llama.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::gemma;
using json = nlohmann::json;

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<uint8_t> b64decode(const std::string& in) {
    static int8_t T[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) T[i] = -1;
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[(uint8_t) a[i]] = (int8_t) i;
        T[(uint8_t) '-'] = 62;
        T[(uint8_t) '_'] = 63;
        init = true;
    }
    std::vector<uint8_t> out;
    out.reserve(in.size() * 3 / 4);
    uint32_t buf = 0;
    int bits = 0;
    for (unsigned char c : in) {
        if (T[c] < 0) continue;   // '=', whitespace, a "data:...;base64," prefix's punctuation never reaches here
        buf = (buf << 6) | (uint32_t) T[c];
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t) ((buf >> bits) & 0xFF));
        }
    }
    return out;
}

struct Server {
    std::unique_ptr<Model> model;
    std::unique_ptr<Engine> engine;
    std::unique_ptr<Vision> vision;
    std::unique_ptr<Mtp> mtp;
    int n_draft = 3;
    llama_model* vocab_model = nullptr;
    const llama_vocab* vocab = nullptr;
    std::string marker = "<__media__>";
    std::string model_name;
    std::string api_key;
    int ctx = 4096, batch = 2048;
    int max_queue = 0;                 // > 0: answer 503 "busy" when this many requests are already in (running + waiting)
    std::atomic<int> inflight{0};
    std::mutex mu;   // one sequence on the GPU at a time

    std::vector<llama_token> tokenize(const std::string& text) const {
        std::vector<llama_token> t(text.size() + 8);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), t.data(), (int) t.size(), false, true);
        if (n < 0) {
            t.resize(-n);
            n = llama_tokenize(vocab, text.c_str(), (int) text.size(), t.data(), (int) t.size(), false, true);
        }
        t.resize(std::max(n, 0));
        return t;
    }
    std::string piece(llama_token tok) const {
        char buf[256];
        const int n = llama_token_to_piece(vocab, tok, buf, sizeof buf, 0, false);
        return n > 0 ? std::string(buf, n) : std::string();
    }
};

// split on the marker, keeping empty parts (mtmd's split_text)
std::vector<std::string> split(const std::string& s, const std::string& d) {
    std::vector<std::string> out;
    size_t a = 0, b;
    while ((b = s.find(d, a)) != std::string::npos) {
        out.push_back(s.substr(a, b - a));
        a = b + d.size();
    }
    out.push_back(s.substr(a));
    return out;
}

json completion(Server& S, const json& req) {
    const double t_start = now_ms();
    std::string prompt;
    std::vector<std::string> media;
    const json& p = req.at("prompt");
    if (p.is_string()) {
        prompt = p.get<std::string>();
    } else if (p.is_object()) {
        prompt = p.value("prompt_string", std::string());
        if (p.contains("multimodal_data"))
            for (const auto& m : p.at("multimodal_data")) media.push_back(m.get<std::string>());
    } else {
        throw std::runtime_error("prompt must be a string or {prompt_string, multimodal_data}");
    }
    if (req.value("temperature", 0.0) > 0.0) throw std::runtime_error("only greedy decoding (temperature 0) is implemented");
    const int n_predict = std::min(req.value("n_predict", 512), S.ctx);
    // soft tokens per picture for this request (video frames: 70); 0 = the server's --image-tokens as llama.cpp does
    const int img_tokens = req.value("image_tokens", 0);
    if (img_tokens != 0 && img_tokens != 70 && img_tokens != 140 && img_tokens != 280 && img_tokens != 560 &&
        img_tokens != 1120)
        throw std::runtime_error("image_tokens must be one of 70, 140, 280, 560, 1120");
    std::string grammar;
    if (req.contains("json_schema") && !req.at("json_schema").is_null())
        grammar = json_schema_to_grammar(common_json::parse(req.at("json_schema").dump()));
    else if (req.contains("grammar") && req.at("grammar").is_string())
        grammar = req.at("grammar").get<std::string>();

    // ---- the prompt: BOS, text parts, images (vision runs before taking the GPU lock's sequence)
    Batch b;
    const std::vector<std::string> parts = split(prompt, S.marker);
    if (parts.size() - 1 != media.size())
        throw std::runtime_error("the prompt has " + std::to_string(parts.size() - 1) + " media markers but " +
                                 std::to_string(media.size()) + " images");
    if (llama_vocab_get_add_bos(S.vocab)) b.tokens.push_back(llama_vocab_bos(S.vocab));
    if (!media.empty() && !S.vision) throw std::runtime_error("this server was started without --mmproj");
    // decode + resize every image on its own thread, before taking the GPU (CPU work, ~20 ms an image)
    const double t_pre = now_ms();
    std::vector<std::future<ImageU8>> pre;
    for (const std::string& m : media)
        pre.push_back(std::async(std::launch::async, [&S, m, img_tokens]() {
            std::string data = m;
            const size_t comma = data.find(',');
            if (data.rfind("data:", 0) == 0 && comma != std::string::npos) data = data.substr(comma + 1);
            const std::vector<uint8_t> bytes = b64decode(data);
            ImageU8 raw;
            std::string err;
            if (!decode_image(bytes.data(), bytes.size(), raw, err)) throw std::runtime_error(err);
            return S.vision->preprocess(raw, img_tokens);
        }));
    std::vector<ImageU8> images;
    for (auto& f : pre) images.push_back(f.get());
    const double pre_ms = now_ms() - t_pre;
    std::lock_guard<std::mutex> lk(S.mu);
    double vis_ms = 0;
    int n_images = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        for (llama_token t : S.tokenize(parts[i])) b.tokens.push_back(t);
        if (i + 1 < parts.size()) {
            for (llama_token t : S.tokenize("<|image>")) b.tokens.push_back(t);
            const double tv = now_ms();
            const int n = S.vision->encode(images[i], b.embd);
            vis_ms += now_ms() - tv;
            const int begin = (int) b.tokens.size();
            b.tokens.insert(b.tokens.end(), n, -1);
            b.spans.push_back(begin);
            b.spans.push_back(begin + n);
            for (llama_token t : S.tokenize("<image|>")) b.tokens.push_back(t);
            ++n_images;
        }
    }
    const int n_prompt = (int) b.tokens.size();
    if (n_prompt + 1 > S.ctx) throw std::runtime_error("prompt of " + std::to_string(n_prompt) + " tokens > context");

    // ---- prefill, in chunks of at most `batch` rows that never cut an image span
    const double t_pp = now_ms();
    Engine& E = *S.engine;
    E.reset();
    {
        size_t row = 0, emb_row = 0;
        while (row < b.tokens.size()) {
            size_t end = std::min(b.tokens.size(), row + (size_t) S.batch);
            for (size_t s = 0; s + 1 < b.spans.size(); s += 2)
                if ((size_t) b.spans[s] < end && (size_t) b.spans[s + 1] > end) end = b.spans[s];
            if (end <= row) throw std::runtime_error("an image is larger than the batch");
            Batch c;
            c.tokens.assign(b.tokens.begin() + row, b.tokens.begin() + end);
            size_t n_emb = 0;
            for (int32_t t : c.tokens) n_emb += t < 0;
            const size_t D = (size_t) S.model->cfg.n_embd;
            c.embd.assign(b.embd.begin() + emb_row * D, b.embd.begin() + (emb_row + n_emb) * D);
            emb_row += n_emb;
            for (size_t s = 0; s + 1 < b.spans.size(); s += 2)
                if ((size_t) b.spans[s] >= row && (size_t) b.spans[s] < end) {
                    c.spans.push_back(b.spans[s]);
                    c.spans.push_back(b.spans[s + 1]);
                }
            E.forward(c, end == b.tokens.size());
            row = end;
        }
    }
    const double pp_ms = now_ms() - t_pp;

    // ---- greedy generation under the grammar
    llama_sampler* gs = nullptr;
    if (!grammar.empty()) {
        gs = llama_sampler_init_grammar(S.vocab, grammar.c_str(), "root");
        if (!gs) throw std::runtime_error("invalid grammar");
    }
    std::unique_ptr<llama_sampler, void (*)(llama_sampler*)> gs_guard(gs, [](llama_sampler* x) { if (x) llama_sampler_free(x); });
    const int n_vocab = E.n_vocab();
    std::vector<llama_token_data> cand;
    std::string content;
    int n_gen = 0, n_drafted = 0, n_accepted = 0, n_windows = 0;
    std::string stop_type = "limit";
    // the grammar-constrained greedy choice for scored row r of the last forward
    auto choose = [&](int r) -> int {
        int best = E.argmax(r);
        if (!gs) return best;
        llama_token_data one{best, 0.f, 0.f};
        llama_token_data_array arr{&one, 1, -1, false};
        llama_sampler_apply(gs, &arr);
        if (std::isfinite(arr.data[0].logit)) return best;
        const float* lg = E.logits_host(r);   // the argmax breaks the grammar: the best token that does not
        cand.resize(n_vocab);
        for (int v = 0; v < n_vocab; ++v) cand[v] = {v, lg[v], 0.f};
        llama_token_data_array all{cand.data(), cand.size(), -1, false};
        llama_sampler_apply(gs, &all);
        best = -1;
        for (size_t i = 0; i < all.size; ++i)
            if (std::isfinite(all.data[i].logit) && (best < 0 || all.data[i].logit > lg[best])) best = all.data[i].id;
        return best;
    };
    const double t_tg = now_ms();
    int id = choose(0);
    int h_row = 0;
    std::vector<int32_t> drafts;
    while (true) {
        if (id < 0) {
            stop_type = "grammar";
            break;
        }
        if (gs) llama_sampler_accept(gs, id);
        ++n_gen;
        if (llama_vocab_is_eog(S.vocab, id)) {
            stop_type = "eos";
            break;
        }
        content += S.piece(id);
        if (n_gen >= n_predict) break;
        const int pos = E.n_past();
        int nd = S.mtp ? std::min({S.n_draft, n_predict - n_gen, S.ctx - pos - 1, Engine::kSmall - 1}) : 0;
        drafts.clear();
        if (nd > 0) S.mtp->draft(id, pos, E.hidden_dev(h_row), nd, drafts);
        Batch w;
        w.tokens.push_back(id);
        w.tokens.insert(w.tokens.end(), drafts.begin(), drafts.end());
        if (pos + (int) w.tokens.size() > S.ctx) break;
        E.forward(w, Engine::Logits::All);
        ++n_windows;
        n_drafted += (int) drafts.size();
        // walk the window: row r scores the token after w[r]; a draft survives while it is the constrained choice
        int r = 0;
        bool done = false;
        while (true) {
            const int c = choose(r);
            if (c < 0) {
                stop_type = "grammar";
                done = true;
                break;
            }
            if (!(r + 1 < (int) w.tokens.size() && c == w.tokens[r + 1])) {
                id = c;   // the target's own token: it opens the next window
                break;
            }
            if (gs) llama_sampler_accept(gs, c);
            ++n_gen;
            ++n_accepted;
            if (llama_vocab_is_eog(S.vocab, c)) {
                stop_type = "eos";
                done = true;
                break;
            }
            content += S.piece(c);
            ++r;
            if (n_gen >= n_predict) {
                done = true;
                break;
            }
        }
        E.truncate(pos + r + 1);   // rows 0..r are in the cache; the rest were rejected drafts
        if (done) break;
        h_row = r;
    }
    const double tg_ms = now_ms() - t_tg;
    return json{{"content", content},
                {"tokens_evaluated", n_prompt},
                {"tokens_predicted", n_gen},
                {"stop", true},
                {"stop_type", stop_type},
                {"model", S.model_name},
                {"timings",
                 {{"prompt_n", n_prompt},
                  {"prompt_ms", pp_ms + vis_ms + pre_ms},
                  {"vision_ms", vis_ms},
                  {"image_prep_ms", pre_ms},
                  {"images", n_images},
                  {"prefill_ms", pp_ms},
                  {"predicted_n", n_gen},
                  {"predicted_ms", tg_ms},
                  {"predicted_per_second", tg_ms > 0 ? n_gen * 1000.0 / tg_ms : 0.0},
                  {"windows", n_windows},
                  {"drafted", n_drafted},
                  {"accepted", n_accepted},
                  {"total_ms", now_ms() - t_start}}}};
}

// one synthetic request's worth of work at startup: cuBLAS / MMQ initialization and every CUDA graph the decode loop
// can ask for (windows of 1..8 rows, drafts of 1..n_draft), so the first real request pays none of it
void warmup(Server& S) {
    const double t0 = now_ms();
    Engine& E = *S.engine;
    std::vector<float> emb;
    if (S.vision) {
        ImageU8 img;
        img.nx = img.ny = 96;
        img.rgb.assign(96 * 96 * 3, 128);
        S.vision->encode(S.vision->preprocess(img), emb);
    }
    Batch b;
    b.tokens.push_back(llama_vocab_bos(S.vocab));
    while (b.tokens.size() < 48)
        for (llama_token t : S.tokenize("warm up the engine ")) b.tokens.push_back(t);
    E.reset();
    E.forward(b, true);
    const int base = E.n_past();
    std::vector<int32_t> drafts;
    for (int n = 1; n <= Engine::kSmall; ++n) {
        E.truncate(base);
        Batch w;
        w.tokens.assign(n, b.tokens[1]);
        E.forward(w, Engine::Logits::All);
    }
    if (S.mtp)
        for (int n = 1; n <= std::min(S.n_draft, S.mtp->max_draft()); ++n) S.mtp->draft(b.tokens[1], base, E.hidden_dev(0), n, drafts);
    E.reset();
    std::fprintf(stderr, "warmup: %.0f ms\n", now_ms() - t0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, mmproj, host = "127.0.0.1", key_file, mtp_path;
    int n_draft = 3, max_queue = 0;
    int port = 8091, ctx = 4096, batch = 2048, img_tokens = 280, max_img_tokens = 280;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "-m") model_path = next();
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::stoi(next());
        else if (a == "--api-key-file") key_file = next();
        else if (a == "--ctx" || a == "-c") ctx = std::stoi(next());
        else if (a == "--batch" || a == "-b") batch = std::stoi(next());
        else if (a == "--image-tokens") img_tokens = std::stoi(next());
        else if (a == "--max-image-tokens") max_img_tokens = std::stoi(next());
        else if (a == "--mtp") mtp_path = next();
        else if (a == "--draft") n_draft = std::stoi(next());
        else if (a == "--max-queue") max_queue = std::stoi(next());
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "usage: %s -m model.gguf [--mmproj mmproj.gguf] [--host H] [--port P] [--api-key-file F]\n", argv[0]);
        return 2;
    }
    Server S;
    S.ctx = ctx;
    S.batch = batch;
    S.max_queue = max_queue;
    S.model_name = model_path.substr(model_path.find_last_of('/') + 1);
    if (!key_file.empty()) {
        std::ifstream kf(key_file);
        std::getline(kf, S.api_key);
        while (!S.api_key.empty() && std::isspace((unsigned char) S.api_key.back())) S.api_key.pop_back();
        if (S.api_key.empty()) {
            std::fprintf(stderr, "empty api key file %s\n", key_file.c_str());
            return 2;
        }
    }
    try {
        llama_backend_init();
        llama_log_set([](ggml_log_level lvl, const char* text, void*) { if (lvl >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr); }, nullptr);
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        S.vocab_model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!S.vocab_model) throw std::runtime_error("cannot load the tokenizer from " + model_path);
        S.vocab = llama_model_get_vocab(S.vocab_model);
        S.model = Model::load(model_path);
        if (!mmproj.empty()) {
            S.vision.reset(new Vision(mmproj, img_tokens, std::max(4096, max_img_tokens * 9)));  // 3x3 patches a token
            std::fprintf(stderr, "vision: %s, %.2f GiB on the GPU\n", mmproj.c_str(), S.vision->weight_bytes() / 1073741824.0);
        }
        S.engine.reset(new Engine(*S.model, ctx, batch));
        S.n_draft = n_draft;
        if (!mtp_path.empty()) {
            S.mtp.reset(new Mtp(mtp_path, *S.model, *S.engine, Engine::kSmall - 1));
            std::fprintf(stderr, "mtp: %s, %.2f GiB, %d drafts per step\n", mtp_path.c_str(), S.mtp->bytes() / 1073741824.0, n_draft);
        }
        std::fprintf(stderr, "engine: ctx %d, batch %d, %.2f GiB of buffers\n", ctx, batch, S.engine->buffer_bytes() / 1073741824.0);
        warmup(S);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    httplib::Server http;
    http.set_payload_max_length(64ull << 20);
    auto authorized = [&](const httplib::Request& r) {
        if (S.api_key.empty()) return true;
        const std::string h = r.get_header_value("Authorization");
        return h == "Bearer " + S.api_key || r.get_header_value("X-Api-Key") == S.api_key;
    };
    auto send_err = [](httplib::Response& res, int code, const std::string& msg) {
        res.status = code;
        res.set_content(json{{"error", {{"code", code}, {"message", msg}}}}.dump(), "application/json");
    };
    http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    http.Get("/props", [&](const httplib::Request& r, httplib::Response& res) {
        if (!authorized(r)) return send_err(res, 401, "invalid api key");
        res.set_content(json{{"media_marker", S.marker}, {"n_ctx", S.ctx}, {"model", S.model_name},
                             {"engine", "strata-gemma"}, {"inflight", S.inflight.load()}, {"max_queue", S.max_queue}}.dump(),
                        "application/json");
    });
    http.Post("/completion", [&](const httplib::Request& r, httplib::Response& res) {
        if (!authorized(r)) return send_err(res, 401, "invalid api key");
        json req;
        try {
            req = json::parse(r.body);
        } catch (const std::exception& e) {
            return send_err(res, 400, e.what());
        }
        // one sequence at a time: past max_queue the caller is better served elsewhere at once than after a wait.
        // A request may bring a lower limit of its own ("max_queue": 2): a low-priority caller then only gets the
        // GPU when the queue is short, and never crowds out the others.
        struct Slot {
            std::atomic<int>& n;
            int v;
            explicit Slot(std::atomic<int>& a) : n(a), v(++a) {}
            ~Slot() { --n; }
        } slot(S.inflight);
        const int own = req.is_object() ? req.value("max_queue", 0) : 0;
        if ((S.max_queue > 0 && slot.v > S.max_queue) || (own > 0 && slot.v > own)) return send_err(res, 503, "busy");
        try {
            const json out = completion(S, req);
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            send_err(res, 400, e.what());
        }
    });
    std::fprintf(stderr, "listening on http://%s:%d\n", host.c_str(), port);
    if (!http.listen(host, port)) {
        std::fprintf(stderr, "error: cannot listen on %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
