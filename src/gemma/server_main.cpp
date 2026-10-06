// strata-gemma-server - an HTTP server for Gemma 4 on strata-gemma, speaking the subset of llama-server's API that
// hcap uses, so a client switches by URL:
//
//   POST /completion  {"prompt": "<raw prompt>" | {"prompt_string": "...<__media__>...", "multimodal_data": [b64...]},
//                      "n_predict": 300, "temperature": 0, "json_schema": {...} | "grammar": "<gbnf>",
//                      "image_tokens": 70,   (optional: soft tokens per picture, transformers' resize; video frames)
//                      "image_tokens_list": [560, 70, 70],   (optional: per picture, 0 = image_tokens; drag pieces)
//                      "max_queue": 2,       (optional: 503 "busy" if this many requests are already in; low priority)
//                      "max_wait_ms": 8000,  (optional: 503 "late" if its turn would / did come later than this)
//                      "wait_until_ms": T,   (optional, the same as a unix-epoch time: the caller's upload counts too)
//                      "cache_prefix": 1}    (optional: the prompt through its first N pictures repeats across requests;
//                                             with --prefix-cache-mb its KV is saved and reused - see PrefixCache)
// Requests are served in arrival order.
//                  -> {"content", "tokens_evaluated", "tokens_predicted", "stop", "stop_type", "timings": {...}}
//   GET  /props    -> {"media_marker", "n_ctx", "model", ..., "prefix_cache": {entries, hits, misses, ...}}
//   GET  /health   -> {"status": "ok"}
//
// Decoding is greedy (temperature 0; anything else is refused). A JSON schema or GBNF grammar constrains it with
// llama.cpp's grammar sampler, exactly as llama-server's common_sampler does for greedy sampling: the argmax is taken
// when the grammar accepts it, otherwise the best token the grammar allows. The prompt is tokenized as llama.cpp's
// mtmd does it: BOS, then every text part on its own, each image as "<|image>" + its soft tokens + "<image|>".
//
//   strata-gemma-server -m model.gguf --mmproj mmproj.gguf [--host 0.0.0.0] [--port 8091] [--api-key-file F]
//                       [--ctx 4096] [--batch 2048] [--image-tokens 280] [--max-image-tokens 560]
//                       [--mtp mtp.gguf --draft 3] [--max-queue N] [--slots N] [--mtp-slots M]
//                       [--prefix-cache-mb 0] [--prefix-max-tokens 768]
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
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
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

// Saved prompt prefixes ("cache_prefix": k in a request = the prompt up to the end of its k-th picture is shared with
// other requests). Production's area / binary pages start with the system text, the question and a reference picture
// that repeats - 31 distinct ones over 4.4k area pages in 3 h, the same pixels in 99% of the pages - and that part
// (~600 tokens: a picture through the vision encoder and its prefill) cost ~40% of such a page. Its KV (~225 KB a
// token) is kept in pinned host memory, in fixed slots allocated at startup, and copied back in a few ms over PCIe.
// The key is the exact text and picture bytes of the prefix, compared in full: a hit is the same computation.
struct PrefixCache {
    struct Entry {
        PrefixCache* owner = nullptr;
        std::string key;
        int n_tokens = 0;
        int slot = -1;
        ~Entry() {
            if (owner && slot >= 0) {
                std::lock_guard<std::mutex> l(owner->fmu);
                owner->free.push_back(slot);
            }
        }
    };
    using Ptr = std::shared_ptr<Entry>;
    int max_tokens = 0;
    size_t slot_bytes = 0;
    std::vector<void*> slots;               // pinned host buffers, max_tokens positions each
    std::mutex mu;                          // lru + map + counters
    std::list<Ptr> lru;                     // most recently used first
    std::unordered_map<std::string, std::list<Ptr>::iterator> map;
    std::mutex fmu;                         // the free slots (an evicted entry still in use frees its slot later)
    std::vector<int> free;
    uint64_t hits = 0, misses = 0, stored = 0, evicted = 0, uncached = 0;

    bool on() const { return !slots.empty(); }
    Ptr find(const std::string& key) {
        std::lock_guard<std::mutex> l(mu);
        auto it = map.find(key);
        if (it == map.end()) {
            ++misses;
            return nullptr;
        }
        ++hits;
        lru.splice(lru.begin(), lru, it->second);
        return *it->second;
    }
    // a slot for a new prefix, evicting the least recently used entries; null when every slot is held by a request
    Ptr reserve(const std::string& key, int n_tokens) {
        std::lock_guard<std::mutex> l(mu);
        for (;;) {
            {
                std::lock_guard<std::mutex> f(fmu);
                if (!free.empty()) {
                    auto e = std::make_shared<Entry>();
                    e->owner = this;
                    e->key = key;
                    e->n_tokens = n_tokens;
                    e->slot = free.back();
                    free.pop_back();
                    return e;
                }
            }
            if (lru.empty()) {
                ++uncached;
                return nullptr;
            }
            map.erase(lru.back()->key);
            lru.pop_back();   // frees its slot now, or when the last request using it finishes
            ++evicted;
        }
    }
    void insert(const Ptr& e) {
        std::lock_guard<std::mutex> l(mu);
        if (map.count(e->key)) return;   // another request stored the same prefix meanwhile: this slot goes back
        lru.push_front(e);
        map[e->key] = lru.begin();
        ++stored;
    }
    json stats() {
        std::lock_guard<std::mutex> l(mu);
        return json{{"entries", (int) lru.size()}, {"slots", (int) slots.size()}, {"max_tokens", max_tokens},
                    {"hits", hits}, {"misses", misses}, {"stored", stored}, {"evicted", evicted}, {"uncached", uncached}};
    }
};

struct Server {
    std::unique_ptr<Model> model;
    // --slots N: N engines over the one set of weights, each with its own KV cache, buffers, CUDA stream and graphs.
    // A request runs whole (prefill + decode) on one free engine; two at once overlap on the GPU - one's decode
    // (bandwidth-bound) with the other's prefill / vision (compute-bound) - and fill the gaps where a lone sequence
    // waits on the host (grammar, MTP bookkeeping per token).
    std::vector<std::unique_ptr<Engine>> engines;
    std::unique_ptr<Vision> vision;
    // Vision + prefill run one request at a time (compute-bound: two at once only split the GPU); decode, which is
    // bandwidth-bound, runs outside it, so one engine decodes while another encodes and prefills.
    std::mutex cmu;
    std::vector<std::unique_ptr<Mtp>> mtps;  // per engine, null where there is none (each holds its own 0.22 GB)
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
    // FIFO turns with a time budget: requests are served in arrival order (a plain mutex has no order), and one
    // with "max_wait_ms" is refused (503 "late") when the requests ahead of it would keep it waiting longer - at
    // arrival from the running average service time, and again when its turn comes from the real wait.
    std::mutex qmu;
    std::condition_variable qcv;
    uint64_t next_ticket = 0, next_admit = 0;  // tickets in arrival order; next_admit takes the next free engine
    std::vector<int> free_slots;
    double service_ms = 800.0;         // running average of a request's time on the GPU
    std::atomic<uint64_t> late{0};
    PrefixCache prefix;

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

// A request's CPU work, done before it waits for its turn on the GPU: images decoded and resized, the grammar
// built. Before, it ran after the turn came, with the GPU idle meanwhile.
struct Prepared {
    double t_start = 0, pre_ms = 0;
    std::vector<std::string> parts;
    std::vector<ImageU8> images;
    std::string grammar;
    int n_predict = 0;
    bool profile = false;   // "profile": true - the prompt path's phase times come back in timings.profile
    int prefix_k = 0;                 // "cache_prefix": the prompt through this many pictures is a shared prefix
    std::string prefix_key;           // its exact key (when the cache is on)
    PrefixCache::Ptr prefix;          // a saved one: its pictures are neither decoded nor encoded
};

Prepared prepare(Server& S, const json& req) {
    Prepared P;
    P.t_start = now_ms();
    const double t_start = P.t_start;
    (void) t_start;
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
    const auto valid_tokens = [](int t) { return t == 0 || t == 70 || t == 140 || t == 280 || t == 560 || t == 1120; };
    const int img_tokens = req.value("image_tokens", 0);
    if (!valid_tokens(img_tokens)) throw std::runtime_error("image_tokens must be one of 70, 140, 280, 560, 1120");
    // optional: a budget per picture (0: image_tokens) - drag pieces of ~85 px need far fewer tokens than the scene
    std::vector<int> tok_list;
    if (req.contains("image_tokens_list") && req.at("image_tokens_list").is_array())
        for (const auto& v : req.at("image_tokens_list")) {
            const int t = v.get<int>();
            if (!valid_tokens(t)) throw std::runtime_error("image_tokens_list: each must be one of 0, 70, 140, 280, 560, 1120");
            tok_list.push_back(t);
        }
    std::string grammar;
    if (req.contains("json_schema") && !req.at("json_schema").is_null())
        grammar = json_schema_to_grammar(common_json::parse(req.at("json_schema").dump()));
    else if (req.contains("grammar") && req.at("grammar").is_string())
        grammar = req.at("grammar").get<std::string>();

    // ---- the prompt: text parts and images (decoded and resized here, encoded on the GPU in completion())
    const std::vector<std::string> parts = split(prompt, S.marker);
    if (parts.size() - 1 != media.size())
        throw std::runtime_error("the prompt has " + std::to_string(parts.size() - 1) + " media markers but " +
                                 std::to_string(media.size()) + " images");
    if (!media.empty() && !S.vision) throw std::runtime_error("this server was started without --mmproj");
    if (!tok_list.empty() && tok_list.size() != media.size())
        throw std::runtime_error("image_tokens_list has " + std::to_string(tok_list.size()) + " entries for " +
                                 std::to_string(media.size()) + " images");
    const auto tokens_of = [&](size_t mi) { return tok_list.empty() || !tok_list[mi] ? img_tokens : tok_list[mi]; };
    // a shared prefix through picture k: looked up now, so a saved one's pictures are not even decoded
    const int k = req.value("cache_prefix", 0);
    if (k > 0 && k <= (int) media.size() && S.prefix.on()) {
        P.prefix_k = k;
        std::string key = std::to_string(img_tokens);
        for (int i = 0; i < k; ++i) {
            key += '\x01' + std::to_string(parts[i].size()) + '\x01' + parts[i];
            key += '\x02' + std::to_string(media[i].size()) + '\x02' + media[i] + '\x03' + std::to_string(tokens_of(i));
        }
        P.prefix = S.prefix.find(key);
        P.prefix_key = std::move(key);
    }
    const int skip = P.prefix ? P.prefix_k : 0;
    // decode + resize every image on its own thread, before taking the GPU (CPU work, ~20 ms an image)
    const double t_pre = now_ms();
    std::vector<std::future<ImageU8>> pre;
    for (size_t mi = 0; mi < media.size(); ++mi) {
        const std::string& m = media[mi];
        if ((int) mi < skip) {
            pre.push_back(std::async(std::launch::deferred, []() { return ImageU8(); }));
            continue;
        }
        pre.push_back(std::async(std::launch::async, [&S, m, t = tokens_of(mi)]() {
            std::string data = m;
            const size_t comma = data.find(',');
            if (data.rfind("data:", 0) == 0 && comma != std::string::npos) data = data.substr(comma + 1);
            const std::vector<uint8_t> bytes = b64decode(data);
            ImageU8 raw;
            std::string err;
            if (!decode_image(bytes.data(), bytes.size(), raw, err)) throw std::runtime_error(err);
            return S.vision->preprocess(raw, t);
        }));
    }
    for (auto& f : pre) P.images.push_back(f.get());
    P.pre_ms = now_ms() - t_pre;
    P.parts = parts;
    P.grammar = grammar;
    P.n_predict = n_predict;
    P.profile = req.value("profile", false);
    return P;
}

json completion(Server& S, Prepared P, int slot) {
    Engine& E = *S.engines[slot];
    Mtp* mtp = S.mtps[slot].get();
    const double t_start = P.t_start, pre_ms = P.pre_ms;
    const std::vector<std::string>& parts = P.parts;
    std::vector<ImageU8>& images = P.images;
    const std::string& grammar = P.grammar;
    const int n_predict = P.n_predict;
    Batch b;
    // a saved prefix: its positions are in its KV already (placeholders here, never computed); else the prompt is
    // built whole and prefix_end marks where its shared part ends
    const PrefixCache::Ptr& saved = P.prefix;
    size_t first_part = 0, prefix_end = 0;
    if (saved) {
        b.tokens.assign(saved->n_tokens, 0);
        first_part = P.prefix_k;
        prefix_end = saved->n_tokens;
    } else if (llama_vocab_get_add_bos(S.vocab)) {
        b.tokens.push_back(llama_vocab_bos(S.vocab));
    }
    double vis_ms = 0;
    int n_images = 0;
    const double t_wait = now_ms();
    std::unique_lock<std::mutex> compute(S.cmu);  // until the prefill is done
    const double cw_ms = now_ms() - t_wait;
    // the pictures through the vision encoder first, consecutive ones of one size (video frames) a batch at a time;
    // their embeddings land in b.embd in picture order, as the prompt takes them
    std::vector<int> vis_n(images.size(), 0);
    {
        if (S.vision) S.vision->set_profile(P.profile);
        const double tv = now_ms();
        for (size_t i = first_part; i < images.size();) {
            size_t j = i + 1;
            const int cap = S.vision->max_batch(images[i]);
            while (j < images.size() && (int) (j - i) < cap && images[j].nx == images[i].nx && images[j].ny == images[i].ny) ++j;
            std::vector<const ImageU8*> batch;
            for (size_t k = i; k < j; ++k) batch.push_back(&images[k]);
            const int n = S.vision->encode_batch(batch, b.embd);
            for (size_t k = i; k < j; ++k) vis_n[k] = n;
            i = j;
        }
        vis_ms = now_ms() - tv;
    }
    for (size_t i = first_part; i < parts.size(); ++i) {
        for (llama_token t : S.tokenize(parts[i])) b.tokens.push_back(t);
        if (i + 1 < parts.size()) {
            for (llama_token t : S.tokenize("<|image>")) b.tokens.push_back(t);
            const int n = vis_n[i];
            const int begin = (int) b.tokens.size();
            b.tokens.insert(b.tokens.end(), n, -1);
            b.spans.push_back(begin);
            b.spans.push_back(begin + n);
            for (llama_token t : S.tokenize("<image|>")) b.tokens.push_back(t);
            ++n_images;
            if (!saved && (int) i + 1 == P.prefix_k) prefix_end = b.tokens.size();
        }
    }
    const int n_prompt = (int) b.tokens.size();
    if (n_prompt + 1 > S.ctx) throw std::runtime_error("prompt of " + std::to_string(n_prompt) + " tokens > context");

    // ---- prefill, in chunks of at most `batch` rows that never cut an image span
    const double t_pp = now_ms();
    E.set_profile(P.profile);
    E.reset();
    // the shared prefix is always a forward of its own (saved or not), so a reused one gives the same numbers as
    // the first computation; it is saved when it fits a slot and a slot is free
    PrefixCache::Ptr fresh;
    if (saved) {
        E.kv_load(saved->n_tokens, S.prefix.slots[saved->slot]);
    } else if (prefix_end > 0 && (prefix_end > (size_t) S.prefix.max_tokens || prefix_end >= b.tokens.size())) {
        prefix_end = 0;   // too long for a slot, or nothing after it
    } else if (prefix_end > 0) {
        fresh = S.prefix.reserve(P.prefix_key, (int) prefix_end);
    }
    {
        size_t row = saved ? prefix_end : 0, emb_row = 0;
        while (row < b.tokens.size()) {
            size_t end = std::min(b.tokens.size(), row + (size_t) S.batch);
            if (row < prefix_end) end = std::min(end, prefix_end);
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
            if (fresh && end == prefix_end) E.kv_save((int) prefix_end, S.prefix.slots[fresh->slot]);
            row = end;
        }
    }
    if (fresh) S.prefix.insert(fresh);   // the last forward synchronized the stream: the copy is complete
    const double pp_ms = now_ms() - t_pp;
    json prof = json::object();
    if (P.profile) {
        for (auto& [name, ms] : E.profile_take()) prof[name] = ms;
        E.set_profile(false);
        if (S.vision) {
            for (auto& [name, ms] : S.vision->profile_take()) prof[std::string("vision.") + name] = ms;
            S.vision->set_profile(false);
        }
    }
    compute.unlock();

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
        int nd = mtp ? std::min({S.n_draft, n_predict - n_gen, S.ctx - pos - 1, Engine::kSmall - 1}) : 0;
        drafts.clear();
        if (nd > 0) mtp->draft(id, pos, E.hidden_dev(h_row), nd, drafts);
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
                  {"compute_wait_ms", cw_ms},
                  {"profile", prof},
                  {"images", n_images},
                  {"prefix_n", (int) prefix_end},
                  {"prefix_cached", saved ? 1 : 0},
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
void warmup(Server& S, int slot) {
    const double t0 = now_ms();
    Engine& E = *S.engines[slot];
    Mtp* mtp = S.mtps[slot].get();
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
    if (mtp)
        for (int n = 1; n <= std::min(S.n_draft, mtp->max_draft()); ++n) mtp->draft(b.tokens[1], base, E.hidden_dev(0), n, drafts);
    E.reset();
    std::fprintf(stderr, "warmup slot %d: %.0f ms\n", slot, now_ms() - t0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, mmproj, host = "127.0.0.1", key_file, mtp_path;
    int n_draft = 3, max_queue = 0, slots = 1, mtp_slots = 1, prefix_tokens = 768;
    long long prefix_mb = 0;
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
        else if (a == "--slots") slots = std::max(1, std::stoi(next()));
        else if (a == "--mtp-slots") mtp_slots = std::max(0, std::stoi(next()));
        else if (a == "--mtp") mtp_path = next();
        else if (a == "--draft") n_draft = std::stoi(next());
        else if (a == "--max-queue") max_queue = std::stoi(next());
        else if (a == "--prefix-cache-mb") prefix_mb = std::stoll(next());
        else if (a == "--prefix-max-tokens") prefix_tokens = std::stoi(next());
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
        S.n_draft = n_draft;
        for (int k = 0; k < slots; ++k) {
            S.engines.emplace_back(new Engine(*S.model, ctx, batch));
            S.mtps.emplace_back(!mtp_path.empty() && k < mtp_slots
                                    ? new Mtp(mtp_path, *S.model, *S.engines.back(), Engine::kSmall - 1) : nullptr);
            std::fprintf(stderr, "engine slot %d: ctx %d, batch %d, %.2f GiB of buffers, mtp %s\n", k, ctx, batch,
                         S.engines.back()->buffer_bytes() / 1073741824.0, S.mtps.back() ? "on" : "off");
        }
        if (prefix_mb > 0 && prefix_tokens > 0) {   // pinned host slots for saved prompt prefixes
            S.prefix.max_tokens = prefix_tokens;
            S.prefix.slot_bytes = (size_t) prefix_tokens * S.engines[0]->kv_row_bytes();
            const long long n = prefix_mb * (1ll << 20) / (long long) S.prefix.slot_bytes;
            for (long long k = 0; k < n; ++k) {
                void* h = nullptr;
                if (cudaHostAlloc(&h, S.prefix.slot_bytes, cudaHostAllocDefault) != cudaSuccess) break;
                S.prefix.slots.push_back(h);
                S.prefix.free.push_back((int) k);
            }
            std::fprintf(stderr, "prefix cache: %zu slots of %d tokens (%.0f MiB each, pinned)\n", S.prefix.slots.size(),
                         prefix_tokens, S.prefix.slot_bytes / 1048576.0);
        }
        for (int k = slots - 1; k >= 0; --k) S.free_slots.push_back(k);  // slot 0 (with MTP) is taken first
        for (int k = 0; k < slots; ++k) warmup(S, k);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    httplib::Server http;
    http.set_payload_max_length(64ull << 20);
    // requests waiting for their turn hold a thread each; the default pool (8, up to 32) could fill up and leave new
    // ones queued inside httplib, where their wait is not seen
    http.new_task_queue = [] { return new httplib::ThreadPool(64); };
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
                             {"engine", "strata-gemma"}, {"inflight", S.inflight.load()}, {"max_queue", S.max_queue},
                                  {"service_ms", S.service_ms}, {"late", S.late.load()},
                                  {"slots", (int) S.engines.size()}, {"prefix_cache", S.prefix.stats()}}.dump(),
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
        Prepared prep;
        try {
            prep = prepare(S, req);   // CPU work while other requests use the GPU
        } catch (const std::exception& e) {
            return send_err(res, 400, e.what());
        }
        const double t_arr = now_ms();
        // the wait budget: "wait_until_ms" (unix epoch, so the caller's upload and the proxy count too; the hosts
        // are NTP-synced) or "max_wait_ms" from now; 0: wait as long as it takes
        double max_wait = req.is_object() ? req.value("max_wait_ms", 0.0) : 0.0;
        if (req.is_object() && req.contains("wait_until_ms")) {
            const double epoch_now = std::chrono::duration<double, std::milli>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            max_wait = std::max(1.0, req.value("wait_until_ms", 0.0) - epoch_now);
        }
        int eng;  // the engine this request runs on
        {
            std::unique_lock<std::mutex> lk(S.qmu);
            const double n = (double) S.engines.size();
            const double waiting = (double) (S.next_ticket - S.next_admit);
            const double busy = n - (double) S.free_slots.size();
            // requests that must leave an engine before this one gets one, served n at a time
            const double before = std::max(0.0, waiting + busy - n + 1);
            if (max_wait > 0 && before / n * S.service_ms > max_wait) {
                ++S.late;
                return send_err(res, 503, "late");
            }
            const uint64_t ticket = S.next_ticket++;
            S.qcv.wait(lk, [&] { return S.next_admit == ticket && !S.free_slots.empty(); });
            eng = S.free_slots.back();
            S.free_slots.pop_back();
            ++S.next_admit;
        }
        S.qcv.notify_all();  // the next ticket may take another free engine
        struct Turn {  // the engine goes back, whatever happens to this request
            Server& s;
            int slot;
            ~Turn() {
                {
                    std::lock_guard<std::mutex> g(s.qmu);
                    s.free_slots.push_back(slot);
                }
                s.qcv.notify_all();
            }
        } turn{S, eng};
        if (max_wait > 0 && now_ms() - t_arr > max_wait) {
            ++S.late;
            return send_err(res, 503, "late");
        }
        try {
            const double t0 = now_ms();
            const json out = completion(S, std::move(prep), eng);
            {
                std::lock_guard<std::mutex> g(S.qmu);
                S.service_ms = 0.9 * S.service_ms + 0.1 * (now_ms() - t0);
            }
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
