// Greedy generation with the CPU reference. Slow (seconds per token), for checking the arithmetic.
// Usage: tona-ref MODEL.gguf "PROMPT" [-n TOKENS] [--top 5] [--hash]
//   --hash: print an FNV-1a hash of the logits bits of every step to stderr ("ref-hash STEP HASH"); two builds that
//   print the same lines computed bit-identical logits
//        tona-ref MODEL.gguf --teacher FILE --max N --experts OUT.bin
//   --teacher: run the text of FILE token by token (teacher forcing) and write the chosen experts per token and
//   layer as int16 [N][layers][active] to OUT.bin
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <numeric>
#include <string>

#include "backend/nan_guard.h"
#include "cpu/ref.h"
#include "model/hasher.h"
#include "model/model_registry.h"
#include "model/tokenizer.h"
#include "server/sampling.h"
#include "tona_args.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    int n_gen = 8;
    int top = 5;
    int max_tokens = 256;
    bool hash = false;
    const char* teacher = nullptr;
    const char* experts_out = nullptr;
};

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 2; i < argc; i++) o.hash = o.hash || !std::strcmp(argv[i], "--hash");
    for (int i = 2; i + 1 < argc; i++) {
        if (!std::strcmp(argv[i], "-n"))
            o.n_gen = tona::tools::parse_int(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--top"))
            o.top = tona::tools::parse_int(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--teacher"))
            o.teacher = argv[i + 1];
        else if (!std::strcmp(argv[i], "--max"))
            o.max_tokens = tona::tools::parse_int(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--experts"))
            o.experts_out = argv[i + 1];
    }
    return o;
}

void write_experts(std::ofstream& out, const tona::ref::Trace& trace) {
    for (const auto& layer : trace.experts)
        for (int e : layer) {
            const int16_t v = (int16_t)e;
            out.write(reinterpret_cast<const char*>(&v), 2);
        }
    out.flush();
}

void print_hash(const Options& o, size_t step, const std::vector<float>& logits) {
    if (!o.hash) return;
    tona::Hasher h;
    h.bytes(logits.data(), logits.size() * sizeof(float));
    std::fprintf(stderr, "ref-hash %zu %016llx\n", step, (unsigned long long)h.digest());
}

int run_teacher(tona::ref::Oracle& fw, const tona::Tokenizer& tok, const Options& o) {
    std::ifstream f(o.teacher, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(f)), {});
    auto ids = tok.encode(text);
    if ((int)ids.size() > o.max_tokens) ids.resize(o.max_tokens);
    std::ofstream out(o.experts_out ? o.experts_out : "/dev/null", std::ios::binary);
    int64_t hits = 0;
    size_t not_finite = 0;
    const auto t0 = Clock::now();
    for (size_t i = 0; i < ids.size(); i++) {
        tona::ref::Trace trace;
        const auto logits = fw.step(ids[i], &trace);
        print_hash(o, i, logits);
        not_finite += std::count_if(logits.begin(), logits.end(), [](float x) { return !std::isfinite(x); });
        write_experts(out, trace);
        if (i + 1 < ids.size()) hits += tona::server::argmax(logits.data(), logits.size()) == ids[i + 1];
        if ((i + 1) % 16 == 0)
            std::fprintf(stderr, "%zu/%zu tokens, %.2f s/token, top-1 hits %.1f %%\n", i + 1, ids.size(),
                         std::chrono::duration<double>(Clock::now() - t0).count() / (i + 1), 100.0 * hits / (double)i);
    }
    std::fprintf(stderr, "%zu steps, %zu non-finite logits\n", ids.size(), not_finite);
    return not_finite ? 3 : 0;
}

// Prints the k most likely tokens to stderr; returns the best
int show_top(const std::vector<float>& logits, const tona::Tokenizer& tok, int k) {
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    tona::nan::report("logits (tona-ref)", tona::nan::count(logits.data(), logits.size()));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return tona::nan::clean_one(logits[a]) > tona::nan::clean_one(logits[b]); });
    std::fprintf(stderr, "  top:");
    for (int i = 0; i < k; i++)
        std::fprintf(stderr, " %d %.3f '%s'", idx[i], logits[idx[i]], tok.decode(idx[i]).c_str());
    std::fprintf(stderr, "\n");
    return idx[0];
}

int run_generate(tona::ref::Oracle& fw, const tona::Tokenizer& tok, const char* prompt, const Options& o) {
    auto ids = tok.encode(prompt);
    std::fprintf(stderr, "Prompt: %zu tokens\n", ids.size());
    std::vector<float> logits;
    auto t0 = Clock::now();
    size_t step = 0;
    for (int32_t id : ids) logits = fw.step(id), print_hash(o, step++, logits);
    const double t_prompt = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("%s", prompt);
    std::fflush(stdout);
    t0 = Clock::now();
    for (int i = 0; i < o.n_gen; i++) {
        const int next = show_top(logits, tok, o.top);
        std::printf("%s", tok.decode(next).c_str());
        std::fflush(stdout);
        if (next == tok.eos()) break;
        logits = fw.step(next);
        print_hash(o, step++, logits);
    }
    const double t_gen = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("\n");
    std::fprintf(stderr, "Prompt %.1f s (%.2f s/token), generation %.1f s (%.2f s/token)\n", t_prompt,
                 t_prompt / ids.size(), t_gen, t_gen / o.n_gen);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: tona-ref MODEL.gguf \"PROMPT\" [-n TOKENS] [--top K]\n");
        return 2;
    }
    try {
        const Options o = parse_options(argc, argv);
        const auto m = tona::load_model(argv[1]);
        tona::Tokenizer tok(m->gguf());
        const auto fw = tona::ref::make_oracle(*m);
        return o.teacher ? run_teacher(*fw, tok, o) : run_generate(*fw, tok, argv[2], o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
