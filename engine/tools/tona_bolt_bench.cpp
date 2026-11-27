// Speed of a model on the GPU backend through the generic backend interface (the tona-gpu-gen of the other
// architectures is written for one model family): the prompt through the prefill, then greedy decode in passes of one
// token, both timed, with the counters of the expert tiers and, with TONA_MEGA_PROF=1, the time per step of a pass.
// Usage: tona-bolt-bench MODEL.gguf IDS.txt [--prompt N] [--gen N] [--ctx N] [--prefill-rounds R] [--widths W,...]
//   IDS.txt: token ids (tona-tokenize); the first N are the prompt (default all but --gen). The decode starts after a
//   prompt prefilled in one call; --prefill-rounds R repeats the prefill from a reset state R more times (warm).
//   --repeat R: R times (reset, prefill, greedy decode), one "generated ids hash" line per repeat with the expert
//   counters of the run: the expert cache state carries over from one repeat to the next, so equal hashes in a row
//   say the output does not depend on it (TONA_DETERMINISTIC=1)
//   --logits-out FILE: also write the logits of the first decode pass (float32, one vocabulary) for the deviation
//   between two plans (numpy.fromfile, dtype float32); every repeat prints a hash of them as "first logits hash"
//   --check-widths 2,3,4: teacher-forced passes of each width over the 16 ids after the prompt; hash of the logits of
//   every position against passes of one token (bit-identical widths: 0 mismatches)
//   --widths 1,2,4,8: instead of the greedy decode, for every width W passes of W tokens taken from the ids file after
//   the prompt (teacher forced, all accepted: the cost of verifying W drafted tokens of the real text), --gen tokens
//   each, one after the other; every width prints its own "decode:" line and "pass time".
// Prints "decode: ... tok/s" and "prefill: ... tok/s" lines (the rates are named in them).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/builtin.h"
#include "backend/env.h"
#include "backend/registry.h"
#include "gpu/mega.h"
#include "model/model_registry.h"
#include "tona_args.h"

namespace {

struct Options {
    std::string model, ids;
    int prompt = 0, gen = 128, rounds = 0, repeat = 1;
    std::vector<int> widths;
    int64_t ctx = 0;
    std::vector<int> check_widths;  // --check-widths 2,3,4: logits of every position against passes of one token
    std::string logits_out;  // --logits-out FILE: the logits of the first decode pass of the first repeat (float32)
};

std::vector<int> parse_widths(const std::string& list) {
    std::vector<int> w;
    for (size_t at = 0; at < list.size();) {
        const size_t end = std::min(list.find(',', at), list.size());
        w.push_back(tona::tools::parse_int(list.substr(at, end - at).c_str()));
        if (w.back() < 1 || w.back() > 8) throw std::invalid_argument("--widths: 1 to 8");
        at = end + 1;
    }
    return w;
}

Options parse(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("usage: tona-bolt-bench MODEL.gguf IDS.txt [options]");
    Options o;
    o.model = argv[1], o.ids = argv[2];
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--prompt"))
            o.prompt = tona::tools::parse_int(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--gen"))
            o.gen = tona::tools::parse_int(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--ctx"))
            o.ctx = tona::tools::parse_int64(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--widths"))
            o.widths = parse_widths(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--check-widths"))
            o.check_widths = parse_widths(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--logits-out"))
            o.logits_out = argv[i + 1];
        else if (!std::strcmp(argv[i], "--repeat"))
            o.repeat = tona::tools::parse_int(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--prefill-rounds"))
            o.rounds = tona::tools::parse_int(argv[i + 1]);
        else
            throw std::invalid_argument(std::string("unknown option: ") + argv[i]);
    }
    return o;
}

std::vector<int32_t> read_ids(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(path + ": cannot read");
    std::vector<int32_t> ids;
    for (long v; f >> v;) ids.push_back((int32_t)v);
    return ids;
}

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

int32_t argmax(const float* lg, size_t n) { return (int32_t)(std::max_element(lg, lg + n) - lg); }

void print_profile(tona::gpu::Mega& mg) {
    double total = 0;
    const auto prof = mg.profile();
    for (const auto& [name, us] : prof)
        if (!name.starts_with("nu#") && !name.starts_with("gd#")) total += us;
    std::printf("profile: %.2f ms per token (mean over all passes, prompt tokens of a prefill excluded)\n",
                total / 1e3);
    for (const auto& [name, us] : prof) std::printf("  %-24s %8.0f us  %4.1f %%\n", name.c_str(), us, 100 * us / total);
}

struct Counters {
    int64_t hits, misses, cpu, gd;
    double upload;
};
Counters counters(const tona::backend::Backend& be) {
    const tona::backend::Stats& s = be.stats();
    return {s.expert_hits, s.expert_misses, s.cpu_experts, s.gpu_direct, s.upload_bytes};
}

void report_decode(const Counters& a, const Counters& b, int tokens, double sec, int width) {
    const double h = (double)(b.hits - a.hits), m = (double)(b.misses - a.misses);
    std::printf("decode: %d tokens in %.3f s = %.2f tok/s (width %d, one sequence, no drafter)\n", tokens, sec,
                tokens / sec, width);
    std::printf(
        "experts during decode: %.1f %% hits, %.1f misses per token and layer-sum, %.0f MB reloaded per token, "
        "%.1f computed by the CPU and %.1f read by the GPU over PCIe per token\n",
        100.0 * h / std::max(1.0, h + m), m / tokens, (b.upload - a.upload) / 1e6 / tokens,
        (double)(b.cpu - a.cpu) / tokens, (double)(b.gd - a.gd) / tokens);
}

// Passes of each width over the text that follows the prompt, all tokens accepted
void decode_forced(tona::backend::Backend& be, const std::vector<int32_t>& ids, int pos, const Options& o) {
    for (const int w : o.widths) {
        const Counters c0 = counters(be);
        const int passes = o.gen / w;
        const auto t0 = Clock::now();
        for (int i = 0; i < passes; i++, pos += w) {
            (void)be.forward(ids.data() + pos, w);
            be.commit(w);
        }
        const double sec = since(t0);
        report_decode(c0, counters(be), passes * w, sec, w);
        std::printf("pass time: %.2f ms per pass of %d tokens\n", 1e3 * sec / passes, w);
    }
}

void prefill_rounds(tona::backend::Backend& be, const std::vector<int32_t>& ids, int n_prompt, int rounds) {
    for (int round = 0; round <= rounds; round++) {
        if (round > 0) be.reset();
        const auto t0 = Clock::now();
        be.prefill(ids.data(), n_prompt - 1, nullptr);
        const double sec = since(t0);
        std::printf("prefill: %d tokens in %.3f s = %.0f tok/s (round %d%s)\n", n_prompt - 1, sec, (n_prompt - 1) / sec,
                    round, round == 0 ? ", first, expert tiers cold" : "");
    }
}

uint64_t fnv(const void* p, size_t bytes, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < bytes; i++) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
    return h;
}

// Greedy decode of gen tokens in passes of one; FNV-1a of the generated ids: equal runs print equal values. The logits
// of the first pass go to `first` (a hash of the bits and, with a path, the file)
uint64_t decode_greedy(tona::backend::Backend& be, int32_t tok, int gen, size_t vocab, const std::string& dump) {
    const Counters c0 = counters(be);
    const auto t1 = Clock::now();
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < gen; i++) {
        const float* lg = be.forward(&tok, 1);
        if (i == 0) {
            std::printf("first logits hash: %016llx\n", (unsigned long long)fnv(lg, vocab * sizeof(float)));
            if (!dump.empty()) std::ofstream(dump, std::ios::binary).write((const char*)lg, vocab * sizeof(float));
        }
        tok = argmax(lg, vocab);
        be.commit(1);
        h = (h ^ (uint64_t)tok) * 1099511628211ull;
    }
    report_decode(c0, counters(be), gen, since(t1), 1);
    return h;
}

// Hashes of the logits of 16 positions after the prompt, computed in passes of w tokens
std::vector<uint64_t> width_hashes(tona::backend::Backend& be, const std::vector<int32_t>& ids, int n_prompt, int w,
                                   size_t vocab) {
    be.reset();
    be.prefill(ids.data(), n_prompt - 1, nullptr);
    std::vector<uint64_t> hs;
    for (int pos = n_prompt - 1; pos + w <= n_prompt - 1 + 16; pos += w) {
        const float* lg = be.forward(ids.data() + pos, w);
        for (int r = 0; r < w; r++) hs.push_back(fnv(lg + r * vocab, vocab * sizeof(float)));
        be.commit(w);
    }
    return hs;
}

void check_widths(tona::backend::Backend& be, const std::vector<int32_t>& ids, int n_prompt, const Options& o,
                  size_t vocab) {
    const std::vector<uint64_t> ref = width_hashes(be, ids, n_prompt, 1, vocab);
    for (const int w : o.check_widths) {
        const std::vector<uint64_t> hs = width_hashes(be, ids, n_prompt, w, vocab);
        int bad = 0;
        for (size_t i = 0; i < hs.size(); i++) bad += hs[i] != ref[i];
        std::printf("width %d: %d of %zu positions differ from passes of one token\n", w, bad, hs.size());
    }
}

// --repeat: reset, prefill and greedy decode R times, a hash line per repeat with the expert counters of that repeat
void repeat_decode(tona::backend::Backend& be, const std::vector<int32_t>& ids, int n_prompt, const Options& o,
                   size_t vocab) {
    for (int r = 0; r < o.repeat; r++) {
        if (r > 0) {
            be.reset();
            be.prefill(ids.data(), n_prompt - 1, nullptr);
        }
        const Counters r0 = counters(be);
        const uint64_t h = decode_greedy(be, ids[n_prompt - 1], o.gen, vocab, r == 0 ? o.logits_out : std::string());
        const Counters r1 = counters(be);
        std::printf("generated ids hash: %016llx (repeat %d, expert hits %lld misses %lld cpu %lld)\n",
                    (unsigned long long)h, r, (long long)(r1.hits - r0.hits), (long long)(r1.misses - r0.misses),
                    (long long)(r1.cpu - r0.cpu));
    }
}

int run(int argc, char** argv) {
    const Options o = parse(argc, argv);
    const std::unique_ptr<tona::ModelBase> model = tona::load_model(o.model);
    const std::vector<int32_t> ids = read_ids(o.ids);
    const int n_prompt = o.prompt > 0 ? o.prompt : (int)ids.size();
    if (n_prompt < 2 || (size_t)n_prompt > ids.size()) throw std::runtime_error("the ids file has too few tokens");
    if ((size_t)n_prompt + o.gen * o.widths.size() + 8 > ids.size() && !o.widths.empty())
        throw std::runtime_error("--widths needs the ids of the continuation: prompt + gen per width + 8 ids");
    tona::backend::register_builtin();
    tona::backend::CreateOptions co;
    co.max_ctx = o.ctx > 0 ? o.ctx : (int64_t)n_prompt + o.gen + 64;
    const auto be = tona::backend::registry::create("auto", *model, co);
    auto* mega = dynamic_cast<tona::gpu::Mega*>(be.get());
    const size_t vocab = model->moe_shape().vocab;
    prefill_rounds(*be, ids, n_prompt, o.rounds);
    if (mega && tona::env::flag("TONA_MEGA_PROF")) (void)mega->profile(true);  // the prefill is not in the profile
    if (!o.check_widths.empty()) {
        check_widths(*be, ids, n_prompt, o, vocab);
        return 0;
    }
    if (!o.widths.empty()) {
        decode_forced(*be, ids, n_prompt - 1, o);
        return 0;
    }
    repeat_decode(*be, ids, n_prompt, o, vocab);
    if (mega && tona::env::flag("TONA_MEGA_PROF")) print_profile(*mega);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
