// Simulates the server's suffix rule (server/suffix, Generator::generate) on the agent corpus
// (tools/agent_corpus.py): a draft from the history once TONA_SUFFIX_MIN tokens match, else the drafter. How many
// tokens do passes from the history bring with at most K draft tokens (today 3, passes up to 4 tokens)?
// Assumption as in tona-spec-sim: the model produces exactly the recorded output.
// Usage: tona-sfx-policy MODEL.gguf CORPUS [--test N] [--min M] [--drafter R]
//   --drafter R: tokens per drafter pass (measured ~2.6), for the overall figure
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model/gguf.h"
#include "model/tokenizer.h"
#include "server/suffix.h"
#include "tona_args.h"
#include "tona_corpus.h"

namespace {

constexpr int kMaxCap = 17;

// Union of experts u(b) at batch b (research/measurements/2026-10-22-expert-routing.md), bandwidth model as in
// tona-spec-sim: dense weights once from VRAM, experts u(b) token sets.
constexpr double kUnion[16] = {1,    1.57, 2.06, 2.50, 2.89, 3.28, 3.62, 3.95,
                               4.23, 4.51, 4.78, 5.05, 5.27, 5.49, 5.72, 5.94};
constexpr double kDenseSeconds = 4.11 / 315.0;
constexpr double kExpertSeconds = 0.98 * (0.865 / 315.0 + 0.135 / 55.0);

double pass_time(int batch) { return kDenseSeconds + kExpertSeconds * kUnion[batch - 1]; }

struct Options {
    const char* model;
    const char* corpus;
    int n_test = 40;
    int min_match = 8;
    double drafter_rate = 2.6;
};

struct Tally {
    int64_t out_tokens = 0, history_passes = 0, history_tokens = 0, drafter_tokens = 0;
    double history_seconds = 0;
    int64_t hist[kMaxCap] = {};
};

// One recorded output: the first token comes from the previous pass, then one draft per pass.
void simulate_output(Tally& t, const std::vector<int32_t>& out, std::vector<int32_t>& seq,
                     tona::server::SuffixIndex& idx, const Options& o, int cap) {
    t.out_tokens += (int64_t)out.size();
    size_t j = 0;
    std::vector<int32_t> draft;
    while (j < out.size()) {
        const int32_t last = out[j];
        idx.sync(seq);
        const int matched = idx.draft(last, cap, draft);
        seq.push_back(last);
        if (matched < o.min_match || draft.empty()) {
            t.drafter_tokens++;
            j++;
            continue;
        }
        int accepted = 0;
        while (accepted < (int)draft.size() && j + 1 + accepted < out.size() &&
               draft[accepted] == out[j + 1 + accepted])
            accepted++;
        t.history_passes++, t.history_tokens += accepted + 1, t.hist[accepted + 1]++;
        t.history_seconds += pass_time(1 + (int)draft.size());
        for (int k = 0; k < accepted; k++) seq.push_back(out[j + 1 + k]);
        j += accepted + 1;
    }
}

Tally simulate(const tona::Tokenizer& tok, const std::vector<tona::tools::CorpusSession>& raw, size_t first,
               const Options& o, int cap) {
    Tally t;
    for (size_t s = first; s < raw.size(); s++) {
        std::vector<int32_t> seq;
        tona::server::SuffixIndex idx;
        for (const auto& seg : raw[s].segments) {
            const auto tokens = tok.encode(seg.text);
            if (seg.is_output)
                simulate_output(t, tokens, seq, idx, o, cap);
            else
                seq.insert(seq.end(), tokens.begin(), tokens.end());
        }
    }
    return t;
}

void print_tally(const Tally& t, const Options& o, int cap) {
    const double passes = t.history_passes + t.drafter_tokens / o.drafter_rate;
    const double seconds = t.history_seconds + t.drafter_tokens / o.drafter_rate * pass_time(4);
    std::printf(
        "K = %2d: %lld output tokens, %.1f %% from %lld history passes (%.2f per pass), tokens per pass overall "
        "%.2f, time model %.1f tok/s equivalent\n",
        cap, (long long)t.out_tokens, 100.0 * t.history_tokens / t.out_tokens, (long long)t.history_passes,
        (double)t.history_tokens / t.history_passes, t.out_tokens / passes, t.out_tokens / seconds);
    std::printf("        distribution of tokens per history pass:");
    for (int b = 1; b <= cap + 1; b++) std::printf(" %d:%.0f%%", b, 100.0 * t.hist[b] / t.history_passes);
    std::printf("\n");
}

Options parse(int argc, char** argv) {
    Options o{argv[1], argv[2]};
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--test"))
            o.n_test = tona::tools::parse_int(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--min"))
            o.min_match = tona::tools::parse_int(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--drafter"))
            o.drafter_rate = tona::tools::parse_double(argv[i + 1]);
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    try {
        const Options o = parse(argc, argv);
        tona::Gguf g(o.model);
        tona::Tokenizer tok(g);
        const auto raw = tona::tools::read_corpus(o.corpus);
        const size_t first = raw.size() > (size_t)o.n_test ? raw.size() - o.n_test : 0;
        for (int cap : {3, 7, 15}) print_tally(simulate(tok, raw, first, o, cap), o, cap);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
