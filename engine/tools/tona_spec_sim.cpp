// Simulates SuffixDecoding on an agent corpus (tools/agent_corpus.py): how many tokens would a verify pass deliver
// on average if the model produced exactly the recorded output?
// Usage: tona-spec-sim MODEL.gguf CORPUS [--test N] [--global-sessions N] [--depth D] [--pattern P]
//                    [--alpha A] [--max-spec S] [--min-prob P] [--tree] [--no-global] [--no-request]
//                    [--union u1,u2,...] [--hit H] [--cost-aware]
//   --union: measured union of experts for batch 1, 2, ... (multiples of one token, tools/expert_stats.py)
//   --hit: hit rate of the VRAM expert cache; misses come over PCIe
//   --cost-aware: cut a path draft to the length that maximizes expected tokens per time
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "model/gguf.h"
#include "model/tokenizer.h"
#include "spec/suffix.h"
#include "tona_args.h"
#include "tona_corpus.h"

namespace {

constexpr int kHistogramCap = 17;

struct Options {
    const char* model = nullptr;
    const char* corpus = nullptr;
    int n_test = 40;
    int n_global = 200;
    int depth = 40;
    bool use_global = true;
    bool use_request = true;
    bool cost_aware = false;
    std::vector<double> uni = {1.0};
    double hit = 1.0;
    tona::spec::SpecParams sp;
};

struct ValueOption {
    const char* name;
    void (*set)(Options&, const char*);
};

constexpr ValueOption kValueOptions[] = {
    {"--test", [](Options& o, const char* v) { o.n_test = tona::tools::parse_int(v); }},
    {"--global-sessions", [](Options& o, const char* v) { o.n_global = tona::tools::parse_int(v); }},
    {"--depth", [](Options& o, const char* v) { o.depth = tona::tools::parse_int(v); }},
    {"--pattern", [](Options& o, const char* v) { o.sp.max_pattern = tona::tools::parse_int(v); }},
    {"--alpha", [](Options& o, const char* v) { o.sp.alpha = (float)tona::tools::parse_double(v); }},
    {"--max-spec", [](Options& o, const char* v) { o.sp.max_spec = tona::tools::parse_int(v); }},
    {"--min-prob", [](Options& o, const char* v) { o.sp.min_prob = (float)tona::tools::parse_double(v); }},
    {"--hit", [](Options& o, const char* v) { o.hit = tona::tools::parse_double(v); }},
    {"--union",
     [](Options& o, const char* v) {
         o.uni.clear();
         std::stringstream ss(v);
         for (std::string x; std::getline(ss, x, ',');) o.uni.push_back(tona::tools::parse_double(x.c_str()));
     }},
};

bool parse_flag(Options& o, const char* arg) {
    if (!std::strcmp(arg, "--tree"))
        o.sp.tree = true;
    else if (!std::strcmp(arg, "--no-global"))
        o.use_global = false;
    else if (!std::strcmp(arg, "--no-request"))
        o.use_request = false;
    else if (!std::strcmp(arg, "--cost-aware"))
        o.cost_aware = true;
    else
        return false;
    return true;
}

Options parse_options(int argc, char** argv) {
    Options o;
    o.model = argv[1];
    o.corpus = argv[2];
    for (int i = 3; i < argc; i++) {
        if (parse_flag(o, argv[i])) continue;
        for (const ValueOption& opt : kValueOptions) {
            if (std::strcmp(argv[i], opt.name) != 0) continue;
            if (i + 1 >= argc) throw std::runtime_error("value missing");
            opt.set(o, argv[++i]);
        }
    }
    return o;
}

// Time of a verify pass at batch b (bandwidth model): dense weights once from VRAM, experts u(b) token sets; hits
// from VRAM (315 GB/s), misses over PCIe (55 GB/s)
class CostModel {
public:
    CostModel(const std::vector<double>& uni, double hit)
        : uni_(uni), expert_seconds_(0.98 * (hit / 315.0 + (1.0 - hit) / 55.0)) {}

    double pass_time(int b) const { return dense_seconds_ + expert_seconds_ * union_at(b); }

private:
    double union_at(int b) const {
        if (b <= (int)uni_.size()) return uni_[b - 1];
        const double slope = uni_.size() > 1 ? uni_.back() - uni_[uni_.size() - 2] : 1.0;
        return uni_.back() + slope * (b - (int)uni_.size());
    }

    const std::vector<double>& uni_;
    const double dense_seconds_ = 4.11 / 315.0;
    const double expert_seconds_;
};

struct Stats {
    double time = 0;
    int64_t passes = 0, tokens = 0, batch_sum = 0, drafted = 0, with_draft = 0, ctx_tokens = 0;
    std::map<int, int64_t> hist;  // tokens per pass
};

struct Tokenized {
    bool is_output;
    std::vector<int32_t> tokens;
};

// Request-local trie and the matchers over it and over the global trie
struct Request {
    Request(int depth, const tona::spec::SuffixTrie& global, int max_pattern)
        : trie(depth), request_matcher(trie, max_pattern), global_matcher(global, max_pattern) {}

    void feed(int32_t t) {
        trie.append(t);
        request_matcher.push(t);
        global_matcher.push(t);
    }

    tona::spec::SuffixTrie trie;
    tona::spec::Matcher request_matcher;
    tona::spec::Matcher global_matcher;
};

class Simulator {
public:
    Simulator(const Options& o, const tona::Tokenizer& tok, const std::vector<tona::tools::CorpusSession>& raw)
        : o_(o), tok_(tok), raw_(raw), cost_(o.uni, o.hit), global_(o.depth) {
        first_test_ = raw.size() > (size_t)o.n_test ? raw.size() - o.n_test : 0;
        const size_t first_global = first_test_ > (size_t)o.n_global ? first_test_ - o.n_global : 0;
        for (size_t i = first_global; i < first_test_; i++) add_global(tokenize(i));
        std::fprintf(stderr, "global trie: %zu sessions, %zu nodes\n", first_test_ - first_global, global_.size());
    }

    Stats run() {
        for (size_t i = first_test_; i < raw_.size(); i++) {
            const auto session = tokenize(i);
            Request req(o_.depth, global_, o_.sp.max_pattern);
            for (const auto& seg : session) {
                if (seg.is_output) {
                    play_output(req, seg.tokens);
                } else {
                    for (int32_t t : seg.tokens) req.feed(t);
                    stats_.ctx_tokens += (int64_t)seg.tokens.size();
                }
            }
            add_global(session);
        }
        return stats_;
    }

    size_t test_sessions() const { return raw_.size() - first_test_; }
    const CostModel& cost() const { return cost_; }

private:
    std::vector<Tokenized> tokenize(size_t i) const {
        std::vector<Tokenized> s;
        for (const auto& seg : raw_[i].segments) s.push_back({seg.is_output, tok_.encode(seg.text)});
        return s;
    }

    void add_global(const std::vector<Tokenized>& session) {
        for (const auto& seg : session)
            if (seg.is_output) {
                global_.cut();
                global_.append(seg.tokens);
            }
    }

    tona::spec::Draft best_draft(const Request& req) const {
        tona::spec::Draft d;
        if (o_.use_request) d = tona::spec::propose(req.trie, req.request_matcher.matches(), o_.sp);
        if (o_.use_global) {
            tona::spec::Draft dg = tona::spec::propose(global_, req.global_matcher.matches(), o_.sp);
            if (dg.score > d.score) d = std::move(dg);
        }
        return d;
    }

    // Length L with the largest (1 + sum_{i<L} D_i) / pass_time(1 + L); L = 0 means no draft
    void trim_by_cost(tona::spec::Draft& d) const {
        double best = 1.0 / cost_.pass_time(1), acc = 1.0;
        size_t best_len = 0;
        for (size_t len = 1; len <= d.size(); len++) {
            acc += d.prob[len - 1];
            const double rate = acc / cost_.pass_time(1 + (int)len);
            if (rate > best) best = rate, best_len = len;
        }
        d.token.resize(best_len);
        d.parent.resize(best_len);
        d.prob.resize(best_len);
    }

    void play_output(Request& req, const std::vector<int32_t>& out) {
        const int n = (int)out.size();
        for (int at = 0; at < n;) {
            tona::spec::Draft d = best_draft(req);
            if (o_.cost_aware && !o_.sp.tree && d.size() > 0) trim_by_cost(d);
            const int accepted = tona::spec::accepted(d, out.data() + at, n - at);
            const int take = std::min(accepted + 1, n - at);  // +1: the model's own token
            for (int k = 0; k < take; k++) req.feed(out[at + k]);
            at += take;
            record_pass(take, d);
        }
    }

    void record_pass(int take, const tona::spec::Draft& d) {
        stats_.passes++;
        stats_.tokens += take;
        stats_.batch_sum += 1 + (int64_t)d.size();
        stats_.time += cost_.pass_time(1 + (int)d.size());
        stats_.drafted += (int64_t)d.size();
        stats_.with_draft += d.size() > 0;
        stats_.hist[std::min(take, kHistogramCap)]++;
    }

    const Options& o_;
    const tona::Tokenizer& tok_;
    const std::vector<tona::tools::CorpusSession>& raw_;
    CostModel cost_;
    tona::spec::SuffixTrie global_;
    size_t first_test_ = 0;
    Stats stats_;
};

void print_report(const Options& o, const Simulator& sim, const Stats& st, double secs) {
    const CostModel& cost = sim.cost();
    const double tpp = (double)st.tokens / st.passes, batch = (double)st.batch_sum / st.passes;
    // Decode cost model, bandwidth only: dense 4.11 GB once per pass, experts 0.98 GB per token in the batch for
    // disjoint experts (worst case) or once (best case)
    const double worst = tpp * (4.11 + 0.98) / (4.11 + 0.98 * batch);
    std::printf("Configuration: depth %d, pattern <= %d, alpha %.1f, max_spec %d, %s, sources %s%s\n", o.depth,
                o.sp.max_pattern, o.sp.alpha, o.sp.max_spec, o.sp.tree ? "tree" : "path",
                o.use_request ? "request " : "", o.use_global ? "global" : "");
    std::printf("Test: %d sessions, context %lld tokens, output %lld tokens, %.1f s\n", (int)sim.test_sessions(),
                (long long)st.ctx_tokens, (long long)st.tokens, secs);
    std::printf("Tokens per pass %.3f, passes with draft %.1f %%, acceptance %.1f %% of the draft tokens\n", tpp,
                100.0 * st.with_draft / st.passes, st.drafted ? 100.0 * (st.tokens - st.passes) / st.drafted : 0.0);
    std::printf("Verify batch on average %.2f tokens\n", batch);
    std::printf("Decode speedup by the bandwidth model: %.2fx (disjoint experts) to %.2fx (equal)\n", worst, tpp);
    if (o.uni.size() > 1 || o.hit < 1.0)
        std::printf(
            "With measured union and cache hit rate %.1f %%: %.2fx (%.1f instead of %.1f tok/s decode by the "
            "model)\n",
            o.hit * 100, st.tokens * cost.pass_time(1) / st.time, st.tokens / st.time, 1.0 / cost.pass_time(1));
    std::printf("Distribution of tokens per pass:");
    for (const auto& [k, v] : st.hist)
        std::printf(" %d%s:%.1f%%", k, k == kHistogramCap ? "+" : "", 100.0 * v / st.passes);
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: tona-spec-sim MODEL.gguf CORPUS [options]\n");
        return 2;
    }
    try {
        const Options o = parse_options(argc, argv);
        tona::Gguf g(o.model);
        tona::Tokenizer tok(g);
        const auto raw = tona::tools::read_corpus(o.corpus);
        Simulator sim(o, tok, raw);
        const auto t0 = std::chrono::steady_clock::now();
        const Stats st = sim.run();
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        print_report(o, sim, st, secs);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
