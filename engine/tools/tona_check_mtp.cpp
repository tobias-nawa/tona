// The MTP check of tona-check. A golden file (text) holds the token sequence (prompt, then the trunk's greedy
// continuation) and per MTP call its pick and top 5 (token, logit); OUT.f32 holds the full logit rows of the first
// calls (vocab floats each, NaN for tokens outside a truncated head). A replay feeds the golden tokens and picks, so
// both backends see the same inputs whatever their own picks.
#include "tools/tona_check_mtp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tona::check {

namespace {

using Top = std::vector<std::pair<int32_t, float>>;

struct Call {
    int step = 0, kind = 0;  // kind 0: fill, 1 ... steps: chained step
    int32_t pick = -1;
    Top top;
};

struct Golden {
    std::string backend;
    size_t prompt = 0;
    std::vector<int32_t> seq;
    std::vector<Call> calls;
};

Top top5(const backend::Backend& be) {
    const float* lg = be.mtp_logits();
    if (!lg) throw std::runtime_error("--mtp: the backend keeps no MTP logits");
    Top t;
    for (size_t r = 0; r < be.mtp_rows(); r++) {
        const int32_t tok = be.mtp_token(r);
        if (tok < 0) continue;
        t.emplace_back(tok, lg[r]);
        std::push_heap(t.begin(), t.end(), [](auto& a, auto& b) { return a.second > b.second; });
        if (t.size() > 5)
            std::pop_heap(t.begin(), t.end(), [](auto& a, auto& b) { return a.second > b.second; }), t.pop_back();
    }
    std::sort(t.begin(), t.end(),
              [](auto& a, auto& b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
    return t;
}

void write_row(const backend::Backend& be, size_t vocab, std::ofstream& f) {
    std::vector<float> row(vocab, NAN);
    for (size_t r = 0; r < be.mtp_rows(); r++)
        if (be.mtp_token(r) >= 0 && (size_t)be.mtp_token(r) < vocab) row[be.mtp_token(r)] = be.mtp_logits()[r];
    f.write(reinterpret_cast<const char*>(row.data()), (std::streamsize)(row.size() * sizeof(float)));
}

void save(const Golden& g, const std::string& path) {
    std::ofstream f(path);
    f << "tona-mtp-golden 1\nbackend " << g.backend << "\nprompt " << g.prompt << "\nseq " << g.seq.size();
    for (int32_t t : g.seq) f << ' ' << t;
    f << '\n';
    for (const Call& c : g.calls) {
        f << "call " << c.step << ' ' << c.kind << ' ' << c.pick << ' ' << c.top.size();
        for (const auto& [t, l] : c.top) f << ' ' << t << ' ' << std::hexfloat << l << std::defaultfloat;
        f << '\n';
    }
}

Golden load(const std::string& path) {
    std::ifstream f(path);
    std::string line, word;
    Golden g;
    if (!std::getline(f, line) || line != "tona-mtp-golden 1") throw std::runtime_error(path + ": not a golden file");
    while (std::getline(f, line)) {
        std::istringstream in(line);
        in >> word;
        size_t n = 0;
        if (word == "backend") in >> g.backend;
        if (word == "prompt") in >> g.prompt;
        if (word == "seq")
            for (in >> n; n-- > 0;) g.seq.push_back(0), in >> g.seq.back();
        if (word != "call") continue;
        Call c;
        in >> c.step >> c.kind >> c.pick >> n;
        for (std::string l; n-- > 0;)
            c.top.emplace_back(0, 0.f), in >> c.top.back().first >> l,
                c.top.back().second = std::strtof(l.c_str(), nullptr);
        g.calls.push_back(c);
    }
    return g;
}

struct Tally {
    int calls = 0, same = 0;
    double worst = 0;
    std::vector<double> rel;
};

// Own logits of the golden top 5 against theirs, relative to the largest golden logit; prints the call
void compare(const backend::Backend& be, const Call& gold, Tally& t) {
    std::unordered_map<int32_t, float> want;
    for (const auto& [tok, l] : gold.top) want[tok] = NAN;
    for (size_t r = 0; r < be.mtp_rows(); r++)
        if (auto it = want.find(be.mtp_token(r)); it != want.end()) it->second = be.mtp_logits()[r];
    double max_gold = 0, max_diff = 0;
    for (const auto& [tok, l] : gold.top) {
        max_gold = std::max(max_gold, (double)std::fabs(l));
        const float own = want[tok];
        max_diff = std::max(max_diff, std::isnan(own) ? INFINITY : std::fabs((double)own - l));
    }
    const Top own = top5(be);
    const int32_t pick = own.empty() ? -1 : own.front().first;
    const double rel = max_gold > 0 ? max_diff / max_gold : max_diff;
    t.calls++, t.same += pick == gold.pick, t.worst = std::max(t.worst, rel), t.rel.push_back(rel);
    std::fprintf(stderr, "  mtp step %d call %d: top-1 %d golden %d, relative deviation of the golden top 5 %.2e\n",
                 gold.step, gold.kind, pick, gold.pick, rel);
}

// State of one check: the golden file being written or replayed, the row file, the comparison so far
struct Run {
    backend::Backend& be;
    const MtpOptions& o;
    size_t vocab;
    Golden g;
    bool replay = false;
    std::ofstream rows;
    Tally t;
    size_t ci = 0;  // next golden call (replay)
    Golden own;     // replay with --dump: this backend's calls on the golden inputs
};

// The fill after trunk position i and its chained steps; the steps take the golden picks when replaying
void chain(Run& r, size_t i) {
    for (int k = 0; k <= r.o.steps; k++) {
        if (r.replay && r.ci >= r.g.calls.size()) return;
        const int32_t feed = k == 0 ? r.g.seq[i + 1] : (r.replay ? r.g.calls[r.ci - 1].pick : r.g.calls.back().pick);
        const backend::MtpPick pk = k == 0 ? r.be.mtp_fill(feed) : r.be.mtp_step(feed);
        if (r.replay) {
            const Call& gold = r.g.calls[r.ci++];
            compare(r.be, gold, r.t);
            if (!r.o.dump.empty()) r.own.calls.push_back(Call{gold.step, gold.kind, pk.token, top5(r.be)});
            if (r.rows.is_open() && (int)r.ci <= r.o.rows) write_row(r.be, r.vocab, r.rows);
            continue;
        }
        r.g.calls.push_back(Call{(int)(i + 1 - r.g.prompt), k, pk.token, top5(r.be)});
        if (r.rows.is_open() && (int)r.g.calls.size() <= r.o.rows) write_row(r.be, r.vocab, r.rows);
    }
}

int report(Run& r) {
    if (r.replay && !r.o.dump.empty())
        r.own.backend = r.be.info().name, r.own.seq = r.g.seq, r.own.prompt = r.g.prompt, save(r.own, r.o.dump);
    if (!r.replay) {
        save(r.g, r.o.dump);
        std::fprintf(stderr, "MTP dump: %zu tokens (%zu prompt), %zu calls into %s\n", r.g.seq.size(), r.g.prompt,
                     r.g.calls.size(), r.o.dump.c_str());
        return 0;
    }
    std::vector<double> rel = r.t.rel;
    std::sort(rel.begin(), rel.end());
    const double median = rel.empty() ? 0 : rel[rel.size() / 2], p90 = rel.empty() ? 0 : rel[rel.size() * 9 / 10];
    std::fprintf(stderr,
                 "MTP against %s: %d calls, top-1 equal in %d, relative deviation of the golden top 5: median %.2e, "
                 "p90 %.2e, worst %.2e\n",
                 r.g.backend.c_str(), r.t.calls, r.t.same, median, p90, r.t.worst);
    return r.t.same * 10 >= r.t.calls * 9 && median <= r.o.tolerance ? 0 : 1;
}

// The prompt as o.prompt_mode says up to its last token (forward, fill) or up to the one before (prefill); returns
// the first position the main loop forwards
size_t feed_prompt(Run& r) {
    if (r.o.prompt_mode != "forward" && r.o.prompt_mode != "fill" && r.o.prompt_mode != "prefill")
        throw std::invalid_argument("--mtp-prompt: forward, fill or prefill");
    if (r.o.prompt_mode != "prefill" || r.g.prompt < 2) return 0;
    r.be.prefill(r.g.seq.data(), (int)r.g.prompt - 1);
    return r.g.prompt - 1;
}

}  // namespace

int mtp_check(backend::Backend& be, const Tokenizer& tok, const MtpOptions& o, size_t vocab) {
    if (!be.has_mtp()) throw std::runtime_error("--mtp: the backend has no MTP head");
    Run r{be, o, vocab, {}, !o.golden.empty(), {}, {}, 0, {}};
    if (r.replay) r.g = load(o.golden);
    if (!r.replay) r.g.backend = be.info().name, r.g.seq = tok.encode(o.prompt), r.g.prompt = r.g.seq.size();
    const size_t total = r.replay ? r.g.seq.size() : r.g.prompt + (size_t)o.n_gen;
    if (!o.dump.empty()) r.rows.open(o.dump + ".f32", std::ios::binary);
    be.mtp_keep_logits(true);
    for (size_t i = feed_prompt(r); i < r.g.seq.size();
         i++) {  // the dump appends the trunk's greedy token behind the prompt
        const float* lg = be.forward(&r.g.seq[i], 1);
        be.commit(1);
        if (o.prompt_mode == "fill" && i + 1 < r.g.prompt) be.mtp_fill(-1);
        if (i + 1 < r.g.prompt) continue;
        if (!r.replay && r.g.seq.size() < total) r.g.seq.push_back((int32_t)(std::max_element(lg, lg + vocab) - lg));
        if (i + 1 < r.g.seq.size()) chain(r, i);
    }
    return report(r);
}

}  // namespace tona::check
