// The MTP host code every backend shares (backend/mtp_vocab.h, backend/mtp_track.h) against literal copies of the code
// the reference and the Metal backend had before the move. Token streams are replayed through both and the rows, the
// order of the row changes, the chunks of the catch-up and the integers after save, load and reset must be equal.
#include <array>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "backend/mtp_track.h"
#include "backend/mtp_vocab.h"
#include "tests/test_support.h"

namespace {

using namespace tona;
using backend::HSource;
using backend::MtpTrack;
using backend::MtpVocabMap;

// the vocabulary list: Metal's fixed_list as it was, with the 41,500-id table

std::vector<int32_t> old_fixed_list(uint32_t vocab, const char* vf, size_t nmax) {
    static constexpr int32_t builtin[] = {
#include "model/mtp_vocab.inc"
    };
    std::vector<uint8_t> seen(vocab);
    std::vector<int32_t> list;
    auto put = [&](long long id) {
        if (list.size() < nmax && id >= 0 && id < (long long)vocab && !seen[id])
            seen[id] = 1, list.push_back((int32_t)id);
    };
    if (vf) {
        std::ifstream in(vf);
        for (long long id; in >> id;) put(id);
    } else {
        for (int32_t id : builtin) put(id);
    }
    return list;
}

void test_list() {
    const uint32_t vocab = 206819;
    EXPECT(backend::mtp_vocab_list(vocab, backend::MtpList::Wide, nullptr, "t") ==
           old_fixed_list(vocab, nullptr, vocab));
    const auto wide = backend::mtp_vocab_list(vocab, backend::MtpList::Wide, nullptr, "t");
    const auto narrow = backend::mtp_vocab_list(vocab, backend::MtpList::Narrow, nullptr, "t");
    EXPECT(wide.size() > narrow.size() && narrow.size() > 1000);
    // ids outside a small vocabulary are dropped, the order is kept
    EXPECT(backend::mtp_vocab_list(1000, backend::MtpList::Wide, nullptr, "t") == old_fixed_list(1000, nullptr, 1000));

    test::TempDir dir("mtp-list");
    const std::string file = (dir.path() / "list.txt").string();
    std::ofstream(file) << "5 3 5 -1 9999 3 7 0\n12\n";  // duplicates, a negative id, an id outside
    const std::vector<int32_t> want = {5, 3, 7, 0, 12};
    EXPECT(backend::mtp_vocab_list(100, backend::MtpList::Wide, file.c_str(), "t") == want);
    EXPECT(old_fixed_list(100, file.c_str(), 100) == want);
    EXPECT(test::throws(
        [&] { backend::mtp_vocab_list(100, backend::MtpList::Wide, (dir.path() / "none").string().c_str(), "t"); }));
}

// the ring: Mega::mtp_sync_vocab as it was

struct OldRing {
    std::vector<int32_t> vocab;
    std::vector<uint8_t> in_head;
    size_t nstat = 0, nctx = 0, ring = 0;
    std::vector<std::array<int32_t, 3>> events;  // {kind, a, b}: 0 = note(tok, v) 1 = row load (ring index, token)
    struct Out {
        size_t first = 0, n = 0;
    };

    Out sync(const std::vector<int32_t>& tokens, int32_t next, size_t& scan) {
        const size_t dyn = nctx;
        if (dyn == 0) return {};
        if (scan > tokens.size()) scan = 0;
        Out o{ring, 0};
        auto take = [&](int32_t t) {
            if (t < 0 || in_head[t] || o.n == dyn) return;
            const size_t slot = nstat + ring;
            if (vocab[slot] >= 0) in_head[vocab[slot]] = 0, events.push_back({0, vocab[slot], 0});
            vocab[slot] = t, in_head[t] = 1, events.push_back({0, t, 1});
            events.push_back({1, (int32_t)ring, t});
            ring = (ring + 1) % dyn, o.n++;
        };
        for (; scan < tokens.size(); scan++) take(tokens[scan]);
        take(next);
        return o;
    }
};

struct RingCase {
    uint32_t vocab;
    size_t nstat, dyn;
};

void replay_ring(const RingCase& c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> list;
    list.reserve(c.nstat);
    for (size_t i = 0; i < c.nstat; i++) list.push_back((int32_t)(i * 3 % c.vocab));  // every third id has a row
    OldRing old;
    old.vocab = list, old.in_head.assign(c.vocab, 0), old.nstat = list.size(), old.nctx = c.dyn;
    for (int32_t t : list) old.in_head[t] = 1;
    old.vocab.resize(old.nstat + c.dyn, -1);
    MtpVocabMap now(list, c.dyn, c.vocab);

    std::vector<int32_t> tokens;
    size_t scan_old = 0, scan_new = 0;
    std::vector<std::array<int32_t, 3>> events;
    for (int step = 0; step < 400; step++) {
        const int action = (int)(rng() % 10);
        if (action == 0) tokens.clear(), scan_old = scan_new = 0;  // reset
        if (action == 1 && !tokens.empty())
            tokens.resize(rng() % tokens.size()), scan_old = scan_new = 0;  // load_state
        const int grow =
            action >= 8 ? (int)(rng() % 3 * (c.dyn + 3)) : (int)(rng() % 4);  // a burst longer than the ring
        for (int i = 0; i < grow; i++) tokens.push_back((int32_t)(rng() % c.vocab));
        const int32_t next = rng() % 4 == 0 ? -1 : (int32_t)(rng() % c.vocab);

        old.events.clear(), events.clear();
        const OldRing::Out o = old.sync(tokens, next, scan_old);
        const auto n = now.sync(tokens, next, scan_new, [&](size_t ring, int32_t prev, int32_t t) {
            if (prev >= 0) events.push_back({0, prev, 0});
            events.push_back({0, t, 1});
            events.push_back({1, (int32_t)ring, t});
        });
        EXPECT(o.first == n.first && o.n == n.n);
        EXPECT(old.events == events);
        EXPECT(scan_old == scan_new);
        EXPECT(old.vocab == now.tokens());
        for (uint32_t t = 0; t < c.vocab; t += 7) EXPECT((bool)old.in_head[t] == now.in_head(t));
    }
}

void test_ring() {
    replay_ring({200, 20, 7}, 1);  // small ring: wraps all the time
    replay_ring({200, 20, 1}, 2);  // one row
    replay_ring({500, 100, 64}, 3);
    replay_ring({50, 10, 40}, 4);  // ring as large as the tokens not in the list
    replay_ring({200, 20, 0}, 5);  // no context rows: nothing happens, scan stays
    MtpVocabMap none;
    EXPECT(none.empty() && none.token(3) == -1);
    MtpVocabMap m({1, 2}, 2, 10);
    size_t scan = 5;
    const std::vector<int32_t> toks = {9, 9, 8};
    EXPECT(m.sync(toks, -1, scan, [](size_t, int32_t, int32_t) {}).n == 2 &&
           scan == 3);  // scan beyond the history: reset
    EXPECT(!m.in_head(100) && m.in_head(9) && m.token(2) == 9 && m.token(3) == 8);
    m.append_free(3);
    EXPECT(m.size() == 7 && m[6] == -1 && m.n_static() == 2 && m.n_ctx() == 2);
}

// the bookkeeping: Mega::mtp_h_of, mtp_fill's loop and the invariants as they were

enum class OldSrc { None, Pass, Saved, Bad };

struct OldH {
    OldSrc src;
    int64_t row;
};

OldH old_h_of(int64_t q, bool trunk_ok, int64_t fw_pos, int64_t fw_m, int64_t hlast_pos) {
    if (q < 0) return {OldSrc::None, 0};
    if (trunk_ok && q >= fw_pos && q < fw_pos + fw_m) return {OldSrc::Pass, q - fw_pos};
    if (q == hlast_pos) return {OldSrc::Saved, 0};
    return {OldSrc::Bad, 0};
}

struct Chunk {
    int64_t p;
    int nb;
    bool want;
    std::vector<int32_t> toks;
    std::vector<std::array<int64_t, 2>> h;  // {kind, row}
    bool operator==(const Chunk&) const = default;
};

// the loop of Mega::mtp_fill: valid and pos in, the chunks it would run out
struct PlanIn {
    std::vector<int32_t> tokens;
    int64_t valid, pos;
    int32_t next;
    int bmax;
    bool ok;
    int64_t fw_pos, fw_m, hlast_pos;
};

std::vector<Chunk> old_plan(const PlanIn& in, int64_t& stale) {
    std::vector<Chunk> out;
    const int64_t pos = in.pos, end = pos + (in.next >= 0);
    for (int64_t p = std::min(in.valid, pos); p < end;) {
        const int nb = (int)std::min<int64_t>(in.bmax, end - p);
        Chunk c{p, nb, p + nb == end && in.next >= 0, {}, {}};
        c.toks.reserve(nb), c.h.reserve(nb);
        for (int b = 0; b < nb; b++) {
            c.toks.push_back(p + b < pos ? in.tokens[p + b] : in.next);
            const OldH h = old_h_of(p + b - 1, in.ok, in.fw_pos, in.fw_m, in.hlast_pos);
            if (h.src == OldSrc::Bad) stale++;
            c.h.push_back({h.src == OldSrc::Bad ? (int64_t)OldSrc::None : (int64_t)h.src, h.row});
        }
        out.push_back(c);
        p += nb;
    }
    return out;
}

void replay_track(uint32_t seed) {
    std::mt19937 rng(seed);
    for (int it = 0; it < 3000; it++) {
        const int64_t pos = (int64_t)(rng() % 40), valid = (int64_t)(rng() % 45);
        const int64_t fw_pos = (int64_t)(rng() % 40), fw_m = (int64_t)(rng() % 9), hlast = (int64_t)(rng() % 42) - 1;
        const bool ok = rng() % 3 != 0;
        const int bmax = 1 + (int)(rng() % 8);
        const int32_t next = rng() % 3 == 0 ? -1 : (int32_t)(rng() % 1000);
        std::vector<int32_t> tokens(pos);
        for (int32_t& t : tokens) t = (int32_t)(rng() % 1000);

        MtpTrack t;
        t.valid = valid, t.hlast_pos = hlast;
        for (int64_t q = -2; q < 45; q++) {
            const OldH o = old_h_of(q, ok, fw_pos, fw_m, hlast);
            const backend::HLoc n = t.h_of(q, {ok, fw_pos, fw_m});
            EXPECT((int)o.src == (int)n.src && (o.src != OldSrc::Pass || o.row == n.row));
        }

        int64_t stale_old = 0, stale_new = 0;
        const std::vector<Chunk> want = old_plan({tokens, valid, pos, next, bmax, ok, fw_pos, fw_m, hlast}, stale_old);
        std::vector<Chunk> got;
        int32_t toks[8];
        const float* hs[8];
        static float fake[8];  // distinct addresses stand for "pass row" and "saved row"
        std::vector<std::array<int64_t, 2>> codes;
        t.catch_up(
            {tokens, pos, next}, {bmax, toks, hs},
            [&](int64_t q) -> const float* {
                const backend::HLoc h = t.h_of(q, {ok, fw_pos, fw_m});
                codes.push_back({h.src == HSource::Unknown ? (int64_t)HSource::None : (int64_t)h.src, h.row});
                if (h.src == HSource::Unknown) stale_new++;
                return h.src == HSource::Pass || h.src == HSource::Saved ? &fake[0] : nullptr;
            },
            [&](int64_t p, int nb, bool last) {
                Chunk c{p, nb, last, {toks, toks + nb}, codes};
                got.push_back(c);
                codes.clear();
            });
        // the old h kinds are {None=0, Pass=1, Saved=2} in OldSrc and {None=0, Pass=1, Saved=2} in HSource
        EXPECT(want == got);
        EXPECT(stale_old == stale_new);
    }
}

void test_snapshots() {
    std::mt19937 rng(9);
    for (int it = 0; it < 500; it++) {
        const int64_t valid = (int64_t)(rng() % 50), hlast = (int64_t)(rng() % 50) - 1, pos = (int64_t)(rng() % 50) + 1;
        MtpTrack t;
        t.valid = valid, t.hlast_pos = hlast, t.tent = (int32_t)(rng() % 5) - 1, t.scan = 17;
        // save_state, then change the live values, then load_state (mega_state.cpp as it was)
        t.on_save();
        EXPECT(t.snap_valid == valid && t.snap_hlast_pos == hlast);
        t.valid = valid + 3, t.hlast_pos = 77, t.tent = 4;
        t.on_load(pos);
        EXPECT(t.valid == std::min(valid, pos) && t.hlast_pos == hlast && t.tent == -1 && t.scan == 0);
        t.on_unstash(pos);
        EXPECT(t.snap_valid == pos && t.snap_hlast_pos == pos - 1);
        t.valid = 5, t.hlast_pos = 3, t.tent = 2, t.scan = 8;
        t.on_reset();
        EXPECT(t.valid == 0 && t.hlast_pos == -1 && t.tent == -1 && t.scan == 0);
    }
}

void test_pass_and_fill() {
    const int64_t pos = 12;
    MtpTrack t;
    // a pass of the target: the entry computed ahead counts if it is the pass's first token at the right position
    t.valid = pos, t.tent = 9;
    t.on_target_pass(pos, 9);
    EXPECT(t.valid == pos + 1 && t.tent == -1);
    t.valid = pos, t.tent = 9;
    t.on_target_pass(pos, 8);
    EXPECT(t.valid == pos && t.tent == -1);
    t.valid = pos + 1, t.tent = 9;
    t.on_target_pass(pos, 9);
    EXPECT(t.valid == pos + 1);
    t.after_fill(pos, 6, true);
    EXPECT(t.valid == pos && t.hlast_pos == pos - 1 && t.tent == 6);
    t.hlast_pos = 1;
    t.after_fill(pos + 2, -1, false);  // h unknown: the saved row is not claimed
    EXPECT(t.hlast_pos == 1 && t.valid == pos + 2);
    t.after_prefill(pos);
    EXPECT(t.valid == pos && t.hlast_pos == pos - 1 && t.tent == -1);
    t.pos = 4, t.nb = 3;
    EXPECT(t.next_pos() == 7);
}

}  // namespace

int main() {
    return tona::test::run("test_mtp_shared", [] {
        test_list();
        test_ring();
        for (uint32_t s = 1; s <= 4; s++) replay_track(s);
        test_snapshots();
        test_pass_and_fill();
    });
}
