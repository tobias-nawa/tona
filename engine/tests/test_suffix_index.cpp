// server/SuffixIndex (drafts from the history in the server) against a plain search over the whole history
#include <cstdio>
#include <random>
#include <vector>

#include "server/suffix.h"
#include "tests/test_support.h"

using tona::server::SuffixIndex;

// Like draft: longest match of the end of h + last with an earlier position i (the key end h[i] stands for last),
// only the most recent MAX_POS positions of the key, the younger one on a tie. These are the positions i whose
// key (the KEY tokens ending at h[i]) equals the key at the end of h + last
static std::vector<size_t> key_positions(const std::vector<int32_t>& h, int32_t last) {
    constexpr int KEY = SuffixIndex::KEY;
    const size_t n = h.size();
    auto at = [&](size_t j) { return j == n ? last : h[j]; };
    std::vector<size_t> pos;
    for (size_t i = KEY - 1; i < n; i++) {
        bool eq = true;
        for (int j = 0; j < KEY; j++) eq = eq && h[i - j] == at(n - j);
        if (eq) pos.push_back(i);
    }
    if (pos.size() > SuffixIndex::MAX_POS) pos.erase(pos.begin(), pos.end() - SuffixIndex::MAX_POS);
    return pos;
}

static int reference(const std::vector<int32_t>& h, int32_t last, int k, std::vector<int32_t>& out) {
    constexpr int KEY = SuffixIndex::KEY;
    out.clear();
    const size_t n = h.size();
    if (n < (size_t)KEY) return 0;
    auto at = [&](size_t j) { return j == n ? last : h[j]; };
    const std::vector<size_t> pos = key_positions(h, last);
    int best = 0;
    size_t best_i = 0;
    for (size_t c = pos.size(); c-- > 0;) {
        const size_t i = pos[c];
        if (i + 1 >= n) continue;
        int len = 0;
        while (len < SuffixIndex::MAX_MATCH && len <= (int)i && at(n - len) == h[i - len]) len++;
        if (len > best) best = len, best_i = i;
    }
    if (best < KEY) return 0;
    for (size_t j = best_i + 1; j < n && (int)out.size() < k; j++) out.push_back(h[j]);
    return best;
}

// Random histories against the reference: 0 if every draft matches
int check_random_histories() {
    int fail = 0, checked = 0, hits = 0;
    std::mt19937 rng(7);
    for (int vocab : {3, 8, 50}) {
        SuffixIndex sx;
        std::vector<int32_t> h;
        for (int step = 0; step < 3000; step++) {
            // Mostly append (like the server), sometimes another conversation (rebuild) or repeat a stretch
            const int r = (int)(rng() % 100);
            if (r < 2)
                h.resize(h.size() / 2), h.push_back(-1 - (int)(rng() % 5));
            else if (r < 10 && h.size() > 40) {
                const size_t a = rng() % (h.size() - 20);
                for (size_t j = 0; j < 12; j++) h.push_back(h[a + j]);
            } else {
                h.push_back((int32_t)(rng() % vocab));
            }
            sx.sync(h);
            const int32_t last = (int32_t)(rng() % vocab);
            std::vector<int32_t> a, b;
            const int la = sx.draft(last, 8, a), lb = reference(h, last, 8, b);
            checked++;
            hits += lb > 0;
            if (la != lb || a != b) {
                if (fail < 5)
                    fprintf(stderr, "vocab %d, step %d: length %d vs %d, %zu vs %zu tokens\n", vocab, step, la, lb,
                            a.size(), b.size());
                fail++;
            }
        }
    }
    printf("SuffixIndex: %d cases (%d with a draft), %d mismatches\n", checked, hits, fail);
    return fail ? 1 : 0;
}

int main() { return tona::test::guard("suffix_index", check_random_histories); }
