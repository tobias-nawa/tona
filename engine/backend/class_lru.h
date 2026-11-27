// ClassLRU replacement order of one layer's expert cache (TONA_EXPERT_POLICY=classlru), after llama.cpp PR 31233 and
// tools/cache_sim/classlru.py: every expert has a combined recency/frequency value (CRF, uses decayed by half every
// `half_life` tokens, kept across evictions); the class is floor(log2(CRF)) at its last use; each class is an LRU
// list; the victim is the least recently used expert of the class where age / 2^class is largest, so a rarely used
// expert leaves before a frequently used one of the same age. No admission: every expert that is put in stays in.
// Host code only; the clock is the number of tokens the layer has seen.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <list>
#include <unordered_set>
#include <vector>

namespace tona::backend {

class ClassLru {
public:
    using List = std::list<int>;

    // 4 classes, half-life 128 tokens: the setting both route traces pick
    // (research/measurements/2026-11-02-classlru.md)
    explicit ClassLru(int n_expert, int n_class = 4, double half_life = 128.0)
        : half_life_(half_life),
          lists_(n_class),
          value_(n_expert, 0.0),
          last_(n_expert, 0),
          class_(n_expert, 0),
          fresh_(n_expert, 0) {}

    // The expert is used in this pass (once per layer and pass, however many tokens chose it)
    void use(int e) {
        value_[e] = decayed(e) + 1.0;
        last_[e] = clock_;
    }
    void tick(int tokens) { clock_ += tokens; }

    // A new expert after a miss: most recent end of the class of its CRF
    List::iterator insert(int e) {
        fresh_[e] = 0;
        return place(e, class_for(e), lists_[class_for(e)].begin());
    }
    // A preloaded or profile-seeded expert that nothing used yet: the end where victims leave, infinite age
    List::iterator insert_fresh(int e) {
        fresh_[e] = 1;
        return place(e, 0, lists_[0].end());
    }
    // A hit: to the most recent end of the class of its (just credited) CRF
    List::iterator promote(int e, List::iterator it) {
        fresh_[e] = 0;
        const int c = class_for(e);
        lists_[c].splice(lists_[c].begin(), lists_[class_[e]], it);
        class_[e] = c;
        return it;
    }
    void erase(int e, List::iterator it) {
        lists_[class_[e]].erase(it);
        fresh_[e] = 0;
    }
    void clear() {
        for (List& l : lists_) l.clear();
        std::fill(fresh_.begin(), fresh_.end(), 0);
    }

    // The expert to evict, not in `keep`; -1 if every cached expert is in it
    int victim(const std::unordered_set<int>* keep) const {
        int best = -1;
        double best_score = -1.0;
        for (size_t c = 0; c < lists_.size(); c++) {
            for (auto it = lists_[c].rbegin(); it != lists_[c].rend(); ++it) {
                if (keep && keep->contains(*it)) continue;
                const double s = age(*it) / std::ldexp(1.0, (int)c);
                if (s > best_score) best = *it, best_score = s;
                break;  // only the least recent expert of a class counts
            }
        }
        return best;
    }

    int class_of(int e) const { return class_[e]; }
    double crf(int e) const { return decayed(e); }

private:
    double decayed(int e) const { return value_[e] * std::exp2(-(double)(clock_ - last_[e]) / half_life_); }
    double age(int e) const {
        return fresh_[e] ? std::numeric_limits<double>::infinity() : (double)(clock_ - last_[e]);
    }
    int class_for(int e) const {
        return std::clamp((int)std::log2(std::max(value_[e], 1.0)), 0, (int)lists_.size() - 1);
    }
    List::iterator place(int e, int c, List::iterator at) {
        class_[e] = c;
        return lists_[c].insert(at, e);
    }

    double half_life_;
    int64_t clock_ = 0;
    std::vector<List> lists_;  // front = most recently used
    std::vector<double> value_;
    std::vector<int64_t> last_;
    std::vector<int> class_;
    std::vector<char> fresh_;
};

}  // namespace tona::backend
