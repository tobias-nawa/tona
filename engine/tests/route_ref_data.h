// The reference cases of the routing policies (tests/data/route_ref.txt, written by tools/gen_route_ref.py in float64)
// and the generator of their inputs: logits and bias are not stored, both languages derive them from the seed of the
// case with the same xorshift64* and exact scaling, so every value is the same float.
#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace routeref {

class Stream {
public:
    explicit Stream(uint64_t seed) : x_((seed + 1) * 0x9E3779B97F4A7C15ull) {}
    uint32_t next() {  // the upper 24 bits of the next output
        x_ ^= x_ >> 12;
        x_ ^= x_ << 25;
        x_ ^= x_ >> 27;
        return (uint32_t)((x_ * 0x2545F4914F6CDD1Dull) >> 40);
    }

private:
    uint64_t x_;
};

struct Case {
    std::string name, gate;
    int E = 0, K = 0, with_bias = 0, levels = 0;
    float scale = 1.0f;
    uint64_t seed = 0;
    std::vector<int> experts;     // the chosen experts in rank order
    std::vector<double> weights;  // their weights
    std::vector<float> logits, bias;
};

// Uniform in [-4, 4) in steps of 2^-21; with levels > 0 only that many distinct values (exact ties)
inline std::vector<float> make_logits(uint64_t seed, int n, int levels) {
    Stream g(seed);
    std::vector<float> r(n);
    for (float& v : r) v = levels ? (float)(g.next() % levels) * 0.5f - 1.0f : (float)g.next() * 0x1p-21f - 4.0f;
    return r;
}

// Uniform in [-0.5, 0.5) in steps of 2^-24
inline std::vector<float> make_bias(uint64_t seed, int n) {
    Stream g(seed + 1000);
    std::vector<float> b(n);
    for (float& v : b) v = (float)g.next() * 0x1p-24f - 0.5f;
    return b;
}

inline std::vector<Case> load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<Case> cases;
    std::string line, tag;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        s >> tag;
        if (tag == "case") {
            Case c;
            s >> c.name >> c.E >> c.K >> c.gate >> c.scale >> c.with_bias >> c.levels >> c.seed;
            c.logits = make_logits(c.seed, c.E, c.levels);
            if (c.with_bias) c.bias = make_bias(c.seed, c.E);
            cases.push_back(std::move(c));
        } else if (tag == "experts") {
            for (int e; s >> e;) cases.back().experts.push_back(e);
        } else if (tag == "weights") {
            for (double w; s >> w;) cases.back().weights.push_back(w);
        }
    }
    return cases;
}

}  // namespace routeref
