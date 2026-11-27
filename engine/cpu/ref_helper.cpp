#include "cpu/ref_helper.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace tona::ref {

namespace {

void read_floats(std::FILE* f, std::vector<float>& v, size_t n) {
    v.resize(n);
    if (std::fread(v.data(), sizeof(float), n, f) != n) throw std::runtime_error("helper file is truncated");
}

}  // namespace

std::shared_ptr<const Helper> load_helper(const std::string& path, size_t n_layer, size_t d) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + path);
    char magic[8];
    uint32_t head[4];
    if (std::fread(magic, 1, 8, f) != 8 || std::string(magic, 8) != "TONAHLP1" || std::fread(head, 4, 4, f) != 4) {
        std::fclose(f);
        throw std::runtime_error(path + ": not a helper file");
    }
    if (head[0] != n_layer || head[1] != d) {
        std::fclose(f);
        throw std::runtime_error(path + ": layer count or width differs from the model");
    }
    auto hp = std::make_shared<Helper>();
    hp->h = head[2], hp->whole = head[3] != 0;
    hp->gate.resize(n_layer), hp->up.resize(n_layer), hp->down.resize(n_layer);
    try {
        for (size_t l = 0; l < n_layer; l++) {
            read_floats(f, hp->gate[l], (size_t)hp->h * d);
            read_floats(f, hp->up[l], (size_t)hp->h * d);
            read_floats(f, hp->down[l], d * (size_t)hp->h);
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
    return hp;
}

std::vector<float> helper_apply(const Helper& hp, size_t l, const std::vector<float>& x) {
    const size_t d = x.size(), h = (size_t)hp.h;
    std::vector<float> a(h);
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < h; r++) {
        double g = 0, u = 0;
        for (size_t i = 0; i < d; i++)
            g += (double)hp.gate[l][r * d + i] * x[i], u += (double)hp.up[l][r * d + i] * x[i];
        a[r] = (float)(g / (1.0 + std::exp(-g)) * u);
    }
    std::vector<float> y(d);
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < d; j++) {
        double acc = 0;
        for (size_t r = 0; r < h; r++) acc += (double)hp.down[l][j * h + r] * a[r];
        y[j] = (float)acc;
    }
    return y;
}

}  // namespace tona::ref
