#include "backend/mtp_scorer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

#include "model/quant.h"

namespace tona::backend {

namespace {

constexpr uint32_t kMagic = 0x32435354;  // 'TSC2': format version 2 of the scorer cache (1 had no fingerprint)
constexpr size_t kHeaderWords = 8;       // magic, rank, d, vocab, fingerprint (lo, hi), reserved x 2
constexpr size_t kChunk = 2048;          // head rows per step
// Every 2nd row for W^T W: the top subspace is the same (recall of the draft rows 99.27 % against 99.20 % with all
// rows on real head inputs, research/measurements/2026-11-12-mtp-dynamic-head.md), at half the time
constexpr size_t kGramStride = 2;
constexpr int kIterations = 12;  // as llama.cpp 30380; 12 and 40 gave the same recall

// Rows [r0, r0 + n) of the head as float [n][d]
void head_chunk(const Gguf& g, const TensorInfo& head, size_t r0, size_t n, int d, std::vector<float>& out) {
    const size_t rb = head.bytes / head.dims[1];
    out.resize(n * (size_t)d);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; i++) {
        std::vector<uint8_t> src(rb);
        g.read(head, (uint64_t)(r0 + (size_t)i) * rb, src.data(), rb);
        dequantize_row(head.type, src.data(), &out[(size_t)i * d], d);
    }
}

// G += C^T C over every kGramStride-th row of the chunk C [n][d], lower triangle (G is [d][d], double)
void add_gram(const std::vector<float>& c, size_t n, int d, std::vector<double>& gram) {
    const size_t m = (n + kGramStride - 1) / kGramStride;
    std::vector<float> ct((size_t)d * m);
    for (size_t i = 0; i < m; i++)
        for (int k = 0; k < d; k++) ct[(size_t)k * m + i] = c[i * kGramStride * (size_t)d + k];
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < d; i++) {
        const float* a = &ct[(size_t)i * m];
        for (int j = 0; j <= i; j++) {
            const float* b = &ct[(size_t)j * m];
            float s = 0;
#pragma omp simd reduction(+ : s)
            for (size_t k = 0; k < m; k++) s += a[k] * b[k];
            gram[(size_t)i * d + j] += s;
        }
    }
}

// Orthonormal columns (modified Gram-Schmidt) of q [d][r], column-major in qt [r][d]
void orthonormalize(std::vector<double>& qt, int r, int d) {
    for (int c = 0; c < r; c++) {
        double* v = &qt[(size_t)c * d];
        for (int p = 0; p < c; p++) {
            const double* u = &qt[(size_t)p * d];
            double dot = 0;
            for (int k = 0; k < d; k++) dot += u[k] * v[k];
            for (int k = 0; k < d; k++) v[k] -= dot * u[k];
        }
        double nn = 0;
        for (int k = 0; k < d; k++) nn += v[k] * v[k];
        nn = std::sqrt(nn);
        for (int k = 0; k < d; k++) v[k] = nn > 0 ? v[k] / nn : 0;
    }
}

// Top-r subspace of the symmetric matrix G (lower triangle filled) by subspace iteration from a fixed random start
std::vector<double> top_subspace(std::vector<double>& gram, int d, int r) {
    for (int i = 0; i < d; i++)
        for (int j = i + 1; j < d; j++) gram[(size_t)i * d + j] = gram[(size_t)j * d + i];
    std::vector<double> q((size_t)r * d), z((size_t)r * d);
    uint64_t state = 0x853c49e6748fea9bULL;  // splitmix64
    for (double& v : q) {
        uint64_t x = (state += 0x9e3779b97f4a7c15ULL);
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        v = (double)((x ^ (x >> 31)) >> 11) / 9007199254740992.0 - 0.5;
    }
    orthonormalize(q, r, d);
    for (int it = 0; it < kIterations; it++) {
#pragma omp parallel for schedule(static)
        for (int c = 0; c < r; c++)
            for (int i = 0; i < d; i++) {
                const double* gi = &gram[(size_t)i * d];
                const double* qc = &q[(size_t)c * d];
                double s = 0;
#pragma omp simd reduction(+ : s)
                for (int k = 0; k < d; k++) s += gi[k] * qc[k];
                z[(size_t)c * d + i] = s;
            }
        q = z;
        orthonormalize(q, r, d);
    }
    return q;
}

// Scorer rows of the chunk: p = C B, int8 with a scale per row
void score_chunk(const std::vector<float>& c, size_t n, int d, const std::vector<float>& bt, MtpScorer& s, size_t r0) {
    const int r = s.rank;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; i++) {
        std::vector<float> p((size_t)r);
        float amax = 0;
        for (int k = 0; k < r; k++) {
            float acc = 0;
            const float* w = &c[(size_t)i * d];
            const float* b = &bt[(size_t)k * d];
#pragma omp simd reduction(+ : acc)
            for (int j = 0; j < d; j++) acc += w[j] * b[j];
            p[(size_t)k] = acc, amax = std::max(amax, std::fabs(acc));
        }
        const float sc = amax > 0 ? amax / 127.0f : 1.0f;
        s.scale[r0 + (size_t)i] = sc;
        for (int k = 0; k < r; k++) s.rows[(r0 + (size_t)i) * r + (size_t)k] = (int8_t)std::lrintf(p[(size_t)k] / sc);
    }
}

}  // namespace

MtpScorer make_mtp_scorer(const Gguf& g, const TensorInfo& head, int d, size_t vocab, int rank) {
    std::vector<double> gram((size_t)d * d, 0.0);
    std::vector<float> c;
    for (size_t r0 = 0; r0 < vocab; r0 += kChunk) {
        const size_t n = std::min(kChunk, vocab - r0);
        head_chunk(g, head, r0, n, d, c);
        add_gram(c, n, d, gram);
    }
    const std::vector<double> q = top_subspace(gram, d, rank);
    MtpScorer s;
    s.rank = rank;
    s.basis_t.assign(q.begin(), q.end());
    s.scale.resize(vocab), s.rows.resize(vocab * (size_t)rank);
    for (size_t r0 = 0; r0 < vocab; r0 += kChunk) {
        const size_t n = std::min(kChunk, vocab - r0);
        head_chunk(g, head, r0, n, d, c);
        score_chunk(c, n, d, s.basis_t, s, r0);
    }
    return s;
}

uint64_t head_fingerprint(uint32_t type, uint64_t bytes, size_t rows, const RowReader& read_row) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ULL;
    };
    mix(reinterpret_cast<const uint8_t*>(&type), sizeof type);
    mix(reinterpret_cast<const uint8_t*>(&bytes), sizeof bytes);
    if (rows == 0) return h;
    const size_t rb = (size_t)(bytes / rows);
    std::vector<uint8_t> row(rb);
    const auto one = [&](size_t r) { read_row(r, row.data(), rb), mix(row.data(), rb); };
    one(0);
    for (size_t r = 64; r < rows - 1; r += 64) one(r);
    one(rows - 1);
    return h;
}

MtpScorer read_mtp_scorer(const std::string& path, int rank, int d, size_t vocab, uint64_t fingerprint) {
    MtpScorer s;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return s;
    const size_t want = kHeaderWords * 4 + ((size_t)rank * d + vocab) * 4 + vocab * (size_t)rank;
    if ((size_t)in.tellg() != want) return s;
    in.seekg(0);
    uint32_t h[kHeaderWords] = {};
    if (!in.read(reinterpret_cast<char*>(h), sizeof h)) return s;
    if (h[0] != kMagic || (int)h[1] != rank || (int)h[2] != d || h[3] != vocab) return s;
    if (h[4] != (uint32_t)fingerprint || h[5] != (uint32_t)(fingerprint >> 32)) return s;
    s.basis_t.resize((size_t)rank * d), s.scale.resize(vocab), s.rows.resize(vocab * (size_t)rank);
    in.read(reinterpret_cast<char*>(s.basis_t.data()), (std::streamsize)(s.basis_t.size() * 4));
    in.read(reinterpret_cast<char*>(s.scale.data()), (std::streamsize)(s.scale.size() * 4));
    in.read(reinterpret_cast<char*>(s.rows.data()), (std::streamsize)s.rows.size());
    if (in) s.rank = rank;
    return s;
}

bool write_mtp_scorer(const std::string& path, const MtpScorer& s, int d, size_t vocab, uint64_t fingerprint) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        const uint32_t h[kHeaderWords] = {kMagic,
                                          (uint32_t)s.rank,
                                          (uint32_t)d,
                                          (uint32_t)vocab,
                                          (uint32_t)fingerprint,
                                          (uint32_t)(fingerprint >> 32),
                                          0,
                                          0};
        out.write(reinterpret_cast<const char*>(h), sizeof h);
        out.write(reinterpret_cast<const char*>(s.basis_t.data()), (std::streamsize)(s.basis_t.size() * 4));
        out.write(reinterpret_cast<const char*>(s.scale.data()), (std::streamsize)(s.scale.size() * 4));
        out.write(reinterpret_cast<const char*>(s.rows.data()), (std::streamsize)s.rows.size());
        out.flush();
        if (!out) {
            out.close();
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) == 0) return true;
    std::remove(tmp.c_str());
    return false;
}

namespace {

// $XDG_CACHE_HOME/tona, else ~/.cache/tona; empty if neither is known
std::string cache_dir() {
    const char* x = std::getenv("XDG_CACHE_HOME");
    if (x && *x) return std::string(x) + "/tona";
    const char* home = std::getenv("HOME");
    return home && *home ? std::string(home) + "/.cache/tona" : std::string();
}

}  // namespace

MtpScorer load_or_make_mtp_scorer(const Gguf& g, const TensorInfo& head, int rank, const std::string& model_path,
                                  const char* file) {
    const int d = (int)head.dims[0];
    const size_t vocab = head.dims[1];
    const size_t rb = head.bytes / head.dims[1];
    const uint64_t fp = head_fingerprint(head.type, head.bytes, vocab, [&](size_t row, uint8_t* dst, size_t n) {
        g.read(head, (uint64_t)row * rb, dst, n);
    });
    char fps[17];
    std::snprintf(fps, sizeof fps, "%016llx", (unsigned long long)fp);
    std::vector<std::string> paths;
    if (file) {
        paths.emplace_back(file);
    } else {
        paths.push_back(model_path + ".mtp-scorer-r" + std::to_string(rank));
        if (const std::string dir = cache_dir(); !dir.empty())
            paths.push_back(dir + "/mtp-scorer-r" + std::to_string(rank) + "-" + fps + ".bin");
    }
    bool stale = false;
    for (const std::string& p : paths) {
        MtpScorer s = read_mtp_scorer(p, rank, d, vocab, fp);
        if (s.rank != 0) return s;
        stale = stale || std::ifstream(p).good();
    }
    if (stale)
        std::fprintf(stderr,
                     "MTP scorer file does not match this head (other model, fingerprint %s, or damaged): "
                     "making it again\n",
                     fps);
    const auto t0 = std::chrono::steady_clock::now();
    MtpScorer s = make_mtp_scorer(g, head, d, vocab, rank);
    std::string kept;
    for (const std::string& p : paths) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(p).parent_path(), ec);
        if (write_mtp_scorer(p, s, d, vocab, fp)) {
            kept = p;
            break;
        }
    }
    std::fprintf(stderr, "Initialization: MTP scorer of rank %d in %.2f s%s%s\n", rank,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 kept.empty() ? ", not kept (no writable folder)" : ", kept in ", kept.c_str());
    return s;
}

}  // namespace tona::backend
