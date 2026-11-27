// Random PLE row reads: mmap (6 threads) against O_DIRECT through io_uring.
// Usage: tona-ple-io <gguf> [rows] [depth]
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <vector>

#include "io/uring.h"
#include "model/gguf.h"
#include "tona_args.h"

namespace {

int run(int argc, char** argv) {
    if (argc < 2) return 1;
    const size_t n = argc > 2 ? tona::tools::parse_int64(argv[2]) : 32768;
    const unsigned depth = argc > 3 ? tona::tools::parse_int(argv[3]) : 256;
    tona::Gguf g(argv[1]);
    const tona::TensorInfo* t = g.find("per_layer_token_embd.weight");
    const auto* ti = tona::ggml_type_info(t->type);
    const uint32_t rb = (uint32_t)(t->dims[0] / ti->block_elems * ti->block_bytes);
    const uint64_t rows_total = t->dims[1];
    std::mt19937_64 rng(std::random_device{}());
    std::vector<uint8_t> out(n * rb), ref(n * rb);
    const std::string& path = g.file_path(t->file);
    const uint64_t base = g.file_offset(*t);
    tona::DirectRows dio(path, base, rb, true, depth), bio(path, base, rb, false, depth);
    auto rows_new = [&] {
        std::vector<int64_t> rows(n);
        for (auto& r : rows) r = (int64_t)(rng() % rows_total);
        return rows;
    };
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    auto by_mmap = [&](const std::vector<int64_t>& rows) {
#pragma omp parallel for schedule(static) num_threads(6)
        for (size_t i = 0; i < n; i++) std::memcpy(&ref[i * rb], t->data + (uint64_t)rows[i] * rb, rb);
    };
    for (int rep = 0; rep < 3; rep++) {
        double ms[6];
        // cold: page cache of the file empty
        auto rows = rows_new();
        dio.evict();
        auto t0 = std::chrono::steady_clock::now();
        by_mmap(rows);
        ms[0] = ms_since(t0);
        rows = rows_new();
        dio.evict();
        t0 = std::chrono::steady_clock::now();
        dio.read(rows.data(), n, out.data());
        ms[1] = ms_since(t0);
        by_mmap(rows);
        bool same = !std::memcmp(out.data(), ref.data(), out.size());
        rows = rows_new();
        dio.evict();
        t0 = std::chrono::steady_clock::now();
        bio.read(rows.data(), n, out.data());
        ms[2] = ms_since(t0);
        // warm: the same rows again
        t0 = std::chrono::steady_clock::now();
        bio.read(rows.data(), n, out.data());
        ms[3] = ms_since(t0);
        t0 = std::chrono::steady_clock::now();
        by_mmap(rows);
        ms[4] = ms_since(t0);
        same = same && !std::memcmp(out.data(), ref.data(), out.size());
        t0 = std::chrono::steady_clock::now();
        dio.read(rows.data(), n, out.data());
        ms[5] = ms_since(t0);
        printf(
            "%zu rows, depth %u | cold: mmap %.0f, direct %.0f, buffered %.0f ms | warm: buffered %.1f, mmap %.1f, "
            "direct %.0f ms | equal: %s\n",
            n, depth, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], same ? "yes" : "NO");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
