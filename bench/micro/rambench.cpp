// CPU RAM bandwidth with AVX-512: read, write (streaming), copy, over 1..N threads.
// Output: one JSON line per single measurement. Evaluation: bench/micro/run.py
// Usage: rambench [MiB per buffer, default 2048] [repetitions] [pin: compact (default) | spread]
// compact: thread i on CPU i (Linux numbers the first threads of all cores first, then their SMT siblings, so 8 threads
// of a 2-CCD Zen 5 fill one CCD); spread: threads alternate between the two halves of the cores (one CCD each)
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

static double g_sink = 0;

static void op_read(const char* p, size_t n) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i < n; i += 256) {
        a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i));
        a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + i + 64));
        a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + i + 128));
        a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + i + 192));
    }
    g_sink += (double)_mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3)));
}

static void op_write(char* p, size_t n) {
    __m512i v = _mm512_set1_epi64(42);
    for (size_t i = 0; i < n; i += 64) _mm512_stream_si512((__m512i*)(p + i), v);
    _mm_sfence();
}

static void op_copy(const char* s, char* d, size_t n) {
    for (size_t i = 0; i < n; i += 64) _mm512_stream_si512((__m512i*)(d + i), _mm512_load_si512(s + i));
    _mm_sfence();
}

namespace {

struct Buffers {
    char *a, *b;
    size_t bytes;
    int ncpu;
    bool spread;
};

// CPU of thread i: cores first, SMT siblings after (Linux: CPU i and i + cores share a core); `spread` alternates the
// first half of the threads between the two halves of the cores (one CCD of a 2-CCD part each)
int cpu_of(int i, const Buffers& m) {
    const int half = m.ncpu / 4;
    return m.spread && i < m.ncpu / 2 && half > 0 ? (i % 2) * half + i / 2 : i;
}

void run_op(int kind, const Buffers& m, size_t off, size_t chunk) {
    if (kind == 0)
        op_read(m.a + off, chunk);
    else if (kind == 1)
        op_write(m.b + off, chunk);
    else
        op_copy(m.a + off, m.b + off, chunk);
}

// One repetition of `kind` over t threads; prints the rate
void measure(int kind, int t, const Buffers& m) {
    const char* name = kind == 0 ? "read" : kind == 1 ? "write (stream)" : "copy (stream)";
    std::barrier sync(t + 1);
    std::vector<std::thread> th;
    size_t chunk = m.bytes / t / 4096 * 4096;
    for (int i = 0; i < t; i++) {
        th.emplace_back([&, i] {
            pin(cpu_of(i, m));
            sync.arrive_and_wait();
            run_op(kind, m, (size_t)i * chunk, chunk);
            sync.arrive_and_wait();
        });
    }
    sync.arrive_and_wait();
    auto t0 = std::chrono::steady_clock::now();
    sync.arrive_and_wait();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (auto& x : th) x.join();
    double moved = (double)chunk * t * (kind == 2 ? 2 : 1);
    printf("{\"test\":\"ram\",\"var\":\"%s\",\"param\":%d,\"unit\":\"GB/s\",\"value\":%.6g}\n", name, t,
           moved / s / 1e9);
    fflush(stdout);
}

// Thread counts: the usual steps up to every logical CPU (the cores first, then their SMT siblings)
std::vector<int> thread_counts(int ncpu) {
    std::vector<int> counts;
    for (int c : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32})
        if (c < ncpu) counts.push_back(c);
    counts.push_back(ncpu);
    return counts;
}

}  // namespace

int main(int argc, char** argv) {
    size_t mib = argc > 1 ? strtoull(argv[1], nullptr, 10) : 2048;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    const bool spread = argc > 3 && !strcmp(argv[3], "spread");
    size_t bytes = mib << 20;
    char* a = (char*)aligned_alloc(4096, bytes);
    char* b = (char*)aligned_alloc(4096, bytes);
    memset(a, 1, bytes);
    memset(b, 2, bytes);
    int ncpu = (int)std::thread::hardware_concurrency();
    printf("{\"cpus\":%d,\"buffer_mib\":%zu,\"pin\":\"%s\"}\n", ncpu, mib, spread ? "spread" : "compact");
    const Buffers m{a, b, bytes, ncpu, spread};
    for (int t : thread_counts(ncpu))
        for (int kind = 0; kind < 3; kind++)
            for (int r = 0; r < reps; r++) measure(kind, t, m);
    if (g_sink == 0.5) puts("");
    free(a);
    free(b);
    return 0;
}
