#include "gpu/alloc_trace.h"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "backend/env.h"
#include "gpu/trace_path.h"

namespace tona::gpu::trace {

namespace {

struct Sink {
    std::FILE* f = nullptr;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    std::mutex mu;
    Sink() {
        if (const char* path = tona::env::text("TONA_ALLOC_TRACE")) {
            f = std::fopen(expand_pid(path, static_cast<long>(::getpid())).c_str(), "w");
        }
    }
    ~Sink() {
        if (f) std::fclose(f);
    }
};

Sink& sink() {
    static Sink s;
    return s;
}

}  // namespace

bool on() {
    static const bool v = sink().f != nullptr;
    return v;
}

void record(const char* kind, const void* p, size_t bytes, const char* where) {
    Sink& s = sink();
    if (!s.f) return;
    const std::lock_guard<std::mutex> lock(s.mu);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.t0).count();
    // flushed per line: the process usually ends in an abort
    std::fprintf(s.f, "%.3f %s %p %zu %s\n", ms, kind, p, bytes, where);
    std::fflush(s.f);
}

void record(const char* kind, const void* p, size_t bytes, const std::source_location& loc) {
    const char* file = loc.file_name();
    if (const char* slash = std::strrchr(file, '/')) file = slash + 1;
    char where[160];
    std::snprintf(where, sizeof where, "%s:%u", file, (unsigned)loc.line());
    record(kind, p, bytes, where);
}

}  // namespace tona::gpu::trace
