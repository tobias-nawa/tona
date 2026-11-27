// TONA_ALLOC_TRACE=<file> ("%p" in the path becomes the PID): one line per device or pinned allocation, free and
// borrowed range of the Mega backend, so a fault address can be assigned to a buffer afterwards
// (tools/alloc_trace_lookup.py). Without the variable the wrappers are the plain HIP calls plus one load of a static
// flag.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <source_location>

namespace tona::gpu::trace {

bool on();

// kind: "hipMalloc", "hipHostMalloc", "hipFree", "hipHostFree", or a range name such as "arena" (a borrowed range
// inside an allocation); where: file:line of the call or a label
void record(const char* kind, const void* p, size_t bytes, const char* where);
void record(const char* kind, const void* p, size_t bytes, const std::source_location& loc);

template <class T>
hipError_t dev_malloc(T** p, size_t n, std::source_location loc = std::source_location::current()) {
    const hipError_t e = hipMalloc(p, n);
    if (e == hipSuccess && on()) record("hipMalloc", *p, n, loc);
    return e;
}

template <class T>
hipError_t host_malloc(T** p, size_t n, unsigned flags, std::source_location loc = std::source_location::current()) {
    const hipError_t e = hipHostMalloc(reinterpret_cast<void**>(p), n, flags);
    if (e == hipSuccess && on()) record("hipHostMalloc", *p, n, loc);
    return e;
}

inline hipError_t dev_free(void* p, std::source_location loc = std::source_location::current()) {
    if (on()) record("hipFree", p, 0, loc);
    return hipFree(p);
}

inline hipError_t host_free(void* p, std::source_location loc = std::source_location::current()) {
    if (on()) record("hipHostFree", p, 0, loc);
    return hipHostFree(p);
}

}  // namespace tona::gpu::trace
