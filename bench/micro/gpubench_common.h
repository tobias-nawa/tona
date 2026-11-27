// Helpers shared by the GPU micro-benchmarks: error check, JSON emitter, event timer.
#pragma once
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                               \
    do {                                                                                       \
        hipError_t e_ = (x);                                                                   \
        if (e_ != hipSuccess) {                                                                \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            exit(2);                                                                           \
        }                                                                                      \
    } while (0)

static int g_reps = 5;
static int g_cus = 0;

static void emit(const char* test, const char* var, double param, const char* unit, double value) {
    printf("{\"test\":\"%s\",\"var\":\"%s\",\"param\":%.0f,\"unit\":\"%s\",\"value\":%.6g}\n", test, var, param, unit,
           value);
    fflush(stdout);
}

// Time of `fn` on the default stream in ms
template <class F>
static float timed(F fn) {
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    CHECK(hipEventRecord(a));
    fn();
    CHECK(hipEventRecord(b));
    CHECK(hipEventSynchronize(b));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b));
    CHECK(hipEventDestroy(a));
    CHECK(hipEventDestroy(b));
    return ms;
}
