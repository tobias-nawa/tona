// GPU clock ramp probe: does a warm-up strategy raise the clock before the first real kernel after an idle pause?
// Build: hipcc -O2 --offload-arch=gfx1200 clock_ramp.hip -o clock_ramp
// Run:   clock_ramp <strategy: none|spin|ping|queues> <lead_ms or ping period ms> <hold_ms> <trials> <hwmon dir>
// One trial: 2 s idle, strategy for hold_ms (ping, queues) or lead_ms (spin), then 5 timed full-GPU WMMA kernels.
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_ms() { return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count(); }

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
// Matrix-core load (8 independent accumulators per wave): the clock follows it like it follows prefill GEMMs.
__global__ void fma_work(float* out, int iters) {
    half8 a, b;
    for (int i = 0; i < 8; i++) a[i] = (_Float16)(0.01f * (threadIdx.x % 7 + i)), b[i] = (_Float16)(0.02f * (i + 1));
    float8 c[8] = {};
    for (int i = 0; i < iters; i++)
        for (int j = 0; j < 8; j++) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c[j]);
    float s = 0;
    for (int j = 0; j < 8; j++) s += c[j][0];
    if (s == 12345.f) out[0] = s;
}
// One block of 32 threads: the lightest load that keeps the engine busy.
__global__ void spin(long long ticks, float* out) {
    const long long t0 = wall_clock64();
    float x = 1.f;
    while (wall_clock64() - t0 < ticks) x = x * 1.0000001f + 1e-7f;
    if (x == 12345.f) out[0] = x;
}
// Same with s_sleep between polls: lower activity, to see whether the clock policy still counts the engine as busy.
__global__ void spin_sleep(long long ticks, float* out) {
    const long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(20);
    if (ticks == 12345) out[0] = 1.f;
}
__global__ void tiny(float* out) {
    if (threadIdx.x == 99999) out[0] = 1.f;
}

struct Sample {
    double t, mhz, watt;
};
static std::string hw;
static std::atomic<bool> stop_s{false};
static std::vector<Sample> samples;

static double read_num(const std::string& f) {
    std::ifstream in(hw + "/" + f);
    double v = 0;
    in >> v;
    return v;
}
static void sampler() {
    while (!stop_s) {
        samples.push_back({now_ms(), read_num("freq1_input") / 1e6, read_num("power1_average") / 1e6});
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
static void mean_window(double t0, double t1, double& mhz, double& w) {
    double sm = 0, sw = 0;
    int n = 0;
    for (auto& s : samples)
        if (s.t >= t0 && s.t <= t1) sm += s.mhz, sw += s.watt, n++;
    mhz = n ? sm / n : -1, w = n ? sw / n : -1;
}

int main(int argc, char** argv) {
    if (argc < 6) return 2;
    const std::string strat = argv[1];
    const int p = atoi(argv[2]), hold = atoi(argv[3]), trials = atoi(argv[4]);
    hw = argv[5];
    const int iters = argc > 6 ? atoi(argv[6]) : 20000;
    float* d;
    hipMalloc(&d, 4096);
    int wall_khz = 0;
    hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0);
    std::vector<hipStream_t> extra(8);
    hipStream_t s1;
    hipStreamCreate(&s1);
    std::thread th(sampler);
    printf("strategy=%s p=%d hold=%d iters=%d wall_khz=%d\n", strat.c_str(), p, hold, iters, wall_khz);
    printf("trial,k1_ms,k2_ms,k3_ms,k4_ms,k5_ms,k1_mhz,k5_mhz,hold_mhz,hold_w\n");
    for (int tr = 0; tr < trials; tr++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));
        const double t_hold = now_ms();
        hipStream_t ss = nullptr;
        if (strat == "spin" || strat == "spinsl") {
            hipStreamCreate(&ss);
            // outlives the lead by the work time
            const long long ticks = (long long)wall_khz * (p + 900);
            if (strat == "spin") {
                spin<<<1, 32, 0, ss>>>(ticks, d);
            } else {
                spin_sleep<<<1, 32, 0, ss>>>(ticks, d);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(p));
        } else if (strat == "ping") {
            const double end = now_ms() + hold;
            while (now_ms() < end) {
                tiny<<<1, 1, 0, s1>>>(d);
                hipStreamSynchronize(s1);
                std::this_thread::sleep_for(std::chrono::milliseconds(p));
            }
        } else if (strat == "queues") {  // create streams and touch each once, then idle for the hold
            for (int i = 0; i < p; i++) {
                hipStreamCreate(&extra[i]);
                tiny<<<1, 1, 0, extra[i]>>>(d);
            }
            hipDeviceSynchronize();
            std::this_thread::sleep_for(std::chrono::milliseconds(hold));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(hold));
        }
        const double t_work = now_ms();
        hipEvent_t e[6];
        for (auto& ev : e) hipEventCreate(&ev);
        hipEventRecord(e[0], s1);
        for (int k = 0; k < 5; k++) {
            fma_work<<<256, 256, 0, s1>>>(d, iters);
            hipEventRecord(e[k + 1], s1);
        }
        hipStreamSynchronize(s1);
        const double t_end = now_ms();
        float ms[5];
        for (int k = 0; k < 5; k++) hipEventElapsedTime(&ms[k], e[k], e[k + 1]);
        double m1, w1, m3, w3, mh, wh;
        mean_window(t_work, t_work + ms[0], m1, w1);
        mean_window(t_end - ms[4], t_end, m3, w3);
        mean_window(t_hold, t_work, mh, wh);
        printf("%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.0f,%.0f,%.0f,%.1f\n", tr, ms[0], ms[1], ms[2], ms[3], ms[4], m1, m3, mh,
               wh);
        fflush(stdout);
        if (ss) hipStreamDestroy(ss);
    }
    stop_s = true;
    th.join();
    return 0;
}
