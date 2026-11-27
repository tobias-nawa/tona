#include "backend/metal/profile.h"

#include <cstdio>
#include <stdexcept>

namespace tona::metal {

id<MTLComputeCommandEncoder> Profiler::next(const std::string& name) {
    if (enc_) {
        [enc_ endEncoding];
        [cb_ commit];
        open_.push_back({name_, cb_});
    }
    // A queue hands out a limited number of command buffers (64 by default); creating more blocks until the system
    // recycles one, which took ~25 s per decode pass (sampled: semaphore wait in commandBuffer)
    if (open_.size() >= 32) collect();
    cb_ = [dev_.queue() commandBuffer];
    enc_ = [cb_ computeCommandEncoder];
    name_ = name;
    return enc_;
}

void Profiler::end() {
    if (enc_) {
        [enc_ endEncoding];
        [cb_ commit];
        open_.push_back({name_, cb_});
    }
    enc_ = nil, cb_ = nil;
    collect();
}

void Profiler::collect() {
    for (auto& [name, cb] : open_) {
        [cb waitUntilCompleted];
        if (cb.error) throw std::runtime_error(std::string("Metal: pass failed: ") + cb.error.description.UTF8String);
        auto& t = total_[name];
        t.first += cb.GPUEndTime - cb.GPUStartTime;
        t.second++;
    }
    open_.clear();
}

Profiler::~Profiler() {
    double all = 0;
    for (const auto& [name, t] : total_) all += t.first;
    if (all <= 0) return;
    std::fprintf(stderr, "Metal profile (GPU time per part, all passes):\n");
    for (const auto& [name, t] : total_)
        std::fprintf(stderr, "  %-12s %10.1f ms  %5.1f %%  %lld buffers\n", name.c_str(), t.first * 1e3,
                     100 * t.first / all, (long long)t.second);
    std::fprintf(stderr, "  %-12s %10.1f ms\n", "total", all * 1e3);
}

}  // namespace tona::metal
