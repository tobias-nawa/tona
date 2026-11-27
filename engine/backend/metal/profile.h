// TONA_METAL_PROFILE=1: GPU time per part of a pass. Every part (embedding, PLE, gated reads, mixer, experts, ...)
// gets its own command buffer, and the GPU start and end of each buffer are summed per part name; Apple7 GPUs cannot
// take timestamps between dispatches, only at command buffer boundaries. The split adds ~25 µs per part
// (research/measurements/2026-11-11-mac-dispatch.md), so profiled runs are not speed results. After command buffers
// of ~2 s the system left the GPU idle for 0.1 to 0.3 s between the next submissions (25 s over ~200 decode buffers
// after a 2048-token chunk; without the profiler one buffer per pass, and decode was not slower), so profile with
// short parts only. The totals go to stderr when the backend ends.
#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "backend/metal/device.h"

namespace tona::metal {

class Profiler {
public:
    explicit Profiler(Device& dev) : dev_(dev) {}
    ~Profiler();
    // A new command buffer for the part `name`; the previous one is committed
    id<MTLComputeCommandEncoder> next(const std::string& name);
    id<MTLCommandBuffer> current() const { return cb_; }
    // Commit the last part, wait for all, add their GPU times to the totals
    void end();

private:
    void collect();

    Device& dev_;
    id<MTLCommandBuffer> cb_ = nil;
    id<MTLComputeCommandEncoder> enc_ = nil;
    std::string name_;
    std::vector<std::pair<std::string, id<MTLCommandBuffer>>> open_;
    std::map<std::string, std::pair<double, int64_t>> total_;  // seconds and command buffers per part
};

}  // namespace tona::metal
