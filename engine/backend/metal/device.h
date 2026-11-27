// Metal device, queue, kernel library and the helpers the pass encoders share. Objective-C++ only (included by .mm).
#pragma once
#import <Metal/Metal.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend/metal/keepalive.h"

namespace tona::metal {

// A range inside a Metal buffer
struct Buf {
    id<MTLBuffer> b = nil;
    uint64_t off = 0;
    Buf at(uint64_t bytes) const { return {b, off + bytes}; }
    float* f32() const { return (float*)((uint8_t*)b.contents + off); }
    int32_t* i32() const { return (int32_t*)((uint8_t*)b.contents + off); }
};

// Kernel library: the prebuilt tona.metallib next to the program (bin/ or lib/tona/), or, for development only,
// the sources under TONA_METAL_SOURCES compiled at start
id<MTLLibrary> load_library(id<MTLDevice> dev);

class Device {
public:
    Device();
    id<MTLDevice> dev() const { return dev_; }
    id<MTLCommandQueue> queue() const { return queue_; }
    id<MTLComputePipelineState> pipe(const std::string& name);
    Buf floats(uint64_t n);  // shared storage, zeroed
    uint64_t allocated() const { return allocated_; }
    // Before a pass (keepalive.h)
    void touch() {
        if (keep_) keep_->touch();
    }
    void keepalive_reads(const std::vector<id<MTLBuffer>>& bufs) {
        if (keep_) keep_->reads(bufs);
    }

private:
    id<MTLDevice> dev_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> lib_ = nil;
    std::map<std::string, id<MTLComputePipelineState>> pipes_;
    uint64_t allocated_ = 0;
    // Probe (TONA_METAL_PIN_OWN=1): every buffer of floats() in a residency set on the queue, so pages the GPU has not
    // used for a while (decode's during a long prefill) stay resident;
    // research/measurements/2026-11-15-mac-cold-submit.log
    id own_ = nil;  // id<MTLResidencySet>
    std::unique_ptr<Keepalive> keep_;
};

// One dispatch: pipeline, buffers at indices 0.., constant bytes at the next index, threadgroups and their size
struct Dispatch {
    id<MTLComputeCommandEncoder> enc;
    Device& dev;

    template <class P>
    void run(const std::string& kernel, std::initializer_list<Buf> bufs, const P& params, MTLSize grid, MTLSize tg) {
        run_raw(kernel, bufs, &params, sizeof(P), grid, tg);
    }
    void run_raw(const std::string& kernel, std::initializer_list<Buf> bufs, const void* params, size_t bytes,
                 MTLSize grid, MTLSize tg);
};

}  // namespace tona::metal
