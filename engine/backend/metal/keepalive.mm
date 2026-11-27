#include "backend/metal/keepalive.h"

namespace tona::metal {

Keepalive::Keepalive(id<MTLDevice> dev, double every_ms, double window_s) : every_(every_ms / 1e3), window_(window_s) {
    if (every_ms <= 0) return;
    if (dev) {
        queue_ = [dev newCommandQueue];
        buf_ = [dev newBufferWithLength:16 options:MTLResourceStorageModePrivate];
    }
    thread_ = std::thread([this] { run(); });
}

Keepalive::~Keepalive() {
    {
        const std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void Keepalive::reads(const std::vector<id<MTLBuffer>>& bufs) {
    const std::lock_guard<std::mutex> lock(mu_);
    reads_ = bufs;
}

void Keepalive::touch() {
    bool asleep = false;
    {
        const std::lock_guard<std::mutex> lock(mu_);
        const Clock::time_point now = Clock::now();
        asleep = !inside(now);
        last_ = now, touched_ = true;
    }
    if (asleep) cv_.notify_all();  // during decode the thread keeps its timer: a wake per pass would send per pass
}

bool Keepalive::due(Clock::time_point now) const {
    const std::lock_guard<std::mutex> lock(mu_);
    return idle(now);
}

long long Keepalive::sent() const {
    const std::lock_guard<std::mutex> lock(mu_);
    return sent_;
}

// 16 bytes of every weight buffer: a command buffer that uses them keeps them ready; the host does not wait
void Keepalive::send() {
    if (!queue_) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        std::vector<id<MTLBuffer>> reads;
        {
            const std::lock_guard<std::mutex> lock(mu_);
            reads = reads_;
        }
        for (id<MTLBuffer> b : reads) [blit copyFromBuffer:b sourceOffset:0 toBuffer:buf_ destinationOffset:0 size:16];
        [blit endEncoding];
        [cb commit];
    }
}

void Keepalive::run() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!stop_) {
        if (!inside(Clock::now())) {  // outside the window: sleep until the next pass
            cv_.wait(lock, [&] { return stop_ || inside(Clock::now()); });
            continue;
        }
        cv_.wait_for(lock, every_);
        if (stop_ || !idle(Clock::now())) continue;
        sent_++;
        lock.unlock();
        send();
        lock.lock();
    }
}

}  // namespace tona::metal
