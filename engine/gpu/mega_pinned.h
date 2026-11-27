#pragma once

#include <cstddef>
#include <vector>

namespace tona::gpu {

// Pinned host buffers for copies per pass. Copies from/to ordinary (pageable) memory are pinned by the ROCm runtime
// each time per userptr; freeing triggers a KFD eviction of all queues of the process through the MMU notifier
// (ROCm/rocm-systems#12528, gfx1200): the megakernel stalls, barriers time out, the desktop waits.
template <class T>
struct PinnedAlloc {
    using value_type = T;
    PinnedAlloc() = default;
    template <class U>
    // NOLINTNEXTLINE(misc-explicit-constructor): std::allocator rebinding converts implicitly
    PinnedAlloc(const PinnedAlloc<U>&) {}
    T* allocate(size_t n);
    void deallocate(T* p, size_t) noexcept;
    template <class U>
    bool operator==(const PinnedAlloc<U>&) const {
        return true;
    }
};
using PinnedVec = std::vector<float, PinnedAlloc<float>>;

}  // namespace tona::gpu
