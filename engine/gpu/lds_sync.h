// Workgroup barrier for the sites that protect only LDS. __syncthreads() is a release/acquire fence over all address
// spaces plus a barrier; in the default WGP mode the fences make every wave wait for its in-flight global loads
// (s_wait_loadcnt 0) and invalidate the L0 cache (global_inv scope:SCOPE_SE), so a prefetch issued before the barrier
// is waited for there. The fence on the local address space only waits for the LDS operations (s_wait_dscnt 0).
// Use it only where no thread reads, after the barrier, global memory that another thread of the block wrote before
// it; then LDS is all the barrier has to order. Data from other blocks is ordered by the grid barrier (its agent
// fence), not by this one. Compiled probe: research/measurements/2026-11-18-b7-lds-barriers.md.
#pragma once

#include <hip/hip_runtime.h>

namespace tona::gpu {
namespace {

__device__ __forceinline__ void lds_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

}  // namespace
}  // namespace tona::gpu
