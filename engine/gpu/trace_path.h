// TONA_ALLOC_TRACE path expansion. Plain C++ (no HIP) so a unit test can run it.
#pragma once

#include <string>
#include <string_view>

namespace tona::gpu::trace {

// Replaces every "%p" in the path by the PID: parallel processes with the same TONA_ALLOC_TRACE would otherwise open
// the file with "w" and overwrite each other's trace.
std::string expand_pid(std::string_view path, long pid);

}  // namespace tona::gpu::trace
