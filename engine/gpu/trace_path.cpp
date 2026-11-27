#include "gpu/trace_path.h"

namespace tona::gpu::trace {

std::string expand_pid(std::string_view path, long pid) {
    const std::string num = std::to_string(pid);
    std::string out;
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '%' && i + 1 < path.size() && path[i + 1] == 'p') {
            out += num;
            ++i;
        } else {
            out += path[i];
        }
    }
    return out;
}

}  // namespace tona::gpu::trace
