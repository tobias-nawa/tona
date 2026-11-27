// Small helpers shared by the HIP check and probe tools: error check and an owning device buffer.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

inline void hip_check(hipError_t err, const char* expr, const char* file, int line) {
    if (err == hipSuccess) return;
    fprintf(stderr, "%s:%d: %s: %s\n", file, line, expr, hipGetErrorString(err));
    std::exit(1);
}
#define CHECK(x) hip_check((x), #x, __FILE__, __LINE__)

namespace tona::tools {

// Device memory freed at scope end
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) { CHECK(hipMalloc(&ptr_, bytes)); }
    ~DeviceBuffer() { (void)hipFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    template <class T = void>
    T* get() const {
        return static_cast<T*>(ptr_);
    }

private:
    void* ptr_ = nullptr;
};

}  // namespace tona::tools
