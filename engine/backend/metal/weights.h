// Weights of the model on the GPU. Matrices stay in the GGUF mapping: every file becomes one no-copy Metal buffer
// (unified memory, nothing is copied). Small tensors the kernels read as plain floats (norm weights, conv weights,
// biases) are dequantized once into f32 buffers.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend/metal/arena.h"
#include "backend/metal/device.h"
#include "model/qwen4exp.h"

namespace tona::metal {

// A matrix [k inputs, rows outputs] or a stack of experts [k, rows, experts]
struct Mat {
    Buf w;
    std::string kernel_type;  // suffix of the mv_/gather_ kernels, e.g. "Q4K"
    uint32_t k = 0, rows = 0, row_bytes = 0, experts = 1;
    uint64_t expert_bytes = 0;
    const uint8_t* host = nullptr;  // tensors the GPU never maps: their rows in the file mapping
};

class Weights {
public:
    Weights(Device& dev, const Qwen4ExpModel& m);
    // The tensors of a second file (the MTP head's GGUF), mapped the same way, without the ones named in skip
    void add(const Gguf& g, const std::vector<std::string>& skip = {});
    Mat mat(T t) const;                     // throws for a type without a kernel
    static Mat shape(const TensorInfo& t);  // type, sizes and row bytes, without a buffer
    Buf vec(T t);                           // dequantized to f32, once per tensor
    uint64_t mapped_bytes() const { return mapped_ + (arena_ ? arena_->bytes() : 0); }
    // The expert arena (TONA_METAL_ARENA_GB) or nullptr: its experts are not in the buffers of the mapping
    Arena* arena() const { return arena_.get(); }
    std::vector<id<MTLBuffer>> buffers() const {
        std::vector<id<MTLBuffer>> out;
        out.reserve(bufs_.size());
        for (const auto& [b, base] : bufs_) out.push_back(b);
        return out;
    }
    // Probe (TONA_METAL_PIN): keep the mapped weights from being dropped under memory pressure for the process's
    // life: "rset" (a residency set on the queue), "mlock" (all windows), "mlock-dense" (all but the routed experts).
    // Returns the bytes pinned. research/measurements/2026-11-14-mac-pin.md
    uint64_t pin(const std::string& mode);

private:
    Device& dev_;
    std::vector<std::pair<id<MTLBuffer>, const uint8_t*>>
        bufs_;             // buffer windows over the mapped files and their bases
    std::map<T, Buf> at_;  // where each tensor lies
    std::map<T, Buf> vecs_;
    uint64_t mapped_ = 0;
    T host_ = nullptr;                                        // the one tensor read on the host (the n-gram table)
    std::vector<std::pair<const uint8_t*, uint64_t>> dense_;  // ranges of the tensors that are not routed experts
    id rset_ = nil;                                           // id<MTLResidencySet> (macOS 15) while pinned
    std::unique_ptr<Arena> arena_;
};

// ggml type id -> kernel suffix; the types the backend has kernels for, the ones it declares to the registry
const std::map<uint32_t, std::string>& kernel_types();
// ggml type id -> kernel suffix ("" if there is no kernel)
std::string kernel_type(uint32_t ggml_type);

}  // namespace tona::metal
