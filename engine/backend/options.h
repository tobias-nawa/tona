// What the server tells a backend when it creates it: the backend-neutral part of the reference's MegaOptions. A
// backend maps it to its own options; fields it cannot serve (e.g. no DFlash drafter) make its constructor throw.
#pragma once
#include <cstdint>
#include <vector>

namespace tona {
class Gguf;
struct MtpHead;
}  // namespace tona

namespace tona::backend {

struct CreateOptions {
    int64_t max_ctx = 0;               // length of the KV cache
    std::vector<int64_t> feat_layers;  // inputs of these layers are kept per token, for the DFlash drafter
    const Gguf* drafter = nullptr;     // DFlash drafter, or null
    const MtpHead* mtp = nullptr;      // MTP head, or null
};

}  // namespace tona::backend
