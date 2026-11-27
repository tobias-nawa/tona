// The helper file of the draft pass: per layer one SwiGLU FFN d -> h -> d.
// Layout, little endian: "TONAHLP1", u32 n_layer, u32 d, u32 h, u32 whole, then per layer F32 gate[h][d], up[h][d],
// down[d][h].
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpu/ref.h"

namespace tona::ref {

std::shared_ptr<const Helper> load_helper(const std::string& path, size_t n_layer, size_t d);

// down(silu(gate x) * (up x)) of layer l
std::vector<float> helper_apply(const Helper& hp, size_t l, const std::vector<float>& x);

}  // namespace tona::ref
