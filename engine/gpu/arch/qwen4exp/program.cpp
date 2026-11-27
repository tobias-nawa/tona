// The Coder as an ArchProgram: construction.
#include "gpu/arch/qwen4exp/program.h"

#include <algorithm>
#include <stdexcept>

#include "backend/env.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

bool fixed_shapes_ok(const Hparams& h, const backend::ExpertLayout& experts) {
    return h.dn_head_dim == DS && h.head_dim == 256 && h.rope_dims == 64 && h.idx_dim == 128 && h.n_hc <= 4 &&
           h.n_expert_used * BMAX <= UMAX && experts.max_experts() <= (uint32_t)MEGA_MAX_EXPERTS_WIDE;
}

// Cells per token (top-k blocks plus tail) in LDS and in the merge; the WMMA attention computes at most 16 heads per
// KV head; the layered prefill loads embedding and PLE only in layer 0 (PLE layer <= 1)
bool attention_shapes_ok(const Hparams& h) {
    return h.idx_top_k + h.idx_block - 1 <= MAX_CELLS && h.n_head / std::max<uint32_t>(h.n_head_kv, 1) <= 16 &&
           h.ple_layer <= 1;
}

}  // namespace

// The kernels assume fixed shapes of the model
void qwen4exp_check_shapes(const Qwen4ExpModel& m, const backend::ExpertLayout& experts) {
    if (!fixed_shapes_ok(m.hp(), experts))
        throw std::runtime_error("gpu::Mega: shapes do not match the fixed kernel assumptions");
    if (!attention_shapes_ok(m.hp()))
        throw std::runtime_error("gpu::Mega: attention or PLE do not match the fixed kernel assumptions");
}

Qwen4ExpProgram::Qwen4ExpProgram(ProgramBuilder& pb)
    : pb_(pb),
      g_(pb.mega()),
      h_(pb.hparams()),
      m_(pb.model()),
      dm_(dims_of(pb.hparams())),
      hc8_(pb.mega().int8_ && !env::flag("TONA_MEGA_HC_BF16")) {}

}  // namespace tona::gpu
