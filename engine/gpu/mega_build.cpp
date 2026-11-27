// Mega::build(): the op programs and everything they point to.
#include "gpu/mega_build.h"

#include <stdexcept>

#include "backend/env.h"
#include "gpu/arch/nemotron_h/kernel_set.h"
#include "gpu/arch/nemotron_h/program.h"
#include "gpu/arch/qwen35moe/kernel_set.h"
#include "gpu/arch/qwen35moe/program.h"
#include "gpu/arch/qwen4exp/kernel_set.h"
#include "gpu/arch/qwen4exp/program.h"
#include "gpu/mega_dump.h"
#include "gpu/mega_util.h"
#include "model/arch.h"

namespace tona::gpu {

void Mega::build() {
    if (const char* f = env::text("TONA_MEGA_DUMP_PROGRAM")) dump_ = std::make_unique<ProgramDump>(*this, f);
    ProgramBuilder pb(*this);
    const std::unique_ptr<ArchProgram> arch = make_arch_program(pb);
    pb.run(*arch);
}

// The one place that maps the loaded model to the architecture's program, kernels and shape checks
std::unique_ptr<ArchProgram> make_arch_program(ProgramBuilder& pb) {
    switch (pb.base().arch()) {
        case ArchId::Qwen4Exp:
            return std::make_unique<Qwen4ExpProgram>(pb);
        case ArchId::Qwen35Moe:
            return std::make_unique<Qwen35MoeProgram>(pb);
        case ArchId::NemotronHMoe:
            return std::make_unique<NemotronHProgram>(pb);
        default:
            throw std::runtime_error(std::string("gpu::Mega: no HIP program for ") +
                                     std::string(arch_name(pb.base().arch())));
    }
}

const KernelSet& kernels_of(const ModelBase& m) {
    switch (m.arch()) {
        case ArchId::Qwen4Exp:
            // The size class of the experts per layer picks the instantiation of the same architecture
            return expert_cap(model_as<Qwen4ExpModel>(m).hp().n_expert) > MEGA_MAX_EXPERTS ? qwen4exp_512_kernels()
                                                                                           : qwen4exp_kernels();
        case ArchId::Qwen35Moe:
            return qwen35moe_kernels();
        case ArchId::NemotronHMoe:
            return nemotron_h_kernels();
        default:
            throw std::runtime_error(std::string("gpu::Mega: no HIP kernels for ") + std::string(arch_name(m.arch())));
    }
}

void check_arch_shapes(const ModelBase& m, const backend::ExpertLayout& experts) {
    switch (m.arch()) {
        case ArchId::Qwen4Exp:
            return qwen4exp_check_shapes(model_as<Qwen4ExpModel>(m), experts);
        case ArchId::Qwen35Moe:
            return qwen35moe_check_shapes(model_as<Qwen35MoeModel>(m), experts);
        case ArchId::NemotronHMoe:
            return nemotron_h_check_shapes(model_as<NemotronHMoeModel>(m), experts);
        default:
            throw std::runtime_error(std::string("gpu::Mega: no HIP kernels for ") + std::string(arch_name(m.arch())));
    }
}

void ProgramBuilder::run(ArchProgram& arch) {
    prefill_ = arch.has_prefill();
    arch.alloc_buffers();
    alloc_io();
    init_cache_layout();
    setup_host_io();
    arch.build_target();
    if (g_.mtp_on_) arch.build_mtp();
    setup_diagnostics();
    if (g_.drafter_) arch.build_drafter();
    alloc_slots();
    upload_programs();
}

void ProgramBuilder::select_program(const Target& t, bool wide_mark) {
    tg_ = t;
    wide_mark_ = wide_mark;
}

}  // namespace tona::gpu
