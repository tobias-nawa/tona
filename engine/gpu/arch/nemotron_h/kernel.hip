// The megakernel of Bolt (nemotron_h_moe): the instantiations of k_mega with its profile and the functions that launch
// and probe them. A launch needs the kernel in the same translation unit (no -fgpu-rdc); the host code in
// gpu/mega_launch.cpp reaches these functions through the KernelSet.

#include "gpu/arch/nemotron_h/kernel_set.h"
#include "gpu/arch/nemotron_h/profile.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_dev.h"
#include "gpu/mega_kernel.h"
#include "gpu/mega_util.h"

static_assert(!TONA_MATVEC_STAGED, "row_staged needs an LDS buffer, k_mega passes none");

namespace tona::gpu {
namespace {

static_assert(kBoltFattnSplits == FA_SPL, "the host sizes the partial results of the full attention by this");

using MegaKernel = void (*)(const MegaOp*, const DevTask*, MegaParams*, unsigned*, unsigned long long*, MegaScalars);

// One instantiation per width class; no Q6_K in the file, so one layout
MegaKernel kernel_ptr(int width_class) {
    return width_class == 0   ? k_mega<1, NemotronHProfile>
           : width_class == 1 ? k_mega<4, NemotronHProfile>
                              : k_mega<8, NemotronHProfile>;
}

// No RoPE in the model, so no YaRN table
void init_tables(const rope::YarnTable&) { upload_tables(); }

KernelProbe probe(int width_class, KernelTraits) {
    const void* kernel = reinterpret_cast<const void*>(kernel_ptr(width_class));
    hipFuncAttributes fa{};
    HIPCHECK(hipFuncGetAttributes(&fa, kernel));
    KernelProbe p;
    p.vgprs = fa.numRegs, p.max_threads = fa.maxThreadsPerBlock;
    p.lds_bytes = fa.sharedSizeBytes, p.scratch_bytes = fa.localSizeBytes;
    HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&p.blocks_per_cu, kernel, THREADS, 0));
    return p;
}

void launch(int width_class, KernelTraits, const KernelArgs& a) {
    hipLaunchKernelGGL(kernel_ptr(width_class), dim3(a.blocks), dim3(THREADS), 0, 0, static_cast<const MegaOp*>(a.ops),
                       static_cast<const DevTask*>(a.tasks), a.dparams, a.bar, a.prof, a.scalars);
}

}  // namespace

const KernelSet& nemotron_h_kernels() {
    static constexpr KernelSet set{init_tables, probe, launch};
    return set;
}

}  // namespace tona::gpu
