// The megakernel of Ornith (qwen35moe): the instantiations of k_mega with its profile and the functions that
// launch and probe them. A launch needs the kernel in the same translation unit (no -fgpu-rdc); the host code in
// gpu/mega_launch.cpp reaches these functions through the KernelSet.

#include "gpu/arch/qwen35moe/kernel_set.h"
#include "gpu/arch/qwen35moe/profile.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_dev.h"
#include "gpu/mega_kernel.h"
#include "gpu/mega_util.h"

static_assert(!TONA_MATVEC_STAGED, "row_staged needs an LDS buffer, k_mega passes none");

namespace tona::gpu {
namespace {

static_assert(kFattnSplits == FA_SPL, "the host sizes the partial results of the full attention by this");

using MegaKernel = void (*)(const MegaOp*, const DevTask*, MegaParams*, unsigned*, unsigned long long*, MegaScalars);

// One instantiation per width class and Q6_K layout variant (see Q6Layout)
template <int NB>
MegaKernel kernel_of(KernelTraits traits) {
    return traits.q6_tile ? k_mega<NB, Qwen35MoeProfile<Q6Layout::kBoth>>
                          : k_mega<NB, Qwen35MoeProfile<Q6Layout::kPlain>>;
}

MegaKernel kernel_ptr(int width_class, KernelTraits traits) {
    return width_class == 0 ? kernel_of<1>(traits) : width_class == 1 ? kernel_of<4>(traits) : kernel_of<8>(traits);
}

void init_tables(const rope::YarnTable& yarn) {
    upload_tables();
    upload_yarn(yarn);
    check_yarn(yarn, "gpu/arch/qwen35moe/kernel.hip");
}

KernelProbe probe(int width_class, KernelTraits traits) {
    const void* kernel = reinterpret_cast<const void*>(kernel_ptr(width_class, traits));
    hipFuncAttributes fa{};
    HIPCHECK(hipFuncGetAttributes(&fa, kernel));
    KernelProbe p;
    p.vgprs = fa.numRegs, p.max_threads = fa.maxThreadsPerBlock;
    p.lds_bytes = fa.sharedSizeBytes, p.scratch_bytes = fa.localSizeBytes;
    HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&p.blocks_per_cu, kernel, THREADS, 0));
    return p;
}

void launch(int width_class, KernelTraits traits, const KernelArgs& a) {
    hipLaunchKernelGGL(kernel_ptr(width_class, traits), dim3(a.blocks), dim3(THREADS), 0, 0,
                       static_cast<const MegaOp*>(a.ops), static_cast<const DevTask*>(a.tasks), a.dparams, a.bar,
                       a.prof, a.scalars);
}

}  // namespace

const KernelSet& qwen35moe_kernels() {
    static constexpr KernelSet set{init_tables, probe, launch};
    return set;
}

}  // namespace tona::gpu
