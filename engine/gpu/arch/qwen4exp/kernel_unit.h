// The functions that launch and probe the k_mega instantiations of one profile of qwen4exp, for the kernel
// translation units kernel.hip (Coder, 256 experts) and kernel_512.hip (512 experts). Each unit includes this header
// and names its profile once: the units are separate because noinline helpers with LDS variables (op_host_wait) are
// allocated in every kernel of their unit, and the Coder's kernel must not carry the other size class's variable.
#pragma once

#include "gpu/arch/qwen4exp/kernel_set.h"
#include "gpu/arch/qwen4exp/profile.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_dev.h"
#include "gpu/mega_kernel.h"
#include "gpu/mega_util.h"

static_assert(!TONA_MATVEC_STAGED, "row_staged needs an LDS buffer, k_mega passes none");

namespace tona::gpu {
namespace {

using MegaKernel = void (*)(const MegaOp*, const DevTask*, MegaParams*, unsigned*, unsigned long long*, MegaScalars);

// One instantiation per width class and Q6_K layout variant (see Q6Layout); Profile<L> is the Coder's or the
// 512-expert one
template <template <Q6Layout> class Profile, int NB>
MegaKernel kernel_of(KernelTraits traits) {
    return traits.q6_tile ? k_mega<NB, Profile<Q6Layout::kBoth>> : k_mega<NB, Profile<Q6Layout::kPlain>>;
}

template <template <Q6Layout> class Profile>
MegaKernel kernel_ptr(int width_class, KernelTraits traits) {
    return width_class == 0   ? kernel_of<Profile, 1>(traits)
           : width_class == 1 ? kernel_of<Profile, 4>(traits)
                              : kernel_of<Profile, 8>(traits);
}

template <template <Q6Layout> class Profile>
void init_tables(const rope::YarnTable& yarn) {
    upload_tables();
    upload_yarn(yarn);
    check_yarn(yarn, Profile<Q6Layout::kPlain>::unit);
}

template <template <Q6Layout> class Profile>
KernelProbe probe(int width_class, KernelTraits traits) {
    const void* kernel = reinterpret_cast<const void*>(kernel_ptr<Profile>(width_class, traits));
    hipFuncAttributes fa{};
    HIPCHECK(hipFuncGetAttributes(&fa, kernel));
    KernelProbe p;
    p.vgprs = fa.numRegs, p.max_threads = fa.maxThreadsPerBlock;
    p.lds_bytes = fa.sharedSizeBytes, p.scratch_bytes = fa.localSizeBytes;
    HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&p.blocks_per_cu, kernel, THREADS, 0));
    return p;
}

template <template <Q6Layout> class Profile>
void launch(int width_class, KernelTraits traits, const KernelArgs& a) {
    hipLaunchKernelGGL(kernel_ptr<Profile>(width_class, traits), dim3(a.blocks), dim3(THREADS), 0, 0,
                       static_cast<const MegaOp*>(a.ops), static_cast<const DevTask*>(a.tasks), a.dparams, a.bar,
                       a.prof, a.scalars);
}

template <template <Q6Layout> class Profile>
const KernelSet& kernel_set() {
    static constexpr KernelSet set{init_tables<Profile>, probe<Profile>, launch<Profile>};
    return set;
}

}  // namespace
}  // namespace tona::gpu
