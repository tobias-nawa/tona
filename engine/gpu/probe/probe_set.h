// The KernelSet of a megakernel with a test-only profile (probe_kernels.h): one instantiation of k_mega per width
// class, the probe of its resources and the launch, in the translation unit of the profile (no -fgpu-rdc).
#pragma once

#include "gpu/kernel_set.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_dev.h"
#include "gpu/mega_kernel.h"
#include "gpu/mega_util.h"

static_assert(!TONA_MATVEC_STAGED, "row_staged needs an LDS buffer, k_mega passes none");

namespace tona::gpu::probe {
namespace {

using MegaKernel = void (*)(const MegaOp*, const DevTask*, MegaParams*, unsigned*, unsigned long long*, MegaScalars);

template <KernelProfile Prof>
struct ProbeKernels {
    static MegaKernel kernel(int width_class) {
        return width_class == 0 ? k_mega<1, Prof> : width_class == 1 ? k_mega<4, Prof> : k_mega<8, Prof>;
    }
    static void init_tables(const rope::YarnTable& yarn) {
        upload_tables();
        upload_yarn(yarn);
        check_yarn(yarn, "gpu/probe");
    }
    static KernelProbe probe(int width_class, KernelTraits) {
        const void* k = reinterpret_cast<const void*>(kernel(width_class));
        hipFuncAttributes fa{};
        HIPCHECK(hipFuncGetAttributes(&fa, k));
        KernelProbe p;
        p.vgprs = fa.numRegs, p.max_threads = fa.maxThreadsPerBlock;
        p.lds_bytes = fa.sharedSizeBytes, p.scratch_bytes = fa.localSizeBytes;
        HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&p.blocks_per_cu, k, THREADS, 0));
        return p;
    }
    static void launch(int width_class, KernelTraits, const KernelArgs& a) {
        hipLaunchKernelGGL(kernel(width_class), dim3(a.blocks), dim3(THREADS), 0, 0, static_cast<const MegaOp*>(a.ops),
                           static_cast<const DevTask*>(a.tasks), a.dparams, a.bar, a.prof, a.scalars);
    }
    static constexpr KernelSet set{init_tables, probe, launch};
};

}  // namespace
}  // namespace tona::gpu::probe
