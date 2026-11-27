// Stand-in for cpu/experts.cpp on CPUs without AVX-512 (e.g. Apple silicon): experts.cpp is written in Zen 5
// intrinsics and does not compile there. Every entry point throws, so a caller fails loudly instead of computing
// wrong experts. The CPU expert path of a non-x86 backend is decided with that backend (research/21-backends.md).
#include <stdexcept>
#include <string>

#include "cpu/experts.h"

namespace tona::cpu {

namespace {
[[noreturn]] void unsupported(const char* what) {
    throw std::runtime_error(std::string("cpu/experts: ") + what +
                             " needs the AVX-512 build (cpu/experts.cpp); not available on this CPU");
}
}  // namespace

bool expert_type_supported(uint32_t) { unsupported("expert_type_supported"); }
int64_t row_bytes(uint32_t, int64_t) { unsupported("row_bytes"); }
int expert_threads() { unsupported("expert_threads"); }
void quantize_q8(const float*, int64_t, int8_t*, Q8Meta*) { unsupported("quantize_q8"); }
void matvec_q8(const MatvecArgs&) { unsupported("matvec_q8"); }
void experts_ffn(const FfnArgs&, float*, ExpertScratch&) { unsupported("experts_ffn"); }
void experts_ffn_fix(const FfnArgs&, int64_t*, ExpertScratch&) { unsupported("experts_ffn_fix"); }

}  // namespace tona::cpu
