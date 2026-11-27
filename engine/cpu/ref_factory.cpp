#include <stdexcept>

#include "cpu/ref.h"
#include "cpu/ref_nemotron_h.h"
#include "cpu/ref_qwen35moe.h"
#include "cpu/ref_qwen4exp.h"

namespace tona::ref {

std::unique_ptr<Oracle> make_oracle(const ModelBase& m) {
    switch (m.arch()) {
        case ArchId::Qwen4Exp:
            return std::make_unique<Qwen4ExpOracle>(model_as<Qwen4ExpModel>(m));
        case ArchId::Qwen35Moe:
            return std::make_unique<Qwen35MoeOracle>(model_as<Qwen35MoeModel>(m));
        case ArchId::NemotronHMoe:
            return std::make_unique<NemotronHOracle>(model_as<NemotronHMoeModel>(m));
    }
    throw std::logic_error("unhandled ArchId");
}

}  // namespace tona::ref
