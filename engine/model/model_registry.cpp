#include "model/model_registry.h"

#include <stdexcept>

#include "model/nemotron_h_moe.h"
#include "model/qwen35moe.h"
#include "model/qwen4exp.h"

namespace tona {

std::unique_ptr<ModelBase> load_model(const std::string& path) {
    // The header is parsed twice (here and by the module): a cold path, and it keeps the modules independent of the
    // registry
    const ArchId id = detect_arch(Gguf(path));
    switch (id) {
        case ArchId::Qwen4Exp:
            return std::make_unique<Qwen4ExpModel>(path);
        case ArchId::Qwen35Moe:
            return std::make_unique<Qwen35MoeModel>(path);
        case ArchId::NemotronHMoe:
            return std::make_unique<NemotronHMoeModel>(path);
    }
    throw std::logic_error("unhandled ArchId");
}

}  // namespace tona
