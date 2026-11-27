#include "server/drafter_factory.h"

#include <stdexcept>

namespace tona::server {

Drafters make_drafters(const ModelBase& model, const std::string& dflash_path, const std::string& mtp_path) {
    Drafters d;
    if (dflash_path.empty() && mtp_path.empty()) return d;
    if (model.arch() != ArchId::Qwen4Exp)
        throw std::runtime_error(std::string("no drafter classes for ") + std::string(arch_name(model.arch())));
    const Qwen4ExpModel& m = model_as<Qwen4ExpModel>(model);
    if (!dflash_path.empty()) {
        d.dflash_file = std::make_unique<Gguf>(dflash_path);
        d.dflash = std::make_unique<ref::DFlash>(*d.dflash_file, m);
    } else {
        d.mtp = std::make_unique<MtpHead>(mtp_path, m.hp());
    }
    return d;
}

}  // namespace tona::server
