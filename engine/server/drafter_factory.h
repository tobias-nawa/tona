// Builds the drafter of the model's architecture from the command line's files. The drafter classes are written for one
// architecture's model type; the downcast to it lives in drafter_factory.cpp (the architecture mapping), not in the
// startup code.
#pragma once

#include <memory>
#include <string>

#include "cpu/dflash.h"
#include "model/model_base.h"
#include "model/qwen4exp.h"

namespace tona::server {

struct Drafters {
    std::unique_ptr<Gguf> dflash_file;
    std::unique_ptr<ref::DFlash> dflash;
    std::unique_ptr<MtpHead> mtp;
};

// Empty paths build nothing. Throws for an architecture without drafter classes (the options check of
// drafter_options.h has refused the combination before).
Drafters make_drafters(const ModelBase& model, const std::string& dflash_path, const std::string& mtp_path);

}  // namespace tona::server
