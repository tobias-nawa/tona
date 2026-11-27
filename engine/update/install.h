// Installing a verified release next to the running one. Any failure leaves the installed versions untouched: all
// work happens in a private staging directory, and the release becomes visible only through two atomic renames.
#pragma once

#include <atomic>
#include <functional>
#include <string>

#include "update/release.h"

namespace tona::update {

struct InstallHooks {
    // Bytes downloaded so far and in total
    std::function<void(uint64_t done, uint64_t total)> progress;
    const std::atomic<bool>* cancel = nullptr;
};

struct InstallOutcome {
    bool ok = false;
    std::string error;
    bool already_installed = false;  // versions/<v> existed; only the links were switched
};

// Re-verifies the release, downloads and checks every file, installs it as versions/<v>, makes it `current` (the
// old current becomes `previous`), records its seq and prunes old versions. Never runs as root. Does not restart the
// server: see switch.h.
InstallOutcome install_release(const UpdateEnv& env, const Release& release, const InstallHooks& hooks);

}  // namespace tona::update
