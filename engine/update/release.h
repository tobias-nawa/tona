// Finding and verifying a release: fetch the signed manifest, check signature and schema, compare seq.
#pragma once

#include <ctime>
#include <string>
#include <vector>

#include "update/config.h"
#include "update/fetch.h"
#include "update/layout.h"
#include "update/manifest.h"
#include "update/minisig.h"

namespace tona::update {

// Everything an update run depends on; tests replace every part
struct UpdateEnv {
    Layout layout;
    UpdateBase base;
    FetchConfig fetch;
    std::vector<TrustedKey> keys;
    std::string platform = std::string(kPlatform);
    std::time_t now = 0;  // 0: the system clock

    std::time_t clock() const { return now != 0 ? now : std::time(nullptr); }
};

// A manifest whose signature and schema checked out, with the exact bytes that were signed
struct Release {
    Manifest manifest;
    std::string manifest_text;
    std::string signature_text;
};

struct VerifyOutcome {
    bool ok = false;
    std::string error;
    Release release;
};

// Signature first; only then is the manifest read. `last_seq`: highest seq accepted before.
VerifyOutcome verify_release(const UpdateEnv& env, std::string manifest_text, std::string signature_text,
                             uint64_t last_seq);

struct CheckOutcome {
    enum class Kind { UpToDate, Available, Failed };
    Kind kind = Kind::Failed;
    std::string error;
    Release release;  // when Available
};

// Downloads the latest manifest and decides if it is newer than the installed state
CheckOutcome check_for_update(const UpdateEnv& env);

}  // namespace tona::update
