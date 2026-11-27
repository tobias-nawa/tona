// Checkpoint files of the prompt cache on the SSD: format, writing, and reading with validation. The generator
// (server/prompt_cache.cpp) decides when to write and what to fetch; this file only knows the bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "model/hasher.h"

namespace tona::server {

constexpr uint32_t DISK_MAGIC = 0x43504549;  // "IEPC": value from before the rename to Tona, kept on purpose (a new one
                                             // would invalidate every disk cache)
constexpr uint32_t DISK_VERSION = 3;         // raise when the content of a checkpoint changes (3: no forward token)
constexpr uint64_t FNV_BASIS = Hasher::BASIS;

uint64_t fnv(uint64_t h, const void* p, size_t n);

struct DiskHead {
    uint32_t magic, version;
    uint64_t fingerprint, ntok;
    int64_t pos;
    int32_t commit, parity;
    uint64_t n_logits, n_state, n_kv, n_pool, n_dring;
};

// Name of the file for these tokens: <fingerprint>-<hash of the tokens>.bin in dir
std::string checkpoint_path(const std::string& dir, uint64_t fingerprint, const std::vector<int32_t>& tokens);

// Writes the checkpoint to path as a new file with mode 0600; an old file or link under that name is removed first and
// never followed. false if the write failed.
bool write_checkpoint(const std::string& path, uint64_t fingerprint, const std::vector<float>& logits,
                      const backend::HostSnap& snap);

// Whether dir may hold checkpoints: nobody but the owner may be able to write into it (otherwise others could plant
// links or replace files). A directory that is ours and was just created or holds nothing but checkpoint files is set
// to 0700; any other that is writable by group or others is refused, why says in what way.
bool secure_checkpoint_dir(const std::string& dir, bool created, std::string& why);

// Head and tokens of a file of this model: false for a foreign, outdated or truncated file. has_logits: the file holds
// the logits of its last token (a checkpoint inside a prompt has none)
bool read_checkpoint_tokens(const std::string& path, uint64_t fingerprint, std::vector<int32_t>& tokens,
                            bool& has_logits);

// What a file must hold to be accepted
struct CheckpointExpect {
    uint64_t fingerprint;
    const std::vector<int32_t>& tokens;
    size_t vocab;        // logits (a file may also hold none)
    size_t state_bytes;  // of the backend
};

// The whole file into logits and snap. It must hold exactly the expected tokens, logits and state size, and have
// exactly the size its head announces (a broken or foreign file gives false, nothing is trusted before the checks)
bool read_checkpoint(const std::string& path, const CheckpointExpect& want, std::vector<float>& logits,
                     backend::HostSnap& snap);

}  // namespace tona::server
