// Which model architecture a GGUF holds (general.architecture), what each one needs from a backend, and which ggml
// weight types a file uses. Design: docs/architecture/model-architectures.md section 3.1.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "model/gguf.h"

namespace tona {

enum class ArchId { Qwen4Exp, Qwen35Moe, NemotronHMoe };

// Text format of the chat template family (selects the renderer and the tool-call parser in the server)
// Qwen3Coder: the Coder's template (reasoning-effort lines in the system message); Qwen35: Ornith's, the same tags
// and tool blocks without those lines
enum class ChatFamily { Qwen3Coder, Qwen35, Nemotron };

// Where the multi-token-prediction head of a model lives
enum class MtpSource { None, SeparateGguf, EmbeddedInModel };

// Which drafter files a model can run with: the DFlash2 drafter, a separate MTP file. Declared by each model module
// (ModelBase::drafters); the server only compares it with what the user asked for. A model with neither runs one token
// per pass.
struct DrafterSupport {
    bool dflash = false;
    bool mtp_file = false;
    bool required = false;  // the model has no pass without a drafter
};

// Recurrent layer kind of an architecture
enum class RecurrentKind { DeltaNet, Mamba2 };

// Capabilities a backend must have for an architecture; backends compare this with what they implement
struct ArchNeeds {
    RecurrentKind recurrent;
    bool full_attention;  // attention over the whole context (no indexer)
    bool indexer;         // QSA: indexer scores blocks, attention reads the top-k
    bool latent_moe;      // experts work in a down-projected latent space
    bool ple;             // n-gram embedding table
    bool embedded_mtp;    // MTP layer inside the model file
};

const ArchNeeds& arch_needs(ArchId id);

// The GGUF's architecture name for id, and the id for a name; detect_arch throws and lists every known name
std::string_view arch_name(ArchId id);
ArchId detect_arch(const Gguf& g);

// One ggml type used by a file, with the first tensor (directory order) that has it, for error messages
struct WeightType {
    uint32_t type;
    std::string first_tensor;
};
// Every distinct type of the tensors a backend computes with (not the host-read embedding tables), ascending by id
std::vector<WeightType> weight_types(const Gguf& g);

}  // namespace tona
