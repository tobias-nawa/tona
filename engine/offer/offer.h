// What the installer (tona setup) and the client (tona) offer: tasks, the recommendation per task and hardware class,
// and the card of every model (language verdicts, rates, strengths, license). One source: data/modeldb/offer.json,
// turned into C++ by tools/gen_offer_data.py at build time, so both programs show the same numbers, each with its
// basis word and its source. Sizes on disk come from the pinned catalog (toolchain/models.toml).
//
// Every lookup returns a pointer into static data (nullptr when the id is unknown); nothing here allocates after the
// first call and nothing is written, so any thread may read.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tona::offer {

// English text of the screens, from data/modeldb/offer.json
using Text = std::string;

// Where a number or verdict comes from. The word is shown, never only a color.
enum class Basis {
    Measured,     // in Tona, a file under research/measurements/
    Provisional,  // measurement incomplete or only a user report
    Vendor,       // model card
    Independent,  // a third-party test
    None,         // nothing known (only with Level::Open)
};
// "measured", "provisional", "vendor", "independent", "open"
std::string basis_word(Basis b);

enum class Level { Strong, Weak, Open };  // Open: unknown, never a guess

struct HwClass {
    std::string id;  // e.g. "pc-16gb-vram-64gb-ram"
    Text label;
    std::string doc;        // docs/hardware/... page of the class
    bool measured = false;  // the numbers of the cards were measured on this class
};

struct Language {
    std::string id;     // "rust"
    std::string label;  // "Rust"
};

struct Task {
    std::string id;  // "coding", "knowledge", "documents", "transcription", "speech", "images"
    Text label, help;
    bool available = false;  // false: "coming", coming_note says why
    Text coming_note;
};

struct Offer {
    std::string task, hw_class;
    std::string recommended;  // model id of the catalog (installer/core/catalog.h)
    std::string alternative;  // at most one; empty: none
    Text reason;
    std::string source;
};

struct LanguageVerdict {
    std::string language;  // Language::id
    Level level = Level::Open;
    Basis basis = Basis::None;
    Text detail;  // the count behind the verdict, e.g. "8 of 22 exercises within 4 attempts"
    std::string source;
};

struct Remark {  // one line of "strong in" or "weak in"
    Text text;
    Basis basis = Basis::None;
    std::string source;
    Text brief;  // keywords for the one-line list; empty: the list shows `text`
};

struct Rate {
    uint32_t context_tokens = 0;        // prompt tokens before the measured pass
    std::optional<double> decode_tps;   // generated tokens per second
    std::optional<double> prefill_tps;  // prompt tokens per second
    Basis basis = Basis::None;
    Text note;  // what was measured (code, drafter)
    std::string source;
};

struct License {
    Text text;              // one line for the UI
    bool verified = false;  // the license text itself was read
    std::string source;
};

struct ModelCard {
    std::string id;                          // catalog id (toolchain/models.toml)
    std::string name;                        // display name: "Coder", "Ornith", "Bolt"
    std::string modeldb;                     // record in data/modeldb/models/
    uint64_t size_bytes = 0;                 // sum of the pinned files
    std::vector<LanguageVerdict> languages;  // one per entry of languages(), in that order
    std::vector<Remark> strengths, weaknesses;
    std::vector<Rate> rates;  // by context, ascending
    License license;

    const LanguageVerdict* verdict(const std::string& language) const;
    // The rate measured at the largest context not above `context_tokens`; the smallest one if all are above.
    const Rate* rate_at(uint32_t context_tokens) const;
    double size_gb() const { return static_cast<double>(size_bytes) / 1e9; }
};

// Defined in the generated offer_data.cpp
const std::vector<HwClass>& hw_classes();
const std::vector<Language>& languages();
const std::vector<Task>& tasks();
const std::vector<Offer>& offers();
const std::vector<ModelCard>& model_cards();

const Task* find_task(const std::string& id);
const HwClass* find_hw_class(const std::string& id);
// The recommendation and the alternative for a task on a hardware class; nullptr when there is none (a task that is
// "coming", or a class without measurements for it).
const Offer* offer_for(const std::string& task, const std::string& hw_class);
const ModelCard* model_card(const std::string& id);

}  // namespace tona::offer
