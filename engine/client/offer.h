// What Tona offers, as the client reads it: tasks and model cards. The numbers and verdicts come from the shared offer
// library (engine/offer/, data/modeldb/offer.json): `build_offer` copies them into these view types in the language of
// the user. Nothing is added here: a value the data does not know stays "open", never a guess.
#pragma once

#include <string>
#include <vector>

#include "offer/offer.h"

namespace tona::install {
struct Hardware;
}

namespace tona::client {

// How a number or verdict was found. Shown as a word, never as colour alone.
using offer::Basis;

struct Task {
    std::string id;     // "coding", "knowledge", ...
    std::string label;  // in the language of the user
    bool coming = false;
};

struct ModelCard {
    std::string id;                   // catalog id; empty for a model that is not built in yet
    std::string name;                 // display name ("Coder", "Ornith")
    bool coming = false;              // "Bald in Tona": announced, not built in
    std::string note;                 // coming: what is known
    std::vector<std::string> tasks;   // task ids
    std::vector<std::string> strong;  // languages it is good at (labels of the offer data)
    std::vector<std::string> weak;    // languages it is weaker at
    std::vector<std::string> open;    // languages nobody measured yet: shown as "still open", never guessed
    std::string weak_note;            // what is open ("not measured yet")
    double decode_tok_s = 0;          // writing speed at speed_context, 0 = unknown
    double decode_tok_s_hi = 0;       // > 0: decode_tok_s .. decode_tok_s_hi is an estimate for this machine (a Mac)
    uint32_t speed_context = 0;       // prompt tokens before the measured pass (the card's rate)
    Basis speed_basis = Basis::None;
    std::string source;  // file of the measurement or the model card
    double size_gb = 0;
    std::string license;
    bool license_checked = false;  // false: "licence text not checked yet"
};

struct Offer {
    std::vector<Task> tasks;
    std::vector<ModelCard> cards;
    // The cards are the models of this machine's class (a Mac): they fit by the class, so the PC rule of thumb (RAM
    // plus graphics memory) that hides larger files does not apply. A Mac streams the experts from its SSD.
    bool fits_by_class = false;

    const ModelCard* find(const std::string& id) const;
    const ModelCard* find_by_name(const std::string& name) const;
};

// The offer data as this machine sees it: on a Mac only the models of its class (the same choice as `tona setup`,
// install::match_hw_class) with the range of the Mac estimate; elsewhere every card with the measured rates
Offer build_offer();
// The same for given hardware; nullptr: every card with the measured rates
Offer build_offer(const install::Hardware* h);

bool card_strong_in(const ModelCard& c, const std::string& language);

}  // namespace tona::client
