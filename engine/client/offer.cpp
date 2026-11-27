#include "client/offer.h"

#include <algorithm>

#include "client/text.h"
#include "core/flow.h"
#include "core/hw.h"
#include "core/module.h"

namespace tona::client {

const ModelCard* Offer::find(const std::string& id) const {
    for (const ModelCard& c : cards)
        if (!c.id.empty() && c.id == id) return &c;
    return nullptr;
}

const ModelCard* Offer::find_by_name(const std::string& name) const {
    for (const ModelCard& c : cards)
        if (c.name == name) return &c;
    return nullptr;
}

namespace {

constexpr uint32_t kSpeedContext = 16384;

std::string language_label(const std::string& id) {
    for (const offer::Language& l : offer::languages())
        if (l.id == id) return l.label;
    return id;
}

// The tasks a model is recommended for (as recommendation or alternative on any hardware class)
std::vector<std::string> tasks_of(const std::string& model_id) {
    std::vector<std::string> out;
    for (const offer::Offer& o : offer::offers())
        if ((o.recommended == model_id || o.alternative == model_id) &&
            std::find(out.begin(), out.end(), o.task) == out.end())
            out.push_back(o.task);
    return out;
}

ModelCard card_from(const offer::ModelCard& m) {
    ModelCard c;
    c.id = m.id, c.name = m.name, c.size_gb = m.size_gb(), c.tasks = tasks_of(m.id);
    bool any_open = false;
    for (const offer::LanguageVerdict& v : m.languages) {
        if (v.level == offer::Level::Strong) c.strong.push_back(language_label(v.language));
        if (v.level == offer::Level::Weak) c.weak.push_back(language_label(v.language));
        if (v.level == offer::Level::Open) c.open.push_back(language_label(v.language)), any_open = true;
    }
    if (any_open) c.weak_note = tr("card_open");
    if (!m.rates.empty()) {
        // The same point as the installer's list (a coding agent works with about 16K in the window), so both
        // programs show one number per model, and the context is named next to it.
        const offer::Rate& r = *m.rate_at(kSpeedContext);
        c.decode_tok_s = r.decode_tps.value_or(0), c.speed_basis = r.basis, c.source = r.source;
        c.speed_context = r.context_tokens;
    }
    c.license = m.license.text, c.license_checked = m.license.verified;
    return c;
}

// The models the class of this Mac offers, with the range of the Mac estimate instead of the PC rates
void apply_mac_class(const install::Hardware& h, Offer& o) {
    const install::HwMatch match = install::match_hw_class(h);
    const install::Module* module = install::find_module(h);
    const install::Settings settings = module ? module->recommend(h) : install::Settings{};
    std::vector<std::string> allowed;
    for (const offer::Task& t : offer::tasks())
        if (const offer::Offer* of = offer::offer_for(t.id, match.cls))
            for (const std::string& id : {of->recommended, of->alternative})
                if (!id.empty()) allowed.push_back(id);
    std::erase_if(o.cards, [&](const ModelCard& c) {
        return !c.coming && std::find(allowed.begin(), allowed.end(), c.id) == allowed.end();
    });
    o.fits_by_class = true;
    for (ModelCard& c : o.cards) {
        if (c.coming) continue;
        const install::Estimate e = module ? module->estimate(h, settings, c.id) : install::Estimate{};
        c.decode_tok_s = e.decode.lo, c.decode_tok_s_hi = e.decode.hi;
        c.speed_context = e.context_tokens, c.speed_basis = e.decode.hi > 0 ? Basis::Provisional : Basis::None;
        if (e.decode.hi <= 0) c.decode_tok_s = 0;
    }
}

}  // namespace

Offer build_offer(const install::Hardware* h) {
    Offer o;
    for (const offer::Task& t : offer::tasks()) {
        o.tasks.push_back({t.id, t.label, !t.available});
        if (t.available) continue;
        // A task without a model is announced, never hidden: one "coming" row with the note of the data
        ModelCard c;
        c.name = t.label, c.coming = true, c.note = t.coming_note, c.tasks = {t.id};
        o.cards.push_back(c);
    }
    for (const offer::ModelCard& m : offer::model_cards()) o.cards.push_back(card_from(m));
    if (h && h->os == "macos") apply_mac_class(*h, o);
    return o;
}

Offer build_offer() {
#ifdef __APPLE__
    // once per program: the hardware does not change while `tona` runs (the folder only sets where free space is read)
    static const install::Hardware h = install::detect(".");
    return build_offer(&h);
#else
    return build_offer(nullptr);
#endif
}

bool card_strong_in(const ModelCard& c, const std::string& language) {
    return std::find(c.strong.begin(), c.strong.end(), language) != c.strong.end();
}

}  // namespace tona::client
