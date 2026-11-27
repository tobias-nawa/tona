#include "offer/offer.h"

namespace tona::offer {

std::string basis_word(Basis b) {
    static const char* const words[] = {"measured", "provisional", "vendor", "independent", "open"};
    return words[static_cast<size_t>(b)];
}

const LanguageVerdict* ModelCard::verdict(const std::string& language) const {
    for (const auto& v : languages)
        if (v.language == language) return &v;
    return nullptr;
}

const Rate* ModelCard::rate_at(uint32_t context_tokens) const {
    const Rate* best = nullptr;
    for (const auto& r : rates)
        if (r.context_tokens <= context_tokens && (!best || r.context_tokens >= best->context_tokens)) best = &r;
    if (!best)
        for (const auto& r : rates)
            if (!best || r.context_tokens < best->context_tokens) best = &r;
    return best;
}

namespace {

template <class T>
const T* find_by_id(const std::vector<T>& v, const std::string& id) {
    for (const auto& x : v)
        if (x.id == id) return &x;
    return nullptr;
}

}  // namespace

const Task* find_task(const std::string& id) { return find_by_id(tasks(), id); }

const HwClass* find_hw_class(const std::string& id) { return find_by_id(hw_classes(), id); }

const Offer* offer_for(const std::string& task, const std::string& hw_class) {
    for (const auto& o : offers())
        if (o.task == task && o.hw_class == hw_class) return &o;
    return nullptr;
}

const ModelCard* model_card(const std::string& id) { return find_by_id(model_cards(), id); }

}  // namespace tona::offer
