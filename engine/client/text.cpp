#include "client/text.h"

#include <cmath>
#include <format>

namespace tona::client {

namespace {

struct Entry {
    const char* key;
    const char* en;
};

// One table for every sentence the client shows: the views never write sentences themselves
constexpr Entry kTexts[] = {
#include "client/texts_common.inc"
#include "client/texts_dialogs.inc"
#include "client/texts_models.inc"
#include "client/texts_pages.inc"
};

const Entry* find(std::string_view key) {
    for (const Entry& e : kTexts)
        if (key == e.key) return &e;
    return nullptr;
}

std::string group(const std::string& digits, char sep) {
    std::string out;
    for (size_t i = 0; i < digits.size(); i++) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += sep;
        out += digits[i];
    }
    return out;
}

}  // namespace

std::string tr(std::string_view key) {
    const Entry* e = find(key);
    return e ? e->en : std::string(key);
}

std::string fill(std::string text, const std::string& a, const std::string& b, const std::string& c) {
    for (const std::string* arg : {&a, &b, &c}) {
        const size_t at = text.find("{}");
        if (at == std::string::npos) break;
        text.replace(at, 2, *arg);
    }
    return text;
}

size_t text_count() { return sizeof kTexts / sizeof kTexts[0]; }
const char* text_key(size_t i) { return kTexts[i].key; }
bool text_complete(size_t i) { return *kTexts[i].en; }

std::string fnum(double v) {
    const std::string digits = std::format("{:.0f}", std::fabs(v));
    return (v < 0 && digits != "0" ? "-" : "") + group(digits, ',');
}

std::string fdec(double v, int places) { return std::format("{:.{}f}", v, places); }

std::string fctx(uint32_t tokens) {
    return std::format("{:.0f}K", tokens % 1000 == 0 ? tokens / 1000.0 : tokens / 1024.0);
}

}  // namespace tona::client
