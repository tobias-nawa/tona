#include "update/url.h"

#include <algorithm>
#include <charconv>

namespace tona::update {

namespace {

bool plain_text(std::string_view s) {
    return std::all_of(s.begin(), s.end(), [](char c) {
        const auto u = static_cast<unsigned char>(c);
        return u > 0x20 && u < 0x7f && c != '\\' && c != '#';
    });
}

bool valid_host(std::string_view h) {
    return !h.empty() && h.size() <= 253 && std::all_of(h.begin(), h.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '-';
    });
}

}  // namespace

std::string Url::origin() const { return scheme + "://" + host + ":" + std::to_string(port); }

std::optional<Url> parse_url(std::string_view text) {
    if (!plain_text(text)) return std::nullopt;
    Url u;
    if (text.starts_with("https://")) {
        u.scheme = "https";
        u.port = 443;
        text.remove_prefix(8);
    } else if (text.starts_with("http://")) {
        u.scheme = "http";
        u.port = 80;
        text.remove_prefix(7);
    } else {
        return std::nullopt;
    }
    const size_t slash = text.find('/');
    if (slash == std::string_view::npos) return std::nullopt;
    std::string_view authority = text.substr(0, slash);
    u.target = std::string(text.substr(slash));
    if (const size_t colon = authority.find(':'); colon != std::string_view::npos) {
        const std::string_view digits = authority.substr(colon + 1);
        const auto r = std::from_chars(digits.data(), digits.data() + digits.size(), u.port);
        if (digits.empty() || r.ec != std::errc() || r.ptr != digits.data() + digits.size() || u.port < 1 ||
            u.port > 65535)
            return std::nullopt;
        authority = authority.substr(0, colon);
    }
    if (!valid_host(authority)) return std::nullopt;
    u.host = std::string(authority);
    return u;
}

}  // namespace tona::update
