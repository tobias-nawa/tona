// Minimal absolute-URL splitting for the fetcher (http and https only)
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace tona::update {

struct Url {
    std::string scheme;  // "http" or "https"
    std::string host;
    int port = 0;
    std::string target;  // path and query, starts with '/'

    std::string origin() const;  // scheme://host:port
};

// nullopt for anything but a plain absolute http(s) URL: no user info, no fragment, no control characters,
// no backslashes, no empty host
std::optional<Url> parse_url(std::string_view text);

}  // namespace tona::update
