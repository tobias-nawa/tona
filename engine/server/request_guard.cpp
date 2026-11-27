#include "server/request_guard.h"

#include <algorithm>
#include <cctype>

namespace tona::server {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

std::string trimmed(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t") - a + 1);
}

}  // namespace

bool is_loopback_bind(const std::string& bind_host) {
    const std::string h = lower(bind_host);
    return h == "127.0.0.1" || h == "localhost" || h == "::1" || h == "[::1]";
}

bool host_allowed(const std::string& host_header, int port) {
    const std::string h = lower(trimmed(host_header));
    // The port follows the last colon after the name; an IPv6 name is in brackets and holds colons itself
    const size_t name_end = h.starts_with('[') ? h.find(']') + 1 : h.find(':');
    if (name_end == 0) return false;  // '[' without ']'
    const std::string name = h.substr(0, name_end);
    const std::string rest = name_end == std::string::npos ? "" : h.substr(name_end);
    if (name != "127.0.0.1" && name != "localhost" && name != "[::1]") return false;
    if (rest.empty()) return port == 80;
    return rest == ":" + std::to_string(port);
}

bool origin_allowed(const std::string& origin, int port, const std::vector<std::string>& extra) {
    const std::string o = lower(trimmed(origin));
    if (o.empty()) return true;
    for (const auto& e : extra)
        if (o == lower(e)) return true;
    const std::string scheme = "http://";
    return o.starts_with(scheme) && host_allowed(o.substr(scheme.size()), port);
}

std::vector<std::string> parse_origin_list(const char* list) {
    std::vector<std::string> out;
    if (!list) return out;
    const std::string s = list;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        const std::string item = trimmed(s.substr(a, b - a));
        if (!item.empty()) out.push_back(item);
        a = b + 1;
    }
    return out;
}

}  // namespace tona::server
