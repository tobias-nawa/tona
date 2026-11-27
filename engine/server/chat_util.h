// String helpers shared by the chat renderer and the output parser
#pragma once

#include <string>

namespace tona::server {

inline bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }

inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && is_ws(s[a])) a++;
    while (b > a && is_ws(s[b - 1])) b--;
    return s.substr(a, b - a);
}

inline bool starts_with(const std::string& s, const std::string& p) { return s.starts_with(p); }
inline bool ends_with(const std::string& s, const std::string& p) { return s.size() >= p.size() && s.ends_with(p); }

}  // namespace tona::server
