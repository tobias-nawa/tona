// Number parsing for command-line options of the tools. A value that is not a number is an error, not a silent 0.
#pragma once

#include <charconv>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "backend/env.h"

namespace tona::tools {

template <class T>
T parse_number(const char* text) {
    T v{};
    const char* end = text + std::strlen(text);
    bool ok = false;
    if constexpr (std::is_same_v<T, double>) {
        ok = env::read_double(text, end, v) == end;  // macOS 15 has no from_chars for double
    } else {
        const auto r = std::from_chars(text, end, v);
        ok = r.ec == std::errc() && r.ptr == end;
    }
    if (!ok) throw std::invalid_argument(std::string("not a number: ") + text);
    return v;
}

inline int parse_int(const char* text) { return parse_number<int>(text); }
inline int64_t parse_int64(const char* text) { return parse_number<int64_t>(text); }
inline double parse_double(const char* text) { return parse_number<double>(text); }

}  // namespace tona::tools
