// Typed access to TONA_* environment variables. Callers read once (function-local static) and keep the result.
// An unreadable value never becomes 0 silently: it warns on stderr and yields the fallback.
#pragma once

#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(__APPLE__)
#include <xlocale.h>
#endif

namespace tona::env {

// std::from_chars for double: the end of the number at the start of [b, e), nullptr if there is none. libc++ has the
// double overload only from macOS 26 on and the release targets macOS 15, so on Apple strtod_l in the C locale reads
// it; what from_chars does not take (leading space, '+', the hex part of "0x...") is cut off first
inline const char* read_double(const char* b, const char* e, double& v) {
#if defined(__APPLE__)
    if (b == e || std::isspace((unsigned char)*b) || *b == '+') return nullptr;
    std::string s(b, e);
    if (const auto x = s.find_first_of("xX"); x != std::string::npos) s.resize(x);  // from_chars stops there
    char* stop = nullptr;
    errno = 0;
    const double r = strtod_l(s.c_str(), &stop, LC_C_LOCALE);
    if (stop == s.c_str() || errno == ERANGE) return nullptr;
    v = r;
    return b + (stop - s.c_str());
#else
    const auto r = std::from_chars(b, e, v);
    return r.ec == std::errc() ? r.ptr : nullptr;
#endif
}

// nullptr if unset or empty
inline const char* text(const char* name) {
    const char* e = std::getenv(name);
    return e && *e ? e : nullptr;
}

// Unset or empty: fallback. Not a number (or trailing characters): warning and fallback.
inline int64_t integer(const char* name, int64_t fallback) {
    const char* e = text(name);
    if (!e) return fallback;
    int64_t v = 0;
    const char* end = e + std::strlen(e);
    const auto r = std::from_chars(e, end, v);
    if (r.ec == std::errc() && r.ptr == end) return v;
    std::fprintf(stderr, "%s: \"%s\" is not a number, using %lld\n", name, e, (long long)fallback);
    return fallback;
}

inline double number(const char* name, double fallback) {
    const char* e = text(name);
    if (!e) return fallback;
    double v = 0;
    const char* end = e + std::strlen(e);
    if (read_double(e, end, v) == end) return v;
    std::fprintf(stderr, "%s: \"%s\" is not a number, using %g\n", name, e, fallback);
    return fallback;
}

// Set, non-empty and not "0"
inline bool flag(const char* name) {
    const char* e = text(name);
    return e && std::strcmp(e, "0") != 0;
}

}  // namespace tona::env
