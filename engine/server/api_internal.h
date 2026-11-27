// Helpers shared by the Anthropic and OpenAI request parsers
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "server/api.h"

namespace tona::server {

// Optional field: missing or null gives the default (some proxies send null instead of omitting the field).
// Numbers outside the range of T are clamped, because the conversion would be undefined otherwise (found with
// scripts/fuzz.sh: "max_tokens": 1e91).
template <class T>
T opt(const json& j, const char* key, T def) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) return def;
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
        if (it->is_number()) {
            const double v = it->template get<double>();
            if (std::isnan(v)) throw std::invalid_argument(std::string(key) + ": not a number");
            const double lo = (double)std::numeric_limits<T>::lowest(), hi = (double)std::numeric_limits<T>::max();
            if (v <= lo) return std::numeric_limits<T>::lowest();
            if (v >= hi) return std::numeric_limits<T>::max();
            if constexpr (std::is_integral_v<T>)
                if (it->is_number_integer()) return it->template get<T>();  // whole and in range: skip the double
            return (T)v;
        }
    }
    return it->template get<T>();
}

// Adds a system message of the history: before the first other message a system message, later appended to the user or
// tool message before it (see api.cpp)
void add_system_message(std::vector<Message>& msgs, std::string text);

// Applies the final request rules: assistant prefill, tool_choice none/any/required/named tool
void finish_request(Request& q, const std::string& choice, const std::string& name);

// max_tokens 0 makes the request a warm-up as in the Anthropic API (empty answer, full usage); it needs a prompt
// without forced start, and no stream. A negative value is an error. Call after finish_request.
void check_max_tokens(Request& q);

// Sampling fields shared by both APIs
void read_sampling(Request& q, const json& r);

}  // namespace tona::server
