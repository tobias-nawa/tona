#include "server/chat.h"

#include <format>

namespace tona::server {

namespace {

void dump_string(const std::string& s, std::string& out) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            default:
                if (c < 0x20)
                    out += std::format("\\u{:04x}", c);
                else
                    out += (char)c;
        }
    }
    out += '"';
}

void dump(const json& j, std::string& out);

void dump_object(const json& j, std::string& out) {
    out += '{';
    bool first = true;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (!first) out += ", ";
        first = false;
        dump_string(it.key(), out);
        out += ": ";
        dump(it.value(), out);
    }
    out += '}';
}

void dump_array(const json& j, std::string& out) {
    out += '[';
    for (size_t i = 0; i < j.size(); i++) {
        if (i) out += ", ";
        dump(j[i], out);
    }
    out += ']';
}

void dump(const json& j, std::string& out) {
    switch (j.type()) {
        case json::value_t::object:
            dump_object(j, out);
            break;
        case json::value_t::array:
            dump_array(j, out);
            break;
        case json::value_t::string:
            dump_string(j.get<std::string>(), out);
            break;
        case json::value_t::boolean:
            out += j.get<bool>() ? "true" : "false";
            break;
        case json::value_t::null:
            out += "null";
            break;
        case json::value_t::number_float: {
            // Like Python repr(float): shortest form, whole numbers with ".0"
            std::string s = j.dump();
            if (s.find_first_of(".eE") == std::string::npos) s += ".0";
            out += s;
            break;
        }
        default:
            out += j.dump();
    }
}

}  // namespace

std::string to_pyjson(const json& j) {
    std::string s;
    dump(j, s);
    return s;
}

}  // namespace tona::server
