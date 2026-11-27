#include "client/screen.h"

#include <cstdlib>

#include "client/format.h"

namespace tona::client {

bool no_color_requested() {
    const char* v = std::getenv("NO_COLOR");
    return v && *v;
}

namespace {

// Keeps a code of an SGR sequence when it is not a colour
bool keeps(int code) {
    const bool colour = (code >= 30 && code <= 49) || (code >= 90 && code <= 107);
    return !colour;
}

// "31;1" -> "1"; empty when nothing is left. 38 and 48 take arguments (5;n or 2;r;g;b): they and what follows go.
std::string filtered_params(const std::string& params) {
    std::string out;
    size_t pos = 0;
    while (pos <= params.size()) {
        const size_t semi = params.find(';', pos);
        const std::string part = params.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
        const int code = part.empty() ? 0 : to_int(part);
        if (code == 38 || code == 48) break;
        if (keeps(code)) out += (out.empty() ? "" : ";") + (part.empty() ? "0" : part);
        if (semi == std::string::npos) break;
        pos = semi + 1;
    }
    return out;
}

}  // namespace

std::string strip_color(const std::string& line) {
    std::string out;
    for (size_t i = 0; i < line.size(); i++) {
        if (line[i] != '\x1b' || i + 1 >= line.size() || line[i + 1] != '[') {
            out += line[i];
            continue;
        }
        const size_t end = line.find_first_not_of("0123456789;", i + 2);
        if (end == std::string::npos || line[end] != 'm') {  // not an SGR sequence: leave it
            out += line[i];
            continue;
        }
        const std::string params = filtered_params(line.substr(i + 2, end - i - 2));
        if (!params.empty()) out += "\x1b[" + params + "m";
        i = end;
    }
    return out;
}

std::string frame_text(const std::vector<std::string>& lines, bool color) {
    std::string frame = "\x1b[H";
    for (size_t i = 0; i < lines.size(); i++)
        frame += (color ? lines[i] : strip_color(lines[i])) + (i + 1 < lines.size() ? "\r\n" : "");
    return frame;
}

}  // namespace tona::client
