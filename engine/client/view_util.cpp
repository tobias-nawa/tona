#include <algorithm>

#include "client/term.h"
#include "client/view.h"

namespace tona::client {

std::string dim(const std::string& s) { return top::paint(top::kDim, s); }
std::string bold(const std::string& s) { return top::paint(top::kBold, s); }
std::string paint_ok(const std::string& s) { return top::paint(top::kGreen, s); }
std::string paint_warn(const std::string& s) { return top::paint(top::kYellow, s); }
std::string paint_bad(const std::string& s) { return top::paint(top::kRed, s); }
std::string paint_info(const std::string& s) { return top::paint(top::kCyan, s); }
std::string paint_active(const std::string& s) { return top::paint(top::kActive, s); }

namespace {

// Removes CSI escape sequences so a selected row's blue background runs uninterrupted across all columns
std::string strip_ansi(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\x1b' && i + 1 < s.size() && s[i + 1] == '[') {
            i += 2;
            while (i < s.size() && (s[i] < '@' || s[i] > '~')) i++;
            continue;
        }
        out += s[i];
    }
    return out;
}

std::string pad_plain(const std::string& s, int w) {
    std::string out;
    int n = 0;
    for (size_t i = 0; i < s.size(); i++) {
        if (((unsigned char)s[i] & 0xc0) != 0x80) {
            if (n == w) break;
            n++;
        }
        out += s[i];
    }
    return out + std::string(static_cast<size_t>(std::max(0, w - n)), ' ');
}

}  // namespace

std::string paint_selected(const std::string& s, int w) {
    return top::paint(top::kSelected, pad_plain(strip_ansi(s), w));
}

std::string rule_line(int w) { return " " + dim(top::repeat("─", std::max(0, w - 2))); }

std::string hint_row(const KeyHints& hints) {
    std::string row = " ";
    for (const auto& [key, text] : hints) row += bold(key) + " " + dim(text) + "  ";
    return row;
}

std::string section_row(const std::string& title) { return " " + bold(title); }

Lines wrap_lines(const std::string& text, int w) {
    Lines out;
    std::string line;
    size_t pos = 0;
    while (pos <= text.size()) {
        const size_t space = text.find(' ', pos);
        const std::string word = text.substr(pos, space == std::string::npos ? std::string::npos : space - pos);
        if (!line.empty() && top::width(line) + 1 + top::width(word) > w) out.push_back(line), line.clear();
        line += (line.empty() ? "" : " ") + word;
        if (space == std::string::npos) break;
        pos = space + 1;
    }
    out.push_back(line);
    return out;
}

Lines box_lines(const std::string& title, const Lines& body, int inner) {
    Lines out;
    const std::string t = " " + title + " ";
    out.push_back(dim("┌─") + bold(t) + dim(top::repeat("─", std::max(0, inner - 1 - top::width(t))) + "┐"));
    for (const std::string& row : body) out.push_back(dim("│") + " " + top::fit(row, inner - 2) + " " + dim("│"));
    out.push_back(dim("└" + top::repeat("─", inner) + "┘"));
    return out;
}

void overlay(Lines& screen, const Lines& box, int top_row, int left) {
    for (size_t i = 0; i < box.size(); i++) {
        const size_t row = (size_t)top_row + i;
        if (row >= screen.size()) break;
        screen[row] = std::string((size_t)left, ' ') + box[i];
    }
}

}  // namespace tona::client
