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
    out.push_back("┌─" + bold(t) + top::repeat("─", std::max(0, inner - 1 - top::width(t))) + "┐");
    for (const std::string& row : body) out.push_back("│ " + top::fit(row, inner - 2) + " │");
    out.push_back("└" + top::repeat("─", inner) + "┘");
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
