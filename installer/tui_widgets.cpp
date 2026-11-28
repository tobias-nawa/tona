#include "tui_widgets.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "ftxui/screen/string.hpp"

using namespace ftxui;

namespace tona::install::ui {

namespace {

Element styled(const std::string& s, Color c) { return use_color() ? text(s) | color(c) : text(s); }

}  // namespace

bool use_color() {
    const char* v = std::getenv("NO_COLOR");
    return !(v && *v);
}

Element plain(const std::string& s) { return text(s); }
Element title(const std::string& s) { return text(s) | bold; }
Element dim(const std::string& s) { return use_color() ? text(s) | ftxui::dim : text(s); }
Element good(const std::string& s) { return styled(s, Color::Green); }
Element warn(const std::string& s) { return styled(s, Color::Yellow); }
Element bad(const std::string& s) { return styled(s, Color::Red); }
Element blank() { return text(""); }
Element rule() { return use_color() ? separator() | ftxui::dim : separator(); }

Element pill(const std::string& s) {
    const std::string padded = " " + s + " ";
    return use_color() ? text(padded) | bold | color(Color::White) | bgcolor(Color::Blue)
                       : text(padded) | bold | inverted;
}

Element highlight(Element e, bool on) {
    if (!on || !use_color()) return e;
    return hbox({std::move(e), filler()}) | color(Color::White) | bgcolor(Color::Blue);
}

Element highlight_cell(const std::string& s, bool on) {
    if (!on || !use_color()) return plain(s);
    return text(s) | color(Color::White) | bgcolor(Color::Blue);
}

Element key_hints(const std::vector<KeyHint>& hints) {
    Elements parts;
    for (size_t i = 0; i < hints.size(); i++) {
        if (i > 0) parts.push_back(plain("  "));
        const auto& [key, label] = hints[i];
        if (!key.empty()) parts.push_back(title(key));
        if (!key.empty() && !label.empty()) parts.push_back(plain(" "));
        if (!label.empty()) parts.push_back(dim(label));
    }
    return hbox(std::move(parts));
}
std::vector<std::string> wrap_lines(const std::string& s, int width) {
    std::vector<std::string> out;
    std::string line;
    size_t from = 0;
    while (from <= s.size()) {
        const size_t sp = s.find(' ', from);
        std::string word = s.substr(from, sp == std::string::npos ? std::string::npos : sp - from);
        if (!line.empty() && string_width(line) + 1 + string_width(word) > width) out.push_back(line), line.clear();
        // a word wider than a line (a long path) is cut, so that no text runs off the screen
        while (string_width(word) > width) {
            std::string cut;
            int used = 0;
            for (const std::string& g : Utf8ToGlyphs(word)) {
                if (used + string_width(g) > width) break;
                cut += g, used += string_width(g);
            }
            if (!line.empty()) out.push_back(line), line.clear();
            out.push_back(cut);
            word = word.substr(cut.size());
        }
        line += (line.empty() ? "" : " ") + word;
        if (sp == std::string::npos) break;
        from = sp + 1;
    }
    if (!line.empty()) out.push_back(line);
    return out;
}

Elements wrap(const std::string& s, int width) {
    Elements out;
    for (const std::string& line : wrap_lines(s, width)) out.push_back(dim(line));
    return out;
}

Element spread(const std::string& left, const std::string& right, int width) {
    return hbox({title(pad(left, width - string_width(right))), dim(right)});
}

Element spread_good(const std::string& left, const std::string& right, int width) {
    return hbox({title(pad(left, width - string_width(right))), good(right)});
}

std::string fit(const std::string& s, int width) {
    if (string_width(s) <= width) return s;
    std::string out;
    int used = 0;
    for (const std::string& g : Utf8ToGlyphs(s)) {
        const int w = string_width(g);
        if (used + w > width - 1) break;
        out += g, used += w;
    }
    return out + "…";
}

std::string pad(const std::string& s, int width) {
    const std::string t = fit(s, width);
    return t + std::string(static_cast<size_t>(std::max(0, width - string_width(t))), ' ');
}

std::string box(bool on) { return on ? "[x]" : "[ ]"; }
std::string radio(bool on) { return on ? "(•)" : "( )"; }
std::string cursor(bool at) { return at ? "▸ " : "  "; }

std::string bar(const std::vector<double>& parts, int width) {
    static const char* const glyph[] = {"█", "▒", "░"};
    double total = 0;
    for (double p : parts) total += p;
    std::string out;
    int used = 0;
    for (size_t i = 0; i < parts.size() && total > 0; i++) {
        // the last part takes what is left, so the bar always has `width` characters
        const int n = i + 1 == parts.size() ? width - used : static_cast<int>(std::lround(parts[i] / total * width));
        for (int k = 0; k < n && used < width; k++, used++) out += glyph[std::min<size_t>(i, 2)];
    }
    return out;
}

Element bar_el(const std::vector<double>& parts, int width) {
    static const char* const glyph[] = {"█", "▒", "░"};
    double total = 0;
    for (double p : parts) total += p;
    Elements segs;
    int used = 0;
    for (size_t i = 0; i < parts.size() && total > 0; i++) {
        const int n = i + 1 == parts.size() ? width - used : static_cast<int>(std::lround(parts[i] / total * width));
        std::string seg;
        for (int k = 0; k < n && used < width; k++, used++) seg += glyph[std::min<size_t>(i, 2)];
        if (!seg.empty()) segs.push_back(i == 0 ? styled(seg, Color::Blue) : dim(seg));
    }
    return hbox(std::move(segs));
}

Element progress_bar(int pct, int width) {
    const int fill = std::clamp(static_cast<int>(std::lround(pct / 100.0 * width)), 0, width);
    std::string done, rest;
    for (int i = 0; i < fill; i++) done += "█";
    for (int i = fill; i < width; i++) rest += "░";
    return hbox({styled(done, Color::Blue), dim(rest)});
}

Element swatch(const std::string& glyph, bool accent) { return accent ? styled(glyph, Color::Blue) : dim(glyph); }

bool is_back(const Event& e) { return e == Event::Escape || e == Event::Backspace; }
bool is_space(const Event& e) { return e == Event::Character(' '); }
bool is_up(const Event& e) { return e == Event::ArrowUp || e == Event::Character('k'); }
bool is_down(const Event& e) { return e == Event::ArrowDown || e == Event::Character('j'); }
bool is_left(const Event& e) { return e == Event::ArrowLeft || e == Event::Character('h'); }
bool is_right(const Event& e) { return e == Event::ArrowRight || e == Event::Character('l'); }

std::string screen_text(const Screen& s) {
    std::string out;
    for (int y = 0; y < s.dimy(); y++) {
        std::string line;
        for (int x = 0; x < s.dimx(); x++) line += s.PixelAt(x, y).character;
        while (!line.empty() && line.back() == ' ') line.pop_back();
        out += line + "\n";
    }
    return out;
}

std::string render_text(const Element& e, int width, int height) {
    Screen screen = Screen::Create(Dimension::Fixed(width), Dimension::Fixed(height));
    Render(screen, e | size(WIDTH, EQUAL, width) | size(HEIGHT, EQUAL, height));
    return screen_text(screen);
}

}  // namespace tona::install::ui
