// Small building blocks of the terminal pages: lines in a few styles, padding to columns, bars made of characters.
// Nothing here knows a page. Colors are only a second signal: every state also has a word or a character, and NO_COLOR
// switches the colors off.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/screen.hpp"

namespace tona::install::ui {

constexpr int kWidth = 80, kHeight = 24;
// Rows of the content area: the frame takes title, steps, two rules and the key line
constexpr int kBody = 19;

bool use_color();  // false when NO_COLOR is set to anything non-empty (no-color.org)

using KeyHint = std::pair<std::string, std::string>;  // key, label (empty key renders label dim alone)

// Lines in the styles of the pages
ftxui::Element plain(const std::string& s);
ftxui::Element title(const std::string& s);
ftxui::Element dim(const std::string& s);
ftxui::Element good(const std::string& s);  // green: ok, measured
ftxui::Element warn(const std::string& s);  // yellow: look at this
ftxui::Element bad(const std::string& s);   // red: does not work
ftxui::Element blank();
ftxui::Element rule();                                         // dimmed horizontal separator
ftxui::Element pill(const std::string& s);                     // active breadcrumb pill (blue bg, white bold)
ftxui::Element highlight(ftxui::Element e, bool on);           // full-width blue selection bar when `on`
ftxui::Element highlight_cell(const std::string& s, bool on);  // inline blue highlight for a grid cell
ftxui::Element key_hints(const std::vector<KeyHint>& hints);   // bold key + dim description
// A text wrapped at spaces into lines of at most `width` glyphs
std::vector<std::string> wrap_lines(const std::string& s, int width = 79);
// The same as dim lines of at most `width` glyphs (the paragraph of FTXUI loses its spaces here)
ftxui::Elements wrap(const std::string& s, int width = 79);
// `left` and `right` on one line, `right` flush to the edge
ftxui::Element spread(const std::string& left, const std::string& right, int width = kWidth);
ftxui::Element spread_good(const std::string& left, const std::string& right, int width = kWidth);

// Cut to `width` glyphs with an ellipsis; pad to `width` glyphs with spaces
std::string fit(const std::string& s, int width);
std::string pad(const std::string& s, int width);

// "[x]" / "[ ]", "(•)" / "( )", "▸ " / "  "
std::string box(bool on);
std::string radio(bool on);
std::string cursor(bool at);

// A bar of `width` characters split into parts by their share of the total: the glyphs say which part is which, so it
// reads without color (█ first, ▒ second, ░ third)
std::string bar(const std::vector<double>& parts, int width);
ftxui::Element bar_el(const std::vector<double>& parts, int width);
ftxui::Element progress_bar(int pct, int width);
ftxui::Element swatch(const std::string& glyph, bool accent);

// Keys that mean the same on every page (the design of the terminal programs): Esc or Backspace is "back", the space
// bar switches, j and k move like the arrows
bool is_back(const ftxui::Event& e);
bool is_space(const ftxui::Event& e);
bool is_up(const ftxui::Event& e);
bool is_down(const ftxui::Event& e);
bool is_left(const ftxui::Event& e);
bool is_right(const ftxui::Event& e);

// The cells of a rendered screen as plain text, trailing blanks cut: the text dump of a page
std::string screen_text(const ftxui::Screen& s);
// Renders an element into an 80 x 24 screen and returns its text
std::string render_text(const ftxui::Element& e, int width = kWidth, int height = kHeight);

}  // namespace tona::install::ui
