// The views: state in, lines of text out. Colour is only added as ANSI sequences; the characters and words carry the
// meaning on their own, and screen.h strips the colour when NO_COLOR is set.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "client/ui.h"

namespace tona::client {

using Lines = std::vector<std::string>;

constexpr int kMinWidth = 80, kMinHeight = 24;

// The whole screen: exactly s.height lines, each exactly s.width columns wide (padded or cut). Too small a terminal
// gives the "terminal too small" page.
Lines render(const UiState& s);

// Pages (view_*.cpp); each returns the rows of the content area
Lines view_start(const UiState& s, int rows);
Lines view_server_down(const UiState& s, int rows);
Lines view_empty(const UiState& s, int rows);
Lines view_models(const UiState& s, int rows);
Lines view_agents(const UiState& s, int rows);
Lines view_settings(const UiState& s, int rows);
Lines view_help(const UiState& s, int rows);
Lines view_too_small(const UiState& s);

// Dialog boxes drawn over the content (view_modal.cpp)
Lines dialog_confirm(const UiState& s);
Lines dialog_close(const UiState& s);

// Pieces shared by the pages (view_util.cpp)
using KeyHints = std::vector<std::pair<std::string, std::string>>;  // key, text (already translated)
std::string hint_row(const KeyHints& hints);
std::string section_row(const std::string& title);
std::string dim(const std::string& s);
std::string bold(const std::string& s);
std::string paint_ok(const std::string& s);
std::string paint_warn(const std::string& s);
std::string paint_bad(const std::string& s);
std::string paint_info(const std::string& s);
std::string paint_active(const std::string& s);
std::string paint_selected(const std::string& s, int w);
std::string rule_line(int w);
// `text` broken at spaces into lines of at most `w` columns (no escape sequences in `text`); at least one line
Lines wrap_lines(const std::string& text, int w);
// A box of the dialog kind around `body`, `inner` columns wide inside, as lines
Lines box_lines(const std::string& title, const Lines& body, int inner);
// Puts `box` over the screen rows from `top`, indented by `left`
void overlay(Lines& screen, const Lines& box, int top, int left);

}  // namespace tona::client
