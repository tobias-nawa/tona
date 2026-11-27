// The edge between the views and the terminal: NO_COLOR is honoured here, once, and the frame is put together. The
// views add colour freely because everything they say is also in characters and words (█ ▒ ░, "measured").
#pragma once

#include <string>
#include <vector>

namespace tona::client {

// True when NO_COLOR is set to a non-empty value (https://no-color.org)
bool no_color_requested();

// Removes the colour sequences (SGR 30-37, 38, 39, 40-49, 90-97, 100-107) and keeps bold, dim, inverse and reset
std::string strip_color(const std::string& line);

// Cursor home, the lines joined with CR LF, no trailing newline (the last row must not scroll the screen)
std::string frame_text(const std::vector<std::string>& lines, bool color);

}  // namespace tona::client
