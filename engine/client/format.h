// Drawing helpers of the client views: bars and the sparkline in characters that carry the value without colour
// (NO_COLOR is honoured once, at the output edge: client/screen.h).
#pragma once

#include <string>
#include <vector>

namespace tona::client {

// A whole number from text; `fallback` when the text is empty or not a number (no exceptions, no silent zero)
int to_int(const std::string& text, int fallback = 0);
// A decimal number from text, the same way
double to_double(const std::string& text, double fallback = 0);

// "██████░░░░": filled part █, rest ░, width w. A fraction outside 0..1 is cut.
std::string meter(double fraction, int w);
// Two parts of one whole: a as █, b as ▒, the rest ░ ("Tona 35, others 14, free 15" of 64 GB)
std::string meter2(double a, double b, int w);
// One row of eighth blocks, newest value right; values are averaged into w columns; `top` is the full height
std::string sparkline(const std::vector<double>& values, int w, double top);
// Seconds as "42 s", "8 min", "2:05 h"
std::string age(double seconds);
// Clock time as HH:MM
std::string clock_hm(long long epoch_seconds);
// The project of a request for a row: its name, or a dash when the request carried none
std::string project_label(const std::string& project);
// Text cut with "…" to w columns, padded with spaces
std::string cell(const std::string& s, int w);

}  // namespace tona::client
