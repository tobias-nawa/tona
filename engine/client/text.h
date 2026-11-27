// Texts of the client, English only, and the number format. Commands, keys and file names are the same everywhere.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace tona::client {

// The sentence for a key. A key without an entry returns the key itself.
std::string tr(std::string_view key);
// Text with {} filled in order, e.g. fmt(tr("x"), 3). Arguments are strings: the caller formats numbers with fnum.
std::string fill(std::string text, const std::string& a = "", const std::string& b = "", const std::string& c = "");

// The table, for the test that no entry is empty
size_t text_count();
const char* text_key(size_t i);
bool text_complete(size_t i);

// Thousands separator and decimal point: 2,900 and 14.1
std::string fnum(double v);
std::string fdec(double v, int places = 1);
// "16K": a context size in tokens, as the cards of the offer data name it (16000 is 16K, 32768 is 32K)
std::string fctx(uint32_t tokens);

}  // namespace tona::client
