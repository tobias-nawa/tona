// Unicode class of one code point as the pre-tokenizers need it (flags of llama.cpp's unicode_cpt_flags plus
// whitespace), shared by the splitters of model/tokenizer.cpp and model/pretok_pixtral.cpp.
#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "model/unicode_case_tables.h"
#include "model/unicode_tables.h"

namespace tona::unicode {

enum : uint16_t { F_NUMBER = 0x02, F_LETTER = 0x04, F_MARK = 0x10, F_WS = 0x100 };

inline uint16_t cpt_flags(uint32_t c) {
    using namespace tables;
    if (c >= 0x110000) return 1;
    const uint32_t* end = unicode_range_start + std::size(unicode_range_start);
    size_t i = std::upper_bound(unicode_range_start, end, c) - unicode_range_start - 1;
    uint16_t f = unicode_range_flags[i];
    if (std::binary_search(unicode_whitespace, unicode_whitespace + std::size(unicode_whitespace), c)) f |= F_WS;
    return f;
}

// \p{Lu}+\p{Lt} and \p{Ll} (the Tekken regex tells them apart)
template <size_t N>
bool in_ranges(const tables::CaseRange (&r)[N], uint32_t c) {
    const auto* it = std::upper_bound(r, r + N, c, [](uint32_t v, const tables::CaseRange& x) { return v < x.first; });
    return it != r && c <= (it - 1)->last;
}
inline bool is_upper_title(uint32_t c) { return in_ranges(tables::unicode_upper_title, c); }
inline bool is_lower(uint32_t c) { return in_ranges(tables::unicode_lower, c); }

}  // namespace tona::unicode
