# Converts engine/model/quant_tables.h (C++ constexpr arrays) to MSL constant arrays for the Metal kernels.
# Usage: cmake -DIN=quant_tables.h -DOUT=quant_tables.metal -P tables.cmake. shaders.mm does the same at run time.
file(READ ${IN} t)
string(REGEX REPLACE "#pragma once|#include <cstdint>|namespace tona::tables {|}  // namespace[^\n]*" "" t "${t}")
string(REPLACE "inline constexpr uint64_t" "constant ulong" t "${t}")
string(REPLACE "inline constexpr uint32_t" "constant uint" t "${t}")
string(REPLACE "inline constexpr uint8_t" "constant uchar" t "${t}")
string(REPLACE "inline constexpr int8_t" "constant char" t "${t}")
file(WRITE ${OUT} "${t}")
