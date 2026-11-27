# Third-party code

Vendored headers:

| File | Project | Version | License | SHA-256 |
|---|---|---|---|---|
| `httplib.h` | [cpp-httplib](https://github.com/yhirose/cpp-httplib) | v0.51.0 | MIT (`LICENSE-httplib`) | `dfbaccb76432ed6d56ddd9983fd9d262b61ba6ba0958f6b00db35c802607bd35` |
| `nlohmann/json.hpp` | [nlohmann/json](https://github.com/nlohmann/json) | v3.12.0 | MIT (`LICENSE-nlohmann-json`) | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` |

Built from source, not vendored: [FTXUI](https://github.com/ArthurSonzogni/FTXUI) v7.0.3 (FetchContent in
`installer/CMakeLists.txt`), linked into `tona`; MIT (`LICENSE-ftxui`).

Taken unchanged. Check: `sha256sum engine/third_party/httplib.h engine/third_party/nlohmann/json.hpp`

Code ported or taken from [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (b11822), MIT (`LICENSE-ggml`):

| File | What |
|---|---|
| `engine/model/quant_tables.h` | lookup tables of the IQ formats (`ggml/src/ggml-common.h`), unchanged |
| `engine/model/unicode_tables.h` | Unicode category and whitespace tables, unchanged |
| `engine/model/quant.cpp` | dequantization of the GGUF block formats, ported |
| `engine/model/tokenizer.cpp` | UTF-8 decoding and the qwen35 pre-tokenizer split, ported |
| `engine/gpu/mega_mtp.cpp` (`quantize_iq4_*`) | scale search of `quantize_row_iq4_nl_impl` (`ggml/src/ggml-quants.c`): candidate scales and least-squares refit, rewritten |
