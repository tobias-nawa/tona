// The pre-tokenizer of tokenizer.ggml.pre = "pixtral" (and "tekken": Mistral's Tekken vocabulary, used by Nemotron 3.5
// Bolt). Same contract as the other splitters of model/tokenizer.cpp: the length of every word in code points.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tona {

// The Tekken regex as Hugging Face tokenizers evaluates it (the model was trained with that tokenizer):
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//   | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]* | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n/]*
//   | \s*[\r\n]+ | \s+(?!\S) | \s+
// llama.cpp b11822 writes the two classes as "letter or mark that is not a-z" and "... not A-Z", which differs from
// this only for a non-ASCII upper-case letter after a lower-case letter ("a\u00c9": one word there, two here). Where
// the two differ we follow Hugging Face; the case tables are tools/gen_case_tables.py's (Unicode 16).
std::vector<size_t> split_pixtral(const std::vector<uint32_t>& cpts);

}  // namespace tona
