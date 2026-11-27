// Byte-level BPE tokenizer for tokenizer.ggml.model = "gpt2", rebuilt after llama.cpp b11822 (src/llama-vocab.cpp,
// src/unicode.cpp) so the token sequences are identical. The pre-tokenizer (the regex split before BPE) is chosen by
// tokenizer.ggml.pre: "qwen35" and "pixtral" (Tekken). Checked against llama-tokenize in
// tests/tokenizer_vs_llamacpp.py.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "model/gguf.h"

namespace tona {

class Tokenizer {
public:
    explicit Tokenizer(const Gguf& g);

    // Splits code points into the words BPE runs on, as lengths in code points (one function per tokenizer.ggml.pre)
    using SplitFn = std::vector<size_t> (*)(const std::vector<uint32_t>& cpts);

    // Which special tokens in the text become their ID: all (like llama.cpp with parse_special = true), only
    // user-defined ones (<think>, <tool_call>, ...; like parse_special = false) or none (foreign content in the chat:
    // an "<|im_end|>" in a file that was read stays text)
    enum class Special { all, user, none };
    std::vector<int32_t> encode(std::string_view text, Special sp) const;
    std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const {
        return encode(text, parse_special ? Special::all : Special::user);
    }

    // special: emit control tokens as text (otherwise empty)
    std::string decode(int32_t id, bool special = true) const;
    std::string decode(const std::vector<int32_t>& ids, bool special = true) const;

    int32_t n_vocab() const { return (int32_t)tokens_.size(); }
    int32_t eos() const { return eos_; }
    int32_t find(const std::string& text) const;  // -1 if there is no such token text

private:
    void bpe(const std::string& word, std::vector<int32_t>& out) const;
    void encode_text(std::string_view text, std::vector<int32_t>& out) const;

    std::vector<std::string> tokens_;
    std::vector<int32_t> types_;
    std::unordered_map<std::string, int32_t> to_id_;
    std::unordered_map<std::string, int32_t> ranks_;  // key: left + '\x01' + right
    std::vector<int32_t> special_;                    // by text length, descending
    SplitFn split_ = nullptr;
    bool ignore_merges_ = false;
    int32_t eos_ = -1;
};

}  // namespace tona
