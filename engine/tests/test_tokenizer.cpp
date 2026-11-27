// Tokenizer on a tiny synthetic vocabulary written as GGUF in the test: byte-level BPE, the qwen35 word splitter,
// special tokens, invalid UTF-8 and the errors of an unsupported vocabulary. The comparison with llama.cpp on the
// real vocabulary is tests/tokenizer_vs_llamacpp.py (needs the model).
#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "model/gguf.h"
#include "model/pretok_pixtral.h"
#include "model/tokenizer.h"
#include "tests/test_support.h"

using tona::Tokenizer;
using tona::test::GgufWriter;

namespace {

enum TokenType : int32_t { NORMAL = 1, CONTROL = 3, USER = 4 };

// GPT-2 byte -> printable code point map, written down again here on purpose (independent of tokenizer.cpp)
std::string utf8(uint32_t c) {
    std::string s;
    if (c < 0x80) return s += (char)c, s;
    if (c < 0x800) return s += (char)(0xC0 | (c >> 6)), s += (char)(0x80 | (c & 0x3F)), s;
    return s += (char)(0xE0 | (c >> 12)), s += (char)(0x80 | ((c >> 6) & 0x3F)), s += (char)(0x80 | (c & 0x3F)), s;
}

std::string mapped_byte(int b) {
    static const auto table = [] {
        std::vector<std::string> t(256);
        int n = 0;
        for (int i = 0; i < 256; i++) {
            const bool keep = (i >= 0x21 && i <= 0x7E) || (i >= 0xA1 && i <= 0xAC) || (i >= 0xAE && i <= 0xFF);
            t[i] = utf8(keep ? i : 256 + n++);
        }
        return t;
    }();
    return table[b];
}

struct Vocab {
    std::vector<std::string> tokens;
    std::vector<int32_t> types;
    std::vector<std::string> merges;
    std::string model = "gpt2", pre = "qwen35";
    bool add_bos = false;
    uint32_t eos = 0;

    int32_t add(const std::string& text, int32_t type = NORMAL) {
        tokens.push_back(text), types.push_back(type);
        return (int32_t)tokens.size() - 1;
    }
    void write(const std::filesystem::path& path) const {
        GgufWriter w;
        w.header(0, 7 + add_bos);
        w.kv_str("tokenizer.ggml.model", model);
        w.kv_str("tokenizer.ggml.pre", pre);
        w.kv_arr_str("tokenizer.ggml.tokens", tokens);
        w.kv_arr_i32("tokenizer.ggml.token_type", types);
        w.kv_arr_str("tokenizer.ggml.merges", merges);
        w.kv_u32("tokenizer.ggml.eos_token_id", eos);
        w.kv_str("general.architecture", "test");
        if (add_bos) w.kv_bool("tokenizer.ggml.add_bos_token", true);
        w.save(path);
    }
};

// 256 byte tokens (id = byte value), words built from merges, and the special tokens
struct Ids {
    int32_t hello, space_world, apostrophe_t, eos, im_start, im_end, think;
};

Vocab make_vocab(Ids& id) {
    Vocab v;
    for (int b = 0; b < 256; b++) v.add(mapped_byte(b));
    const std::string sp = mapped_byte(' ');
    const std::pair<const char*, const char*> chain[] = {{"h", "e"}, {"he", "l"}, {"hel", "l"}, {"hell", "o"}};
    std::string left;
    for (const auto& [l, r] : chain) {
        v.merges.push_back(std::string(l) + " " + r);
        left = std::string(l) + r;
        id.hello = v.add(left);
    }
    std::string w = sp;
    for (const char c : std::string("world")) {
        v.merges.push_back(w + " " + c);
        w += c;
        id.space_world = v.add(w);
    }
    v.merges.emplace_back("' t");
    id.apostrophe_t = v.add("'t");
    id.eos = v.add("<|endoftext|>", CONTROL);
    id.im_start = v.add("<|im_start|>", CONTROL);
    id.im_end = v.add("<|im_end|>", CONTROL);
    id.think = v.add("<think>", USER);
    v.eos = (uint32_t)id.eos;
    return v;
}

struct Fixture {
    tona::test::TempDir dir{"tokenizer"};
    Ids id{};
    Vocab vocab = make_vocab(id);
    std::unique_ptr<tona::Gguf> gguf;
    std::unique_ptr<Tokenizer> tok;
    Fixture() {
        vocab.write(dir.path() / "v.gguf");
        gguf = std::make_unique<tona::Gguf>((dir.path() / "v.gguf").string());
        tok = std::make_unique<Tokenizer>(*gguf);
    }
};

using Ids_t = std::vector<int32_t>;

void test_merges_and_words(const Fixture& f) {
    const auto& t = *f.tok;
    EXPECT(t.encode("hello") == Ids_t{f.id.hello});
    EXPECT((t.encode("hello world") == Ids_t{f.id.hello, f.id.space_world}));
    EXPECT(t.n_vocab() == (int32_t)f.vocab.tokens.size());
    EXPECT(t.eos() == f.id.eos);
    EXPECT(t.find("<think>") == f.id.think && t.find("no such token") == -1);
    EXPECT(t.encode("").empty());
    // Contractions are words of their own
    EXPECT((t.encode("don't") == Ids_t{'d', 'o', 'n', f.id.apostrophe_t}));
}

void test_whitespace_splits(const Fixture& f) {
    const auto& t = *f.tok;
    // Two spaces: the first stays alone, the second belongs to the next word
    const Ids_t two = t.encode("hello  world");
    EXPECT((two == Ids_t{f.id.hello, (int32_t)' ', f.id.space_world}));
    // A run of whitespace ending in a newline is one word
    EXPECT((t.encode("hello \n\nworld").size() == 1 + 3 + 5));
    EXPECT(t.encode("1234").size() == 4);  // digits are single words
}

void test_round_trip(const Fixture& f) {
    const auto& t = *f.tok;
    for (const char* s :
         {"hello world", "  indented\n\tcode();\n", "h\xC3\xA9llo \xE2\x9C\x93 \xE6\x97\xA5\xE6\x9C\xAC",
          "emoji \xF0\x9F\x98\x80!", "line1\r\nline2", "x = 'it''s' + \"q\"", "0123456789"}) {
        EXPECT(t.decode(t.encode(s)) == s);
    }
}

void test_invalid_utf8(const Fixture& f) {
    const auto& t = *f.tok;
    // Every invalid byte becomes its own U+FFFD, as in llama.cpp
    EXPECT(t.decode(t.encode("a\xFF"
                             "b")) ==
           "a\xEF\xBF\xBD"
           "b");
    EXPECT(t.decode(t.encode("\xC3")) == "\xEF\xBF\xBD");                  // truncated sequence
    EXPECT(t.decode(t.encode("\xC0\x80")) == "\xEF\xBF\xBD\xEF\xBF\xBD");  // overlong
    EXPECT(t.decode(t.encode("\xF4\x90\x80\x80")).size() == 12);           // beyond U+10FFFF
}

void test_special_tokens(const Fixture& f) {
    const auto& t = *f.tok;
    const std::string text = "<|im_start|>hello<|im_end|><think>x";
    const Ids_t all = t.encode(text, Tokenizer::Special::all);
    EXPECT(all.front() == f.id.im_start && all[1] == f.id.hello && all[2] == f.id.im_end && all[3] == f.id.think);
    EXPECT(t.encode(text, true) == all);
    const Ids_t user = t.encode(text, Tokenizer::Special::user);  // control tokens stay text, <think> is an id
    EXPECT(user.size() > all.size() && user.front() != f.id.im_start);
    EXPECT(std::find(user.begin(), user.end(), f.id.think) != user.end());
    EXPECT(t.encode(text, false) == user);
    const Ids_t none = t.encode(text, Tokenizer::Special::none);
    EXPECT(std::find(none.begin(), none.end(), f.id.think) == none.end());
    EXPECT(t.decode(none) == text && t.decode(user) == text);
    EXPECT(t.decode(all) == text);
    EXPECT(t.decode(all, false) == "hello<think>x");  // control tokens vanish, user-defined ones are text
}

void test_decode_edges(const Fixture& f) {
    const auto& t = *f.tok;
    EXPECT(t.decode(-1).empty() && t.decode(t.n_vocab()).empty() && t.decode(999999).empty());
    EXPECT(t.decode(f.id.hello) == "hello" && t.decode(f.id.space_world) == " world");
    EXPECT(t.decode(f.id.eos) == "<|endoftext|>" && t.decode(f.id.eos, false).empty());
    EXPECT(t.decode(Ids_t{}).empty());
}

void test_unsupported_vocabularies(const Fixture& f) {
    auto load = [&](const Vocab& v) {
        v.write(f.dir.path() / "bad.gguf");
        tona::Gguf g((f.dir.path() / "bad.gguf").string());
        Tokenizer t(g);
    };
    Vocab bad = f.vocab;
    bad.model = "llama";
    EXPECT(tona::test::throws([&] { load(bad); }));
    bad = f.vocab, bad.pre = "default";
    EXPECT(tona::test::throws([&] { load(bad); }));
    bad = f.vocab, bad.add_bos = true;
    EXPECT(tona::test::throws([&] { load(bad); }));
    bad = f.vocab, bad.types.pop_back();
    EXPECT(tona::test::throws([&] { load(bad); }));
    EXPECT(!tona::test::throws([&] { load(f.vocab); }));
}

// The Tekken word split ('pixtral'), on code points. Expected lengths are worked out from the regex by hand.
void test_pixtral_split() {
    using Lens = std::vector<size_t>;
    auto split = [](const std::u32string& s) { return tona::split_pixtral(std::vector<uint32_t>(s.begin(), s.end())); };
    EXPECT(split(U"Hello world") == (Lens{5, 6}));
    EXPECT(split(U"helloWorld") == (Lens{5, 5}));  // a capital starts a word (the qwen35 split keeps it as one)
    EXPECT(split(U"HELLOworld") == (Lens{10}));    // upper* then lower+
    EXPECT(split(U"ABc") == (Lens{3}) && split(U"ABC") == (Lens{3}) && split(U"ABCd") == (Lens{4}));
    EXPECT(split(U"x=1") == (Lens{1, 1, 1}));
    EXPECT(split(U"1234") == (Lens{1, 1, 1, 1}));  // digits one by one
    EXPECT(split(U"a  b") == (Lens{1, 1, 2}));
    EXPECT(split(U"a\n\nb") == (Lens{1, 2, 1}));
    EXPECT(split(U"a/\n") == (Lens{1, 2}));  // symbols take newlines and slashes along
    EXPECT(split(U"a // b") == (Lens{1, 3, 2}));
    EXPECT(split(U"don't") == (Lens{3, 2}) && split(U"I'm") == (Lens{1, 2}));  // no contraction rule
    EXPECT(split(U" Hello") == (Lens{6}) && split(U"-Hello") == (Lens{6}) && split(U"\tHello") == (Lens{6}));
    EXPECT(split(U"\n\nHello") == (Lens{2, 5}));  // a newline is no prefix
    EXPECT(split(U"  \n") == (Lens{3}));
    EXPECT(split(U"").empty());
    // Hugging Face classes: a capital after a lower-case word starts a new word (llama.cpp keeps it in the word)
    EXPECT(split(U"a\u00c9") == (Lens{1, 1}) && split(U"aB") == (Lens{1, 1}));
    EXPECT(split(U"\u00c9a") == (Lens{2}) && split(U"a\u00c9b") == (Lens{1, 2}));  // HF classes: Lu, Ll exact
}

// ignore_merges: a word that is a token of its own is emitted as that token even when no merge builds it
void test_pixtral_vocabulary(const Fixture& f) {
    Vocab v = f.vocab;
    v.pre = "pixtral";
    const int32_t xyz = v.add("xyz");  // no merge leads to it
    v.write(f.dir.path() / "pix.gguf");
    const tona::Gguf g((f.dir.path() / "pix.gguf").string());
    const Tokenizer pix(g);
    EXPECT(pix.encode("xyz") == Ids_t{xyz});
    EXPECT((pix.encode("hello world") == Ids_t{f.id.hello, f.id.space_world}));
    v.pre = "qwen35";
    v.write(f.dir.path() / "qw.gguf");
    const tona::Gguf g2((f.dir.path() / "qw.gguf").string());
    EXPECT((Tokenizer(g2).encode("xyz") == Ids_t{'x', 'y', 'z'}));
}

}  // namespace

int main() {
    return tona::test::run("test_tokenizer", [] {
        const Fixture f;
        test_merges_and_words(f);
        test_whitespace_splits(f);
        test_round_trip(f);
        test_invalid_utf8(f);
        test_special_tokens(f);
        test_decode_edges(f);
        test_unsupported_vocabularies(f);
        test_pixtral_split();
        test_pixtral_vocabulary(f);
    });
}
