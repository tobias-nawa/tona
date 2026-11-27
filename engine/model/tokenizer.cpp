#include "model/tokenizer.h"

#include <algorithm>
#include <format>
#include <iterator>
#include <queue>
#include <stdexcept>
#include <variant>

#include "model/pretok_pixtral.h"
#include "model/unicode_class.h"

namespace tona {

namespace {

// llama_token_type
enum : int32_t { T_UNDEF = 0, T_NORMAL = 1, T_UNKNOWN = 2, T_CONTROL = 3, T_USER = 4, T_UNUSED = 5, T_BYTE = 6 };

using unicode::cpt_flags;
using unicode::F_LETTER;
using unicode::F_MARK;
using unicode::F_NUMBER;
using unicode::F_WS;

bool is_cont(std::string_view s, size_t i, size_t k) { return i + k < s.size() && ((uint8_t)s[i + k] & 0xC0) == 0x80; }

// The decoders return the sequence length, or 0 if the sequence is invalid or overlong.
size_t utf8_two(std::string_view s, size_t i, uint32_t& c) {
    if (!is_cont(s, i, 1)) return 0;
    c = (((uint8_t)s[i] & 0x1F) << 6) | (s[i + 1] & 0x3F);
    return c >= 0x80 ? 2 : 0;
}

size_t utf8_three(std::string_view s, size_t i, uint32_t& c) {
    if (!is_cont(s, i, 1) || !is_cont(s, i, 2)) return 0;
    c = (((uint8_t)s[i] & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
    return c >= 0x800 ? 3 : 0;
}

size_t utf8_four(std::string_view s, size_t i, uint32_t& c) {
    if (!is_cont(s, i, 1) || !is_cont(s, i, 2) || !is_cont(s, i, 3)) return 0;
    c = (((uint8_t)s[i] & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
    return (c >= 0x10000 && c <= 0x10FFFF) ? 4 : 0;
}

size_t utf8_sequence(std::string_view s, size_t i, uint32_t& c) {
    const uint8_t b = (uint8_t)s[i];
    if (!(b & 0x80)) return c = b, 1;
    if ((b & 0xE0) == 0xC0) return utf8_two(s, i, c);
    if ((b & 0xF0) == 0xE0) return utf8_three(s, i, c);
    if ((b & 0xF8) == 0xF0) return utf8_four(s, i, c);
    return 0;
}

// Same as unicode_cpt_from_utf8 in llama.cpp: every invalid byte becomes its own U+FFFD.
std::vector<uint32_t> utf8_to_cpts(std::string_view s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        uint32_t c = 0;
        size_t n = utf8_sequence(s, i, c);
        if (n == 0) c = 0xFFFD, n = 1;
        out.push_back(c);
        i += n;
    }
    return out;
}

void append_utf8(std::string& s, uint32_t c) {
    if (c < 0x80)
        s += (char)c;
    else if (c < 0x800)
        s += (char)(0xC0 | (c >> 6)), s += (char)(0x80 | (c & 0x3F));
    else if (c < 0x10000)
        s += (char)(0xE0 | (c >> 12)), s += (char)(0x80 | ((c >> 6) & 0x3F)), s += (char)(0x80 | (c & 0x3F));
    else
        s += (char)(0xF0 | (c >> 18)), s += (char)(0x80 | ((c >> 12) & 0x3F)), s += (char)(0x80 | ((c >> 6) & 0x3F)),
            s += (char)(0x80 | (c & 0x3F));
}

// GPT-2 mapping from a byte to a printable code point (unicode_byte_to_utf8_map)
struct ByteMap {
    uint32_t to_cpt[256];
    std::unordered_map<uint32_t, uint8_t> to_byte;
    ByteMap() {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            const bool keep = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            to_cpt[b] = keep ? b : 256 + n++;
            to_byte[to_cpt[b]] = (uint8_t)b;
        }
    }
};
const ByteMap& byte_map() {
    static const ByteMap m;
    return m;
}

// Word boundaries of the qwen35 pre-tokenizer regex, one-to-one from unicode_regex_split_custom_qwen35 (llama.cpp):
// (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
// | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Each rule method tries to match at pos_; on a match it emits the word and returns true. The rules run in the
// order of the alternatives, so the first one that matches wins.
class Qwen35Splitter {
public:
    explicit Qwen35Splitter(const std::vector<uint32_t>& cpts) : cpts_(cpts), end_(cpts.size()) {}

    // Word lengths in code points
    std::vector<size_t> run() {
        while (pos_ < end_) {
            if (contraction() || letters() || number() || punctuation() || whitespace()) continue;
            emit(pos_ + 1);
        }
        return std::move(lens_);
    }

private:
    static constexpr uint32_t OUT = 0xFFFFFFFF;

    uint32_t cpt(size_t p) const { return p < end_ ? cpts_[p] : OUT; }
    uint16_t flags(size_t p) const { return p < end_ ? cpt_flags(cpts_[p]) : 0; }
    // llama.cpp uses unicode_tolower; for the letters compared here that only changes ASCII
    static uint32_t lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
    static bool is_newline(uint32_t c) { return c == '\r' || c == '\n'; }

    void emit(size_t e) {
        if (e > prev_) lens_.push_back(e - prev_);
        prev_ = pos_ = e;
    }

    bool contraction() {
        if (cpt(pos_) != '\'' || pos_ + 1 >= end_) return false;
        const uint32_t n1 = lower(cpt(pos_ + 1));
        if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') return emit(pos_ + 2), true;
        if (pos_ + 2 >= end_) return false;
        const uint32_t n2 = lower(cpt(pos_ + 2));
        if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l'))
            return emit(pos_ + 3), true;
        return false;
    }

    // One optional leading character that is no letter, digit or newline, then letters and marks
    bool letters() {
        const uint32_t c = cpt(pos_);
        const uint16_t f = flags(pos_);
        if (c == '\r' || c == '\n' || (f & F_NUMBER)) return false;
        if (!(f & (F_LETTER | F_MARK)) && !(flags(pos_ + 1) & (F_LETTER | F_MARK))) return false;
        size_t p = pos_ + 1;
        while (flags(p) & (F_LETTER | F_MARK)) p++;
        emit(p);
        return true;
    }

    bool number() {
        if (!(flags(pos_) & F_NUMBER)) return false;
        emit(pos_ + 1);
        return true;
    }

    // An optional space, symbols, then newlines
    bool punctuation() {
        const uint32_t c = cpt(pos_);
        const uint16_t f = flags(pos_);
        size_t p = pos_;
        uint16_t f2 = c == ' ' ? flags(p + 1) : f;
        if ((f2 & (F_WS | F_LETTER | F_MARK | F_NUMBER)) || !f) return false;
        p += (c == ' ');
        while (!(f2 & (F_WS | F_LETTER | F_MARK | F_NUMBER)) && f2) f2 = flags(++p);
        while (is_newline(cpt(p))) p++;
        emit(p);
        return true;
    }

    // Whitespace: up to the last newline of the run, else all but the last character if a non-space follows, else all
    bool whitespace() {
        size_t n_ws = 0, last_rn = 0;
        while (flags(pos_ + n_ws) & F_WS) {
            if (is_newline(cpt(pos_ + n_ws))) last_rn = pos_ + n_ws + 1;
            n_ws++;
        }
        if (last_rn > 0) return emit(last_rn), true;
        if (n_ws > 1 && cpt(pos_ + n_ws) != OUT) return emit(pos_ + n_ws - 1), true;
        if (n_ws > 0) return emit(pos_ + n_ws), true;
        return false;
    }

    const std::vector<uint32_t>& cpts_;
    const size_t end_;
    size_t pos_ = 0, prev_ = 0;
    std::vector<size_t> lens_;
};

std::vector<size_t> split_qwen35(const std::vector<uint32_t>& cpts) { return Qwen35Splitter(cpts).run(); }

// The pre-tokenizers (tokenizer.ggml.pre) this build has; a new model family with another regex adds a row
struct PreTokenizer {
    const char* name;
    Tokenizer::SplitFn split;
    bool ignore_merges;  // a word that is a token on its own is emitted as that token, without merging (llama.cpp)
};
constexpr PreTokenizer kPreTokenizers[] = {
    {"qwen35", split_qwen35, false},
    {"pixtral", split_pixtral, true},
    {"tekken", split_pixtral, true},
};

const PreTokenizer& pre_tokenizer_for(const std::string& pre) {
    std::string known;
    for (const PreTokenizer& p : kPreTokenizers) {
        if (pre == p.name) return p;
        known += (known.empty() ? "" : ", ") + std::string(p.name);
    }
    throw std::runtime_error(std::format("Tokenizer pre-tokenizer {} is not supported (known: {})", pre, known));
}

std::vector<std::string> string_array(const Gguf& g, const std::string& key) {
    const auto* a = std::get_if<GgufArray>(&g.kv().at(key).v);
    if (!a) throw std::runtime_error("Tokenizer: " + key + " is not an array");
    std::vector<std::string> out;
    out.reserve(a->size());
    for (const auto& e : *a) out.push_back(std::get<std::string>(e.v));
    return out;
}

// BPE over one word (bytes already mapped to code points), as llm_tokenizer_bpe_session::tokenize: always merge the
// pair with the lowest rank first, the left one on a tie.
class BpeMerger {
public:
    BpeMerger(const std::string& word, const std::unordered_map<std::string, int32_t>& ranks)
        : word_(word), ranks_(ranks), queue_(cmp) {
        split_symbols();
        for (int i = 1; i < (int)sym_.size(); i++) add(i - 1, i);
    }

    void merge() {
        while (!queue_.empty()) {
            const Bigram bg = queue_.top();
            queue_.pop();
            Sym& l = sym_[bg.left];
            Sym& r = sym_[bg.right];
            // Stale: one side was merged already, or the pair is a different one by now
            if (l.n == 0 || r.n == 0 || l.n + r.n != bg.size || l.next != bg.right) continue;
            l.n += r.n;
            r.n = 0;
            l.next = r.next;
            if (r.next >= 0) sym_[r.next].prev = bg.left;
            add(l.prev, bg.left);
            add(bg.left, l.next);
        }
    }

    // Texts of the merged symbols, left to right
    std::vector<std::string> pieces() const {
        std::vector<std::string> out;
        for (int i = 0; i < (int)sym_.size(); i++)
            if (sym_[i].n != 0) out.emplace_back(text(i));
        return out;
    }

private:
    struct Sym {
        int prev, next;
        size_t off, n;
    };
    struct Bigram {
        int left, right, rank;
        size_t size;
    };
    static bool cmp(const Bigram& l, const Bigram& r) {
        return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
    }

    std::string_view text(int i) const { return std::string_view(word_).substr(sym_[i].off, sym_[i].n); }

    void split_symbols() {
        for (size_t off = 0; off < word_.size();) {
            const uint8_t b = (uint8_t)word_[off];
            size_t len = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 1;
            len = std::min(len, word_.size() - off);
            const int idx = (int)sym_.size();
            sym_.push_back({idx - 1, off + len == word_.size() ? -1 : idx + 1, off, len});
            off += len;
        }
    }

    void add(int l, int r) {
        if (l < 0 || r < 0) return;
        std::string key(text(l));
        key += '\x01';
        key += text(r);
        auto it = ranks_.find(key);
        if (it != ranks_.end()) queue_.push({l, r, it->second, sym_[l].n + sym_[r].n});
    }

    const std::string& word_;
    const std::unordered_map<std::string, int32_t>& ranks_;
    std::vector<Sym> sym_;
    std::priority_queue<Bigram, std::vector<Bigram>, bool (*)(const Bigram&, const Bigram&)> queue_;
};

// A piece of the input: text (off, len) or a finished token, as tokenizer_st_partition
struct Frag {
    int32_t id;  // -1 = text
    size_t off, len;
};

// Splits one text fragment around every occurrence of the special token text st
void split_around(std::string_view text, const Frag& f, const std::string& st, int32_t sid, std::vector<Frag>& next) {
    size_t off = f.off, len = f.len;
    while (true) {
        const size_t m = text.substr(0, off + len).find(st, off);
        if (m == std::string_view::npos) {
            if (len) next.push_back({-1, off, len});
            return;
        }
        if (m > off) next.push_back({-1, off, m - off});
        next.push_back({sid, 0, 0});
        const size_t r = m + st.size();
        len = off + len - r;
        off = r;
    }
}

}  // namespace

Tokenizer::Tokenizer(const Gguf& g) {
    const std::string model = g.get_string("tokenizer.ggml.model"), pre = g.get_string("tokenizer.ggml.pre");
    if (model != "gpt2")
        throw std::runtime_error(std::format("Tokenizer model {} is not supported (only gpt2)", model));
    const PreTokenizer& pt = pre_tokenizer_for(pre);
    split_ = pt.split;
    ignore_merges_ = pt.ignore_merges;
    if (g.has("tokenizer.ggml.add_bos_token") && std::get<bool>(g.kv().at("tokenizer.ggml.add_bos_token").v))
        throw std::runtime_error("Tokenizer: add_bos_token = true is not supported");

    tokens_ = string_array(g, "tokenizer.ggml.tokens");
    for (int64_t t : g.get_int_array("tokenizer.ggml.token_type")) types_.push_back((int32_t)t);
    if (types_.size() != tokens_.size()) throw std::runtime_error("Tokenizer: token_type does not match tokens");
    to_id_.reserve(tokens_.size());
    for (int32_t i = 0; i < (int32_t)tokens_.size(); i++) to_id_[tokens_[i]] = i;  // the last one wins, as in llama.cpp

    const auto merges = string_array(g, "tokenizer.ggml.merges");
    ranks_.reserve(merges.size());
    for (int32_t i = 0; i < (int32_t)merges.size(); i++) {
        const std::string& m = merges[i];
        const size_t sep = m.empty() ? std::string::npos : m.find(' ', 1);
        std::string key =
            sep == std::string::npos ? std::string("\x01") : m.substr(0, sep) + '\x01' + m.substr(sep + 1);
        ranks_.emplace(std::move(key), i);
    }

    for (int32_t i = 0; i < (int32_t)types_.size(); i++)
        if (types_[i] == T_CONTROL || types_[i] == T_USER || types_[i] == T_UNKNOWN) special_.push_back(i);
    std::stable_sort(special_.begin(), special_.end(),
                     [&](int32_t a, int32_t b) { return tokens_[a].size() > tokens_[b].size(); });
    eos_ = (int32_t)g.get_uint("tokenizer.ggml.eos_token_id");
}

int32_t Tokenizer::find(const std::string& text) const {
    auto it = to_id_.find(text);
    return it == to_id_.end() ? -1 : it->second;
}

void Tokenizer::bpe(const std::string& word, std::vector<int32_t>& out) const {
    if (ignore_merges_) {
        if (int32_t id = find(word); id >= 0) {
            out.push_back(id);
            return;
        }
    }
    BpeMerger merger(word, ranks_);
    merger.merge();
    for (const std::string& s : merger.pieces()) {
        if (int32_t id = find(s); id >= 0) {
            out.push_back(id);
            continue;
        }
        for (char ch : s)  // fall back to single bytes (same in llama.cpp)
            if (int32_t id = find(std::string(1, ch)); id >= 0) out.push_back(id);
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, Special sp) const {
    std::vector<Frag> frags{{-1, 0, text.size()}};
    for (int32_t sid : special_) {
        if (sp == Special::none) break;
        if (sp == Special::user && (types_[sid] == T_CONTROL || types_[sid] == T_UNKNOWN)) continue;
        const std::string& st = tokens_[sid];
        if (st.empty()) continue;
        std::vector<Frag> next;
        for (const Frag& f : frags) {
            if (f.id >= 0)
                next.push_back(f);
            else
                split_around(text, f, st, sid, next);
        }
        frags = std::move(next);
    }

    std::vector<int32_t> out;
    for (const Frag& f : frags) {
        if (f.id >= 0)
            out.push_back(f.id);
        else
            encode_text(text.substr(f.off, f.len), out);
    }
    return out;
}

void Tokenizer::encode_text(std::string_view text, std::vector<int32_t>& out) const {
    const ByteMap& bm = byte_map();
    const auto cpts = utf8_to_cpts(text);
    size_t start = 0;
    for (size_t n : split_(cpts)) {
        // Back to UTF-8 (invalid bytes are U+FFFD by now), then map byte by byte
        std::string raw;
        for (size_t i = start; i < start + n; i++) append_utf8(raw, cpts[i]);
        start += n;
        std::string word;
        for (char ch : raw) append_utf8(word, bm.to_cpt[(uint8_t)ch]);
        bpe(word, out);
    }
}

std::string Tokenizer::decode(int32_t id, bool special) const {
    if (id < 0 || id >= n_vocab()) return {};
    const int32_t t = types_[id];
    if (t == T_CONTROL || t == T_UNKNOWN) return special ? tokens_[id] : std::string();
    if (t == T_USER) return tokens_[id];
    if (t != T_NORMAL) return {};
    const ByteMap& bm = byte_map();
    std::string s;
    for (uint32_t c : utf8_to_cpts(tokens_[id])) {
        auto it = bm.to_byte.find(c);
        if (it != bm.to_byte.end())
            s += (char)it->second;
        else
            append_utf8(s, c);
    }
    return s;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool special) const {
    std::string s;
    for (int32_t id : ids) s += decode(id, special);
    return s;
}

}  // namespace tona
