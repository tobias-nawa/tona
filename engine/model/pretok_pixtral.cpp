#include "model/pretok_pixtral.h"

#include <utility>

#include "model/unicode_class.h"

namespace tona {

namespace {

using unicode::cpt_flags;
using unicode::F_LETTER;
using unicode::F_MARK;
using unicode::F_NUMBER;
using unicode::F_WS;

// Each rule method tries to match at pos_ with the backtracking of the regex alternative it stands for (greedy
// quantifiers give back one element at a time); on a match it emits the word and returns true. The rules run in the
// order of the alternatives, so the first one that matches wins.
class PixtralSplitter {
public:
    explicit PixtralSplitter(const std::vector<uint32_t>& cpts) : cpts_(cpts), end_(cpts.size()) {}

    // Word lengths in code points
    std::vector<size_t> run() {
        while (pos_ < end_) {
            if (cased_word() || number() || punctuation() || whitespace()) continue;
            emit(pos_ + 1);
        }
        return std::move(lens_);
    }

private:
    static constexpr uint32_t OUT = 0xFFFFFFFF;

    uint32_t cpt(size_t p) const { return p < end_ ? cpts_[p] : OUT; }
    uint16_t flags(size_t p) const { return p < end_ ? cpt_flags(cpts_[p]) : 0; }
    static bool is_newline(uint32_t c) { return c == '\r' || c == '\n'; }

    // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and [\p{Ll}\p{Lm}\p{Lo}\p{M}]: a letter or mark that is not a lower-case /
    // not an upper-case letter (see the header)
    bool upper(size_t p) const { return (flags(p) & (F_LETTER | F_MARK)) && !unicode::is_lower(cpt(p)); }
    bool lower(size_t p) const { return (flags(p) & (F_LETTER | F_MARK)) && !unicode::is_upper_title(cpt(p)); }
    // [^\r\n\p{L}\p{N}]: the optional first character of a word
    bool prefix(size_t p) const { return p < end_ && !is_newline(cpt(p)) && !(flags(p) & (F_LETTER | F_NUMBER)); }
    // [^\s\p{L}\p{N}]
    bool other(size_t p) const { return p < end_ && !(flags(p) & (F_WS | F_LETTER | F_NUMBER)); }

    void emit(size_t e) {
        if (e > prev_) lens_.push_back(e - prev_);
        prev_ = pos_ = e;
    }

    size_t run_of(size_t p, bool (PixtralSplitter::*is)(size_t) const) const {
        size_t n = 0;
        while ((this->*is)(p + n)) n++;
        return n;
    }

    // upper* lower+ from j: the longest upper run that a lower letter follows (zero or more uppers give back one by
    // one); the match ends after the lower run. 0 if there is none.
    size_t upper_star_lower_plus(size_t j) const {
        for (size_t k = run_of(j, &PixtralSplitter::upper) + 1; k-- > 0;)
            if (lower(j + k)) return j + k + run_of(j + k, &PixtralSplitter::lower);
        return 0;
    }

    // upper+ lower*
    size_t upper_plus_lower_star(size_t j) const {
        const size_t u = run_of(j, &PixtralSplitter::upper);
        return u == 0 ? 0 : j + u + run_of(j + u, &PixtralSplitter::lower);
    }

    // The first two alternatives, each with the optional prefix character taken first, then left out
    bool cased_word() {
        for (auto form : {&PixtralSplitter::upper_star_lower_plus, &PixtralSplitter::upper_plus_lower_star}) {
            if (prefix(pos_))
                if (const size_t e = (this->*form)(pos_ + 1)) return emit(e), true;
            if (const size_t e = (this->*form)(pos_)) return emit(e), true;
        }
        return false;
    }

    bool number() {
        if (!(flags(pos_) & F_NUMBER)) return false;
        emit(pos_ + 1);
        return true;
    }

    // An optional space, symbols, then newlines and slashes
    bool punctuation() {
        size_t p = pos_;
        if (cpt(p) == ' ' && other(p + 1))
            p++;
        else if (!other(p))
            return false;
        while (other(p)) p++;
        while (is_newline(cpt(p)) || cpt(p) == '/') p++;
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

}  // namespace

std::vector<size_t> split_pixtral(const std::vector<uint32_t>& cpts) { return PixtralSplitter(cpts).run(); }

}  // namespace tona
