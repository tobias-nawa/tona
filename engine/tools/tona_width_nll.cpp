// Quality of a lossy pass variant through the Backend interface (docs/backends/README.md#checks, item 2): texts
// teacher-forced through two backends in lockstep. A runs passes of 1 token (the exact path); B is created after
// setting the given environment variable (e.g. TONA_METAL_MMA=1; A runs with the default) and runs passes of W tokens.
// Per predicted token one line in the format of TONA_MULTI_DUMP, so scripts/nll-compare.py applies the fixed rule
// unchanged:
//   text position nll_A nll_B KL(A||B) top1_equal
// Usage: tona-width-nll MODEL.gguf --env NAME=VALUE [-n TOKENS] [-w WIDTH] [-p PREFILL] [--ctx N] TEXT...
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/builtin.h"
#include "backend/registry.h"
#include "model/model_registry.h"
#include "model/qwen4exp.h"
#include "model/tokenizer.h"

namespace {

struct Options {
    std::string model, env;
    std::vector<std::string> texts;
    int tokens = 10000, width = 4, prefill = 8;
    int64_t ctx = 10240;
};

Options parse(int argc, char** argv) {
    if (argc < 2)
        throw std::invalid_argument("usage: tona-width-nll MODEL.gguf --env NAME=VALUE [-n N] [-w W] TEXT...");
    Options o;
    o.model = argv[1];
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        if (a == "--env" && has_value)
            o.env = argv[++i];
        else if (a == "-n" && has_value)
            o.tokens = std::stoi(argv[++i]);
        else if (a == "-w" && has_value)
            o.width = std::stoi(argv[++i]);
        else if (a == "-p" && has_value)
            o.prefill = std::stoi(argv[++i]);
        else if (a == "--ctx" && has_value)
            o.ctx = std::stoll(argv[++i]);
        else
            o.texts.push_back(a);
    }
    return o;
}

void check(const Options& o) {
    if (o.env.find('=') == std::string::npos || o.texts.empty() || o.width < 2)
        throw std::invalid_argument("need --env NAME=VALUE, -w >= 2 and at least one text file");
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

// log-softmax in double
std::vector<double> log_probs(const float* lg, size_t n) {
    double mx = lg[0];
    for (size_t i = 1; i < n; i++) mx = std::max(mx, (double)lg[i]);
    double sum = 0;
    for (size_t i = 0; i < n; i++) sum += std::exp((double)lg[i] - mx);
    const double lse = mx + std::log(sum);
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++) out[i] = (double)lg[i] - lse;
    return out;
}

void compare_row(int text, int64_t pos, const float* a, const float* b, size_t vocab, int32_t target) {
    const std::vector<double> la = log_probs(a, vocab), lb = log_probs(b, vocab);
    double kl = 0;
    size_t ta = 0, tb = 0;
    for (size_t i = 0; i < vocab; i++) {
        kl += std::exp(la[i]) * (la[i] - lb[i]);
        if (la[i] > la[ta]) ta = i;
        if (lb[i] > lb[tb]) tb = i;
    }
    std::printf("%d %lld %.6f %.6f %.6e %d\n", text, (long long)pos, -la[target], -lb[target], kl, ta == tb ? 1 : 0);
}

// One text: the same prefill in both, then W tokens per step: A in W passes of 1, B in one pass of W
void run_text(tona::backend::Backend& a, tona::backend::Backend& b, const std::vector<int32_t>& t, int text,
              const Options& o, size_t vocab) {
    a.reset(), b.reset();
    a.prefill(t.data(), o.prefill), b.prefill(t.data(), o.prefill);
    std::vector<float> rows((size_t)o.width * vocab);
    for (size_t i = o.prefill; i + o.width < t.size(); i += o.width) {
        for (int j = 0; j < o.width; j++) {
            const float* lg = a.forward(&t[i + j], 1);
            std::copy(lg, lg + vocab, rows.begin() + (ptrdiff_t)(j * vocab));
            a.commit(1);
        }
        const float* lb = b.forward(&t[i], o.width);
        for (int j = 0; j < o.width; j++)
            compare_row(text, (int64_t)i + j, rows.data() + j * vocab, lb + j * vocab, vocab, t[i + j + 1]);
        b.commit(o.width);
    }
}

int run(int argc, char** argv) {
    const Options o = parse(argc, argv);
    check(o);
    tona::backend::register_builtin();
    const auto model_ptr = tona::load_model_as<tona::Qwen4ExpModel>(o.model);
    const tona::Qwen4ExpModel& model = *model_ptr;
    const tona::Tokenizer tok(model.gguf());
    tona::backend::CreateOptions co;
    co.max_ctx = o.ctx;
    auto a = tona::backend::registry::create("auto", model, co);
    const size_t eq = o.env.find('=');
    setenv(o.env.substr(0, eq).c_str(), o.env.substr(eq + 1).c_str(), 1);  // read when B's passes are built
    auto b = tona::backend::registry::create("auto", model, co);
    const size_t vocab = model.hp().vocab;
    std::fprintf(stderr, "A %s, B %s with %s, passes of %d\n", a->info().name.c_str(), b->info().name.c_str(),
                 o.env.c_str(), o.width);
    for (size_t k = 0; k < o.texts.size(); k++) {
        std::vector<int32_t> t = tok.encode(read_file(o.texts[k]), tona::Tokenizer::Special::none);
        if ((int64_t)t.size() > o.tokens + o.prefill + 1) t.resize(o.tokens + o.prefill + 1);
        if ((int64_t)t.size() + 1 > o.ctx) throw std::runtime_error("--ctx is smaller than the text");
        std::fprintf(stderr, "text %zu: %s, %zu tokens\n", k, o.texts[k].c_str(), t.size());
        run_text(*a, *b, t, (int)k, o, vocab);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tona-width-nll: %s\n", e.what());
        return 1;
    }
}
