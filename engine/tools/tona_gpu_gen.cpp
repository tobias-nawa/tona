// Text generation without a server, on the GPU forward pass; optionally checked against the CPU reference.
// Usage: tona-gpu-gen MODEL.gguf "PROMPT" [-n TOKENS] [--slots N] [--ctx N] [--check N] [--ignore-eos]
//                   [--mega [--blocks N]]
//   --mega: megakernel forward pass (int8; TONA_MEGA_F32=1 for float matvecs); without --slots as many expert slots
//     as the free VRAM allows
//   --check N: also compute the first N steps with cpu/ref and compare the logits
//   --teacher FILE [--max N]: teacher-forced pass over a text (instead of PROMPT): mean negative log-likelihood,
//     perplexity and top-1 hit rate of the next token; TONA_GPU_INT8=1 for the int8 path; with --prefill (--mega
//     only) through the prefill path instead of token by token, without top-1
//   --multi-check N (--mega only): two conversations (the prompt and the same tokens shifted) in separate slots,
//     first each alone, then together in passes of 2-4 tokens; bitwise comparison of the logits per row, also
//     with swapped slots. See tona_gpu_gen_multi.cpp for the TONA_MULTI_* variables.
//   --batch-check N / --batch-bench N (--mega only): verify pass against single steps / time per pass of width 1-4
//   --prompts-file FILE (--mega only; instead of PROMPT: tona-gpu-gen MODEL.gguf --prompts-file FILE ...): the model
//     loads once and the prompts of FILE run one after the other, each from a reset state; stdout of each goes to
//     its own file. Format: tona_prompt_list.h.
//   --prompt-nb N (--mega only, without --dflash and --check): the prompt in passes of N tokens (1 to 4) instead of
//     single steps; the generated tokens stay single steps. Default 1. scripts/golden.sh uses 4 for the prefill check
//     (bit-identical to 1: research/measurements/2026-11-02-golden-prefill-nb.md).
//   --dflash DRAFTER.gguf [--draft N]: speculation with the DFlash2 drafter on the GPU, N draft tokens per pass
//     (at most 7), greedy; reports acceptance and tokens per pass. See tona_gpu_gen_spec.cpp for TONA_DFLASH_*.
// stdout carries the generated text only (scripts/golden.sh compares it byte for byte); logs go to stderr.
#include "tona_gpu_gen.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <stdexcept>

#include "backend/env.h"
#include "backend/runtime_env.h"
#include "tona_args.h"
#include "tools/tona_prompt_list.h"

namespace tona::gen {

namespace {

struct ValueOption {
    const char* name;
    void (*set)(Options&, const char*);
};

constexpr ValueOption kValueOptions[] = {
    {"-n", [](Options& o, const char* v) { o.n_gen = tona::tools::parse_int(v); }},
    {"--slots", [](Options& o, const char* v) { o.slots = tona::tools::parse_int(v); }},
    {"--ctx", [](Options& o, const char* v) { o.ctx = tona::tools::parse_int64(v); }},
    {"--check", [](Options& o, const char* v) { o.check = tona::tools::parse_int(v); }},
    {"--blocks", [](Options& o, const char* v) { o.blocks = tona::tools::parse_int(v); }},
    {"--teacher", [](Options& o, const char* v) { o.teacher = v; }},
    {"--prompts-file", [](Options& o, const char* v) { o.prompts_file = v; }},
    {"--max", [](Options& o, const char* v) { o.max_teacher = tona::tools::parse_int(v); }},
    {"--batch-check", [](Options& o, const char* v) { o.batch_check = tona::tools::parse_int(v); }},
    {"--multi-check", [](Options& o, const char* v) { o.multi_check = tona::tools::parse_int(v); }},
    {"--prompt-nb", [](Options& o, const char* v) { o.prompt_nb = tona::tools::parse_int(v); }},
    {"--batch-bench", [](Options& o, const char* v) { o.batch_bench = tona::tools::parse_int(v); }},
    {"--dflash", [](Options& o, const char* v) { o.dflash_path = v; }},
    {"--draft", [](Options& o, const char* v) { o.n_draft = tona::tools::parse_int(v); }},
};

void check_prompt_nb(const Options& o) {
    if (o.prompt_nb < 1 || o.prompt_nb > 4) throw std::invalid_argument("--prompt-nb must be 1 to 4");
    if (o.prompt_nb > 1 && (!o.mega || o.check > 0))
        throw std::invalid_argument("--prompt-nb above 1 needs --mega and cannot be combined with --check");
}

}  // namespace

bool parse_options(int argc, char** argv, Options& o) {
    if (argc < 3) return false;
    o.model_path = argv[1];
    const bool list = !std::strcmp(argv[2], "--prompts-file");
    o.prompt = list ? "" : argv[2];
    for (int i = list ? 2 : 3; i < argc; i++) {
        const char* arg = argv[i];
        if (!std::strcmp(arg, "--ignore-eos")) o.ignore_eos = true;
        if (!std::strcmp(arg, "--mega")) o.mega = true;
        if (!std::strcmp(arg, "--prefill")) o.use_prefill = true;
        if (i + 1 >= argc) break;
        for (const ValueOption& opt : kValueOptions)
            if (!std::strcmp(arg, opt.name)) opt.set(o, argv[i + 1]);
    }
    // The CPU reference (--check) and the teacher pass keep state that reset_for_next_prompt() does not clear
    check_prompt_nb(o);
    if (o.prompts_file && (o.check > 0 || o.teacher))
        throw std::invalid_argument("--prompts-file cannot be combined with --check or --teacher");
    return true;
}

Session::Session(const Options& o) : model(o.model_path), tok(model.gguf()) {
    if (o.dflash_path) {
        if (!o.mega) throw std::runtime_error("--dflash needs --mega");
        drafter_file = std::make_unique<tona::Gguf>(o.dflash_path);
        drafter = std::make_unique<tona::ref::DFlash>(*drafter_file, model);
    }
    if (o.mega) {
        const bool gpu_drafter = drafter && !env::flag("TONA_DFLASH_CPU");
        mg = std::make_unique<tona::gpu::Mega>(
            model, tona::gpu::MegaOptions{o.slots, o.ctx, o.blocks, !env::flag("TONA_MEGA_F32"),
                                          drafter ? drafter->target_layers() : std::vector<int64_t>{},
                                          gpu_drafter ? drafter_file.get() : nullptr, nullptr});
    } else {
        fw = std::make_unique<tona::gpu::Forward>(model, o.slots > 0 ? o.slots : 64, o.ctx);
    }
}

StatsView Session::stats() const {
    if (fw) return {fw->stats().expert_hits, fw->stats().expert_misses, fw->stats().upload_bytes};
    return {mg->stats().expert_hits, mg->stats().expert_misses, mg->stats().upload_bytes};
}

void Session::print_loaded(double load_seconds, int slots_requested) const {
    const size_t vram = fw ? fw->vram_bytes() : mg->vram_bytes();
    const int slots = mg ? mg->slots() : (slots_requested > 0 ? slots_requested : 64);
    std::string mega_info;
    if (mg)
        mega_info = std::format(", megakernel: {} steps, {} blocks (at most {}, LDS {} B, {} VGPRs), verify pass ",
                                mg->ops(), mg->blocks(), mg->max_blocks(), mg->lds_bytes(), mg->vgprs()) +
                    mg->batch_info();
    std::fprintf(stderr, "loaded in %.1f s, VRAM %.2f GB (expert cache %d per layer)%s\n", load_seconds, vram / 1e9,
                 slots, mega_info.c_str());
}

void Session::reset_for_next_prompt() const {
    if (!mg) throw std::runtime_error("--prompts-file needs --mega");
    (*mg).reset();
    // The CPU drafter keeps K/V per position. The GPU drafter ring needs no reset: its attention reads only
    // positions [max(0, p - window + 1), pos + nb) (gpu/mega_op_dattn.h:14-19, 220), each written in this prompt
    // before it is read.
    if (drafter) (*drafter).reset();
}

int argmax(const std::vector<float>& v) { return argmax(v.data(), v.size()); }

double neg_log_prob(const float* g, size_t n, int32_t next) {
    const float mx = *std::max_element(g, g + n);
    double z = 0;
    for (size_t i = 0; i < n; i++) z += std::exp((double)(g[i] - mx));
    return std::log(z) - (double)(g[next] - mx);
}

Deviation deviation(const float* a, const float* b, size_t n) {
    Deviation d{0, 0};
    for (size_t i = 0; i < n; i++) {
        d.max_diff = std::max(d.max_diff, (double)std::fabs(a[i] - b[i]));
        d.max_ref = std::max(d.max_ref, (double)std::fabs(b[i]));
    }
    return d;
}

double disk_read_mb() {
    std::ifstream f("/proc/self/io");
    if (!f) return -1;
    std::string key;
    unsigned long long value = 0, read_bytes = 0;
    while (f >> key >> value)
        if (key == "read_bytes:") read_bytes = value;
    return read_bytes / 1e6;
}

void print_profile(FILE* out, const char* title, tona::gpu::Mega& mg) {
    double total = 0;
    for (const auto& [name, us] : mg.profile())
        if (!name.starts_with("nu#") && !name.starts_with("gd#"))
            total += us;  // TONA_PROF_LAYERS: expert counts, not times
    std::fprintf(out, "%s: %.2f ms\n", title, total / 1e3);
    for (const auto& [name, us] : mg.profile())
        std::fprintf(out, "  %-24s %7.0f µs  %4.1f %%\n", name.c_str(), us, 100 * us / total);
}

namespace {

int run_prompt(Session& s, const Options& o) {
    const auto ids = s.tok.encode(o.prompt);
    if (o.batch_bench > 0 && s.mg) return run_batch_bench(s, o, ids);
    if (s.drafter) return run_speculative(s, o, ids);
    if (o.multi_check > 0 && s.mg) return run_multi_check(s, o, ids);
    if (o.batch_check > 0 && s.mg) return run_batch_check(s, o, ids);
    return run_plain(s, o, ids);
}

// stdout of every prompt goes to its own file; the logs stay on stderr. The first failing prompt ends the run.
int run_prompt_list(Session& s, Options o) {
    const auto entries = tona::tools::parse_prompt_list(tona::tools::read_file_bytes(o.prompts_file));
    for (const auto& e : entries) {
        const std::string prompt = tona::tools::read_file_bytes(e.prompt_path);
        o.prompt = prompt.c_str();
        if (!std::freopen(e.output_path.c_str(), "w", stdout))
            throw std::runtime_error("cannot write " + e.output_path);
        s.reset_for_next_prompt();
        std::fprintf(stderr, "prompt %s -> %s\n", e.prompt_path.c_str(), e.output_path.c_str());
        if (const int rc = run_prompt(s, o)) return rc;
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

}  // namespace tona::gen

int main(int argc, char** argv) {
    tona::runtime_env::set_defaults(argv);
    using namespace tona::gen;
    try {
        Options o;
        if (!parse_options(argc, argv, o)) {
            std::fprintf(stderr,
                         "Usage: tona-gpu-gen MODEL.gguf \"PROMPT\" [-n N] [--slots N] [--ctx N] [--check N]\n");
            return 2;
        }
        const auto t0 = Clock::now();
        Session s(o);
        s.print_loaded(seconds_since(t0), o.slots);
        if (o.check > 0) s.reference = tona::ref::make_oracle(s.model);
        if (o.teacher) return run_teacher(s, o);
        if (!o.prompts_file) return run_prompt(s, o);
        return run_prompt_list(s, o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
