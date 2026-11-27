// Shared pieces of tona-gpu-gen: options, the loaded model with its forward pass, and the numeric helpers the modes
// use. Each mode lives in its own tona_gpu_gen_<mode>.cpp and returns the process exit code.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "cpu/dflash.h"
#include "cpu/ref.h"
#include "gpu/forward.h"
#include "gpu/mega.h"
#include "model/tokenizer.h"

namespace tona::gen {

using Clock = std::chrono::steady_clock;

inline double seconds_between(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}
inline double seconds_since(Clock::time_point from) { return seconds_between(from, Clock::now()); }

struct Options {
    const char* model_path = nullptr;
    const char* prompt = nullptr;
    int n_gen = 32;
    int slots = -1;  // unset: as many expert slots as the free VRAM allows (megakernel) or 64 (plain forward)
    int check = 0;
    int max_teacher = 1 << 30;
    int batch_check = 0;
    int batch_bench = 0;
    int prompt_nb = 1;  // --prompt-nb: prompt tokens per pass without a drafter (--mega only)
    int multi_check = 0;
    int n_draft = 3;
    int blocks = 0;
    int64_t ctx = 8192;
    const char* teacher = nullptr;
    const char* prompts_file = nullptr;  // --prompts-file: the list of prompts instead of the PROMPT argument
    const char* dflash_path = nullptr;
    bool ignore_eos = false;
    bool mega = false;
    bool use_prefill = false;
};

// False if the arguments are too few; the caller prints the usage line.
bool parse_options(int argc, char** argv, Options& o);

struct StatsView {
    int64_t expert_hits;
    int64_t expert_misses;
    double upload_bytes;
};

// The loaded target model behind one of the two forward passes (plain `Forward` or the megakernel), plus the
// optional DFlash2 drafter and the CPU reference.
class Session {
public:
    explicit Session(const Options& o);

    tona::Qwen4ExpModel model;
    tona::Tokenizer tok;
    std::unique_ptr<tona::gpu::Forward> fw;
    std::unique_ptr<tona::gpu::Mega> mg;
    std::unique_ptr<tona::Gguf> drafter_file;
    std::unique_ptr<tona::ref::DFlash> drafter;
    std::unique_ptr<tona::ref::Oracle> reference;

    const std::vector<float>& step(int32_t token) const { return fw ? fw->step(token) : mg->step(token); }
    StatsView stats() const;
    // Back to the state of a fresh process for the next prompt: position, DeltaNet and convolution states, drafter
    // context. The expert cache stays warm (it is no input of the numbers; golden.sh shows it).
    void reset_for_next_prompt() const;
    void print_loaded(double load_seconds, int slots_requested) const;
};

int argmax(const std::vector<float>& v);
inline int argmax(const float* g, size_t n) { return (int)(std::max_element(g, g + n) - g); }

// -log p(next) of a logit row
double neg_log_prob(const float* g, size_t n, int32_t next);

// Largest |a - b| and largest |b| of two logit rows
struct Deviation {
    double max_diff;
    double max_ref;
    double relative() const { return max_diff / max_ref; }
};
Deviation deviation(const float* a, const float* b, size_t n);
inline double relative_deviation(const float* a, const float* b, size_t n) { return deviation(a, b, n).relative(); }

// Bytes this process really read from disk (page cache hits excluded), from /proc/self/io
double disk_read_mb();

// One line per megakernel step with its share of the pass time
void print_profile(FILE* out, const char* title, tona::gpu::Mega& mg);

int run_teacher(Session& s, const Options& o);
int run_teacher_tail(Session& s, const std::vector<int32_t>& ids, int tail);
int run_batch_bench(Session& s, const Options& o, const std::vector<int32_t>& ids);
int run_batch_check(Session& s, const Options& o, const std::vector<int32_t>& ids);
int run_multi_check(Session& s, const Options& o, const std::vector<int32_t>& ids);
int run_speculative(Session& s, const Options& o, const std::vector<int32_t>& ids);
int run_plain(Session& s, const Options& o, const std::vector<int32_t>& ids);

}  // namespace tona::gen
