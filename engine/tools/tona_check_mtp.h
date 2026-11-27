// tona-check --mtp: the MTP head of a backend, through the backend interface only (mtp_fill, mtp_step, mtp_logits),
// so the same tool runs against every backend. --dump OUT writes a golden file, --golden FILE replays one and compares.
#pragma once

#include <cstddef>
#include <string>

#include "backend/backend.h"
#include "model/tokenizer.h"

namespace tona::check {

struct MtpOptions {
    std::string prompt;
    int n_gen = 64;      // greedy trunk tokens after the prompt; each gets a fill and `steps` chained steps
    int steps = 3;       // mtp_step calls after each fill (the default draft length)
    int rows = 8;        // calls whose full logit row goes into OUT.f32
    std::string dump;    // golden file to write
    std::string golden;  // golden file to compare against
    // Pass: top-1 equal in at least 90 % of the calls and the median relative deviation of the golden top-5 logits at
    // most this. The reference rounds activations to int8, Metal computes in f32, and the difference reaches the head
    // through h: on the Coder (DM) 94 % and 5.8e-2 (research/measurements/2026-11-11-mac-mtp.md)
    double tolerance = 0.1;
    // How the prompt goes in: "forward" token by token (entries before the last with h = 0), "fill" the same with
    // mtp_fill(-1) after each token (every entry with its h, decode passes), "prefill" through prefill()
    std::string prompt_mode = "forward";
};

// Returns the exit code: 0, or 1 if the replay misses the bar of MtpOptions::tolerance
int mtp_check(backend::Backend& be, const Tokenizer& tok, const MtpOptions& o, size_t vocab);

}  // namespace tona::check
