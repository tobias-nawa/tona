// Speculative decoding run of tona-gpu-gen (--dflash): drafter proposes, the target model verifies. The loop is in
// tona_gpu_gen_spec.cpp, the stderr report in tona_gpu_gen_spec_report.cpp.
#pragma once

#include <utility>
#include <vector>

#include "tona_gpu_gen.h"

namespace tona::gen {

// Debug and tuning switches; all documented in docs/use/environment.md (tools).
struct SpecEnv {
    bool cpu;          // TONA_DFLASH_CPU: drafter runs on the CPU as reference
    bool check;        // TONA_DFLASH_CHECK: run both drafters and compare their drafts (GPU drafter only)
    bool ema;          // TONA_DFLASH_EMA: measure pass times with a moving average
    bool adaptive;     // not TONA_DFLASH_FIXED: pick the draft length per pass by expected tokens per time
    int prompt_width;  // TONA_PROMPT_NB: prompt tokens per pass
    int probe;         // TONA_DRAFT_PROBE: extra CPU draft of this many tokens, compared with the output at the end
    int small_block;   // TONA_DRAFT_SMALL
    int big_at;        // TONA_DRAFT_BIG_AT
    static SpecEnv read();
};

class SpecRun {
public:
    SpecRun(Session& s, const Options& o, const std::vector<int32_t>& ids);
    int run();

private:
    static constexpr int kWidths = 9;  // pass widths 1..8
    static constexpr int kCalibPositions = 8;

    void feed_prompt();
    bool one_pass();
    void draft_on_cpu(std::vector<int32_t>& block);
    void draft_on_gpu(std::vector<int32_t>& block, int n_drafts);
    void shorten_by_expected_rate(std::vector<int32_t>& block, int n_drafts);
    int count_accepted(const float* logits, const std::vector<int32_t>& block) const;
    void record_calibration(int accepted);
    void inject_features(int count);
    bool emit(const std::vector<int32_t>& block, int accepted, int32_t next, size_t width);

    void report();
    void report_summary();
    void report_probe();
    void report_experts();

    Session& s_;
    tona::gpu::Mega& mg_;
    tona::ref::DFlash& drafter_;
    const Options& o_;
    const std::vector<int32_t>& ids_;
    const SpecEnv env_;
    const size_t vocab_;
    const int64_t d_;            // model width
    const size_t feature_size_;  // target features per token that feed the drafter
    const int32_t im_end_;       // end of the assistant turn
    const int n_draft_;

    std::vector<float> last_logits_;
    std::vector<float> embeddings_;
    std::vector<float> probs_;
    int32_t last_ = 0;
    int64_t pos_ = 0;
    int pending_ = 0;   // GPU drafter: tokens to feed into the next draft
    bool full_ = true;  // the last pass accepted every draft
    std::vector<int32_t> out_;
    int passes_ = 0;
    int hist_[kWidths] = {};
    int width_[kWidths] = {};
    int same_draft_ = 0, cmp_draft_ = 0;
    std::vector<std::pair<size_t, std::vector<int32_t>>> probes_;
    // Per draft position: (selector probability, accepted) while all earlier drafts were accepted
    std::vector<std::pair<float, int>> calib_[kCalibPositions];

    double t_prompt_ = 0;
    double t_target_ = 0, t_draft_ = 0, t_draft_kernel_ = 0, t_select_ = 0;
    double pass_ms_[kWidths];
    double drafter_ms_;
    double model_ms_ = 0;  // sum of the seed times over all passes: projection without reloading
    tona::backend::Stats stats0_{};
    double disk0_ = 0;
};

}  // namespace tona::gen
