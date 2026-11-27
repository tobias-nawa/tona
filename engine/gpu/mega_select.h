// Expert selection of one layer in one pass, done by the CPU while the kernel waits at the router. The decision
// (softmax and top-k per token, the union over the tokens, optionally narrowed by OEA) is backend::ExpertRoute; this
// class feeds it, applies it to the cache and the device (slots, in CPU mode the split of the misses between the CPU
// and the GPU reading pinned RAM) and keeps the statistics.
#pragma once

#include <vector>

#include "backend/expert_route.h"
#include "backend/expert_split.h"
#include "gpu/mega.h"

namespace tona::gpu {

class ExpertSelect {
public:
    ExpertSelect(Mega& g, uint32_t l, int nb);
    void run();

private:
    // router -> ranking, OEA, union (mega_select.cpp)
    const float* router_logits(int b, float* scratch) const;
    void rank_token(int b);
    void report_nan_logits(int b, int bad);
    void mark_first_rows();
    void count_union();
    void note_selection();
    void log_route(int b);
    void record_traces();
    void predict_stats();
    // GPU path
    void assign_slots();
    void read_misses(const std::vector<std::pair<int, int>>& miss);
    // CPU path (mega_select_cpu.cpp)
    void plan_cpu();
    int install_pending(std::unordered_set<int>& keep);
    void split_misses(backend::CpuPlan& P);
    void publish_plan(const backend::CpuPlan& P, int n_miss);
    void choose_fill(backend::CpuPlan& P);

    Mega& g_;
    const uint32_t l_;
    const int nb_;
    const int E_, ne_;
    MegaMailbox& mb_;
    LayerCache& C_;
    backend::ExpertRoute route_;   // the decision: ranking, OEA, union (backend/expert_route.h)
    const std::vector<int>& uni_;  // its union
};

}  // namespace tona::gpu
