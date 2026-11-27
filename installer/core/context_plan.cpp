#include "core/context_plan.h"

#include <algorithm>
#include <cstdlib>
#include <format>

#include "core/module.h"

namespace tona::install {

const std::vector<int>& context_steps() {
    static const std::vector<int> steps = {65536, 131072, 262144};
    return steps;
}

int plan_context(const ContextBudget& b) {
    const double room_bytes = std::max(0.0, b.ram_gb - b.pin_gb) * 1e9;
    const std::vector<int>& steps = context_steps();
    for (auto it = steps.rbegin(); it != steps.rend(); ++it)
        if (*it * b.kv_bytes_per_pos * b.agents <= room_bytes) return *it;
    return kContextFloor;
}

std::string context_text(int ctx) { return std::format("{}K", ctx / 1024); }

int parse_context(const std::string& text) {
    char* end = nullptr;
    const long k = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || k <= 0 || (*end != 'K' && *end != 'k') || end[1] != '\0') return 0;
    return static_cast<int>(k * 1024);
}

void apply_limits(const Module* m, Settings& s) {
    if (!m) return;
    const Limits l = m->limits();
    s.seqs = std::clamp(s.seqs, 1, l.max_agents);
    if (!l.auto_context) s.ctx_auto = false;
    if (l.contexts.empty() || (s.ctx_auto && l.auto_context)) return;
    if (std::find(l.contexts.begin(), l.contexts.end(), s.ctx) != l.contexts.end()) return;
    int pick = *std::min_element(l.contexts.begin(), l.contexts.end());
    for (int c : l.contexts)
        if (c <= s.ctx && c > pick) pick = c;
    s.ctx = pick;
}

void resolve_context(const Module* m, Settings& s) {
    apply_limits(m, s);
    if (m && s.ctx_auto) s.ctx = m->auto_context(s);
}

}  // namespace tona::install
