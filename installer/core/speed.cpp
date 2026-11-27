#include "core/speed.h"

#include <algorithm>
#include <cmath>
#include <format>

namespace tona::install {

namespace {

constexpr double kSameClass = 0.15, kOtherClass = 0.30;  // half widths (plus or minus 15 % and 30 %)
constexpr double kPerUnknown = 0.15, kMaxWidth = 0.6;
constexpr double kPrefillExtra = 0.10;  // prefill is scaled by one ratio of GPU sizes only

}  // namespace

double hit_rate(double share) {
    constexpr double xs[] = {0, 1.0 / 3, 2.0 / 3, 1}, ys[] = {0, 0.85, 0.94, 1};
    const double x = std::clamp(share, 0.0, 1.0);
    for (size_t i = 1; i < std::size(xs); i++)
        if (x <= xs[i]) return ys[i - 1] + (x - xs[i - 1]) / (xs[i] - xs[i - 1]) * (ys[i] - ys[i - 1]);
    return 1;
}

namespace {

// Experts the pass reads: the union of what `agents` tokens route to. One agent reads t.expert_gb; with the share f of
// all experts per token and independent routing the union is 1 - (1-f)^n of them (independent routing is an upper
// bound: the routing of code is skewed, so the union is smaller in practice).
double expert_union_gb(const Traffic& t, int agents) {
    if (agents <= 1 || t.experts_gb <= 0) return t.expert_gb;
    const double f = std::min(1.0, t.expert_gb / t.experts_gb);
    return t.experts_gb * (1 - std::pow(1 - f, agents));
}

}  // namespace

double seconds_per_pass(const Machine& m, const Traffic& t, const Load& load) {
    const double fast = std::min(m.fast_gb, t.experts_gb);
    const double hit = hit_rate(t.experts_gb > 0 ? fast / t.experts_gb : 1);
    const double need = t.experts_gb - fast;  // experts outside GPU memory
    const double in_ram = need <= 0 ? 1 : std::min(1.0, m.ram_gb / need);
    const double experts = expert_union_gb(t, load.agents);
    const double miss = experts * (1 - hit);
    const double ram_s = m.ram_gbs > 0 ? miss * in_ram / m.ram_gbs : 0;
    const double drive_s = m.drive_gbs > 0 ? miss * (1 - in_ram) / m.drive_gbs : 0;
    const double kv_gb = load.agents * load.context * t.kv_bytes_per_pos / 1e9;
    return (t.dense_gb + experts * hit + kv_gb) / m.gpu_gbs + ram_s + drive_s;
}

double seconds_per_token(const Machine& m, const Traffic& t) { return seconds_per_pass(m, t); }

void project(const Anchor& a, const Machine& m, const Traffic& t, bool same_class, Estimate& e) {
    e.context_tokens = a.context_tokens, e.anchor = a.label;
    if (m.gpu_gbs <= 0) {
        e.unknown = {"gpu-bandwidth"};
        e.setup = "the memory bandwidth of this GPU is not known";
        return;
    }
    // An unknown input takes the anchor's value, so the estimate stays near it, and widens the range
    Machine use = m;
    if (use.ram_gbs <= 0 && a.machine.ram_gbs > 0)
        use.ram_gbs = a.machine.ram_gbs, e.unknown.emplace_back("ram-bandwidth");
    if (use.drive_gbs <= 0) use.drive_gbs = a.machine.drive_gbs, e.unknown.emplace_back("drive-speed");
    const bool units_known = m.units > 0;
    if (!units_known) e.unknown.emplace_back("gpu-size");
    e.inputs = {{"gpu-bandwidth", use.gpu_gbs}, {"gpu-experts", use.fast_gb}};
    if (use.ram_gbs > 0) e.inputs.emplace_back("ram-bandwidth", use.ram_gbs);
    if (use.ram_gb > 0) e.inputs.emplace_back("ram-experts", use.ram_gb);
    e.inputs.emplace_back("drive-speed", use.drive_gbs);
    if (units_known) e.inputs.emplace_back("gpu-size", m.units);
    const double speed = seconds_per_token(a.machine, t) / seconds_per_token(use, t);
    const double decode = a.decode * speed * use.decode_factor / a.machine.decode_factor;
    const double prefill =
        a.prefill * (units_known ? m.units / a.machine.units : 1) * use.prefill_factor / a.machine.prefill_factor;
    e.width = std::min(kMaxWidth, (same_class ? kSameClass : kOtherClass) + kPerUnknown * e.unknown.size());
    e.decode = {decode * (1 - e.width), decode * (1 + e.width)};
    if (a.prefill > 0) {
        const double w = std::min(kMaxWidth, e.width + kPrefillExtra);
        e.prefill = {prefill * (1 - w), prefill * (1 + w)};
    }
}

void project_load(const Anchor& a, const Machine& m, const Traffic& t, const Load& load, Estimate& e) {
    e.agents = load.agents, e.load_context = static_cast<uint32_t>(load.context);
    if (e.decode.hi <= 0) return;  // no estimate at all: nothing to scale
    Machine use = m;               // the same stand-ins for unknown inputs as project() took
    if (use.ram_gbs <= 0) use.ram_gbs = a.machine.ram_gbs;
    if (use.drive_gbs <= 0) use.drive_gbs = a.machine.drive_gbs;
    const double base = seconds_per_pass(a.machine, t, {1, static_cast<double>(a.context_tokens)});
    const double speed = base / seconds_per_pass(use, t, load);
    const double each = a.decode * speed * use.decode_factor / a.machine.decode_factor;
    const bool far =
        a.context_tokens > 0 && (load.context > 4.0 * a.context_tokens || load.context * 4 < a.context_tokens);
    const double width = std::min(kMaxWidth, e.width + (load.agents > 1 ? kPerUnknown : 0) + (far ? kPerUnknown : 0));
    e.per_agent = {each * (1 - width), each * (1 + width)};
    e.together = {e.per_agent.lo * load.agents, e.per_agent.hi * load.agents};
    e.width_load = width;
}

std::string range_text(const Range& r) {
    // steps that fit the width of the range: below 10 one decimal, below 100 whole numbers, then multiples of 5 and 50;
    // the low end rounds down and the high end up, so the text never claims more than the range
    const auto step = [](double v) { return v < 10 ? 0.1 : v < 100 ? 1.0 : v < 1000 ? 5.0 : 50.0; };
    const double lo = std::floor(r.lo / step(r.lo) + 1e-9) * step(r.lo),
                 hi = std::ceil(r.hi / step(r.hi) - 1e-9) * step(r.hi);
    if (hi < 10) return std::format("{:.1f}-{:.1f}", lo, hi);
    return std::format("{:.0f}-{:.0f}", lo, hi);
}

}  // namespace tona::install
