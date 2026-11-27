#include "core/describe.h"

#include <cmath>
#include <format>

#include "core/speed.h"

namespace tona::install {

namespace {

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (const std::string& x : v) out += (out.empty() ? "" : sep) + x;
    return out;
}

}  // namespace

std::string context_label(uint32_t tokens) {
    const double k = tokens % 1000 == 0 ? tokens / 1000.0 : tokens / 1024.0;
    return std::format("{:.0f}K", k);
}

std::string basis_text(offer::Basis b) { return offer::basis_word(b); }

namespace {

// "58-78" as the screens write it: with an en dash
std::string dashed(const Range& r) {
    std::string t = range_text(r);
    t.replace(t.find('-'), 1, "–");
    return t;
}

const char* input_label(const std::string& key) {
    if (key == "gpu-bandwidth") return "GPU memory";
    if (key == "gpu-size") return "GPU compute units";
    if (key == "ram-bandwidth") return "RAM";
    if (key == "drive-speed") return "Drive";
    if (key == "gpu-experts") return "Experts in GPU memory";
    if (key == "ram-experts") return "RAM for experts";
    return "?";
}

const char* input_unit(const std::string& key) {
    return key == "gpu-size" ? "" : key == "gpu-experts" || key == "ram-experts" ? "GB" : "GB/s";
}

const char* unknown_label(const std::string& key) {
    if (key == "gpu-bandwidth") return "GPU memory bandwidth";
    if (key == "ram-bandwidth") return "RAM bandwidth";
    if (key == "drive-speed") return "drive speed";
    return "GPU size";
}

}  // namespace

std::string speed_text(const Estimate& e) {
    if (e.decode.hi <= 0) return "unknown";
    return std::format("{} {}", dashed(e.decode), "tok/s");
}

std::string speed_footer(bool mac) {
    // the rows name no rate and no context: one line says which one it is and for which machine
    return mac ? "Decode at 16K context, estimated for this Mac from chip and memory."
               : "Decode at 16K context, estimated for this PC from GPU, RAM and CPU.";
}

std::string this_machine(bool mac) { return mac ? "this Mac" : "this PC"; }

std::string other_machines(bool mac) { return mac ? "other computers" : "other PCs"; }

std::string class_text(const offer::HwClass& c, bool exact, const Hardware& h) {
    if (exact) return c.label;
    if (h.os == "macos")
        return std::format("{} (nearest class; this Mac has {:.0f} GB)", c.label, h.ram_total / 1073741824.0);
    return c.label + " (this machine is no class of its own)";
}

std::string class_title(const offer::HwClass& c, bool exact, bool mac) {
    if (exact) return c.label;
    return mac ? "none, nearest: " + c.label : "none of Tona's classes";
}

std::string load_speed_text(const Estimate& e) {
    if (e.per_agent.hi <= 0) return "unknown";
    std::string t = dashed(e.per_agent) + " tok/s";
    if (e.agents > 1) t += " each, " + dashed(e.together) + " together";
    return t;
}

std::string speed_plain(const Estimate& e) {
    if (e.decode.hi <= 0) return "decode: unknown";
    return std::format("decode {} tok/s at {}", range_text(e.decode), context_label(e.context_tokens));
}

std::vector<std::string> speed_detail_lines(const Estimate& e) {
    if (e.decode.hi <= 0) {
        std::vector<std::string> out = {std::string("Speed: unknown")};
        if (!e.unknown.empty()) out.push_back(std::string("Not found: ") + unknown_label(e.unknown.front()));
        return out;
    }
    const char* tps = "tok/s";
    std::vector<std::string> out = {
        std::format("{} {} {} {} {}", "Decode", dashed(e.decode), tps, "at", context_label(e.context_tokens)),
        std::format("{} {} {}", "Prefill", e.prefill.hi > 0 ? dashed(e.prefill) : "?", tps),
        std::format("{} ±{:.0f} %", "Range", e.width * 100)};
    std::string used;
    for (const auto& [key, value] : e.inputs)
        used += std::format("{}{} {:.{}f}{}{}", used.empty() ? "" : " · ", input_label(key), value,
                            value < 10 && key != "gpu-size" ? 1 : 0, *input_unit(key) ? " " : "", input_unit(key));
    out.push_back(std::string("Inputs: ") + used);
    if (!e.unknown.empty()) {
        std::string names;
        for (const std::string& k : e.unknown) names += (names.empty() ? "" : ", ") + std::string(unknown_label(k));
        out.push_back(std::string("Not found (wider range): ") + names);
    }
    out.push_back(std::string("Anchor: measured on ") + e.anchor);
    return out;
}

std::string languages_with(const offer::ModelCard& c, offer::Level level) {
    std::vector<std::string> out;
    for (const offer::Language& l : offer::languages())
        if (const offer::LanguageVerdict* v = c.verdict(l.id); v && v->level == level) out.push_back(l.label);
    return join(out, ", ");
}

VerdictParts verdict_parts(const offer::ModelCard& c) {
    VerdictParts v;
    const std::string strong = languages_with(c, offer::Level::Strong), weak = languages_with(c, offer::Level::Weak);
    if (!strong.empty()) v.strong = std::string("strong: ") + strong;
    if (!weak.empty()) v.weak = std::string("weak: ") + weak;
    if (!v.strong.empty() || !v.weak.empty()) return v;
    // a model without language verdicts (knowledge): the keywords of its strengths and weaknesses
    std::vector<std::string> st, w;
    for (const offer::Remark& r : c.strengths) st.push_back(r.brief.empty() ? r.text : r.brief);
    for (const offer::Remark& r : c.weaknesses) w.push_back(r.brief.empty() ? r.text : r.brief);
    if (!st.empty()) v.strong = std::string("strong: ") + join(st, ", ");
    if (!w.empty()) v.weak = std::string("weak: ") + join(w, ", ");
    return v;
}

std::string verdict_line(const offer::ModelCard& c) {
    const VerdictParts v = verdict_parts(c);
    return v.strong + (v.strong.empty() || v.weak.empty() ? "" : "   ") + v.weak;
}

std::vector<std::string> detail_lines(const offer::ModelCard& c) {
    std::vector<std::string> out;
    out.reserve(offer::languages().size() + 1);
    out.push_back(std::string("License: ") + c.license.text);
    for (const offer::Language& l : offer::languages())
        if (const offer::LanguageVerdict* v = c.verdict(l.id); v && v->level != offer::Level::Open)
            out.push_back(std::format("{}: {} [{}]", l.label, v->detail, basis_text(v->basis)));
    // the sentences behind the keywords of the list, with their basis
    for (const offer::Remark& r : c.strengths)
        out.push_back(std::format("{} {} [{}]", "Strong:", r.text, basis_text(r.basis)));
    for (const offer::Remark& r : c.weaknesses)
        out.push_back(std::format("{} {} [{}]", "Weak:", r.text, basis_text(r.basis)));
    return out;
}

std::string level_text(offer::Level l, offer::Basis b) {
    if (l == offer::Level::Open) return "still open";
    const char* word = l == offer::Level::Strong ? "strong" : "weak";
    return b == offer::Basis::Measured ? std::string(word) : std::format("{} ({})", word, basis_text(b));
}

}  // namespace tona::install
