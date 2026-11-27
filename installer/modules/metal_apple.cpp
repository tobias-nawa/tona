// Module of the Metal backend: Apple silicon Macs. Recommendations come from measurements on an M1 Max 32 GB
// (research/measurements/2026-11-12-mac-dmt-server-ab.md, 2026-11-11-mac-decode-attn.md); the speeds are those
// measurements scaled to other chips by the bandwidth model of core/speed.h.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <format>
#include <iterator>
#include <string>

#include "core/catalog.h"
#include "core/module.h"
#include "core/pkgs.h"
#include "core/speed.h"
#include "core/sys.h"

namespace tona::install {

namespace {

constexpr double GB = 1e9;
// The Mac model: 20.5 GB of matrices in the GPU working set, the 28.8 GB n-gram table read through the page cache
// (engine/backend/metal/AGENTS.md, Memory). Below 32 GB of RAM it has not run.
constexpr double MIN_RAM_GB = 32 * 1.073741824;  // 32 GiB as sold, in GB
constexpr double MODEL_GPU_GB = 20.5;
// Measured on the M1 Max 32 GB, server with --ctx 32768. coder-dmt-164 with the MTP-164 drafter: bench-set decode
// (medians of 6 rounds: code 35.1, text 26.2), prefill of a 2048-token prompt 224.8 tok/s. coder-iq1_m with
// mtp-q8_0, the start the release installs: decode 17.9 tok/s at 2K and 16K of code, prefill 136.2 tok/s at 16K, all
// 6 starts within 10 % of the median (research/measurements/2026-11-26-mac-release-rates.md). `fixed_gb`: GPU working
// set the experts do not get (dense weights, KV, buffers): coder-iq1_m 26.8 GB working set less the 10.8 GB expert
// budget of that start (same file); coder-dmt-164 keeps its 17 GB of experts resident (ungeprueft: no start log of
// the split). Bytes per token as the Coder's (docs/backends/reference-platform.md), dmt-164 with its 17 GB of experts.
// int8 KV cache per conversation and position: the Coder's 0.23 GB at 16K, the same model as on the PC
// (modules/hip_gfx1200.cpp)
constexpr double KV_BYTES_PER_POS = 0.23e9 / 16384;
struct MacAnchor {
    const char* model;
    double decode, prefill;
    uint32_t context_tokens;
    double fixed_gb;
    Traffic traffic;
};
constexpr MacAnchor ANCHORS[] = {
    {"coder-dmt-164", 35.1, 224.8, 32768, 9.8, {3.52, 0.98, 17.0, 0}},
    {"coder-iq1_m", 17.9, 136.2, 16000, 16.0, {3.52, 0.98, 25.2, 0}},
};
constexpr double ANCHOR_VRAM_GB = 26.8;  // the working set of the 32 GB M1 Max, as recommend() sets it

// Apple chips: peak memory bandwidth in GB/s and GPU cores (docs/hardware/mac-chips.md, sources there). Where a chip
// has two bandwidths by GPU core count (M3 Max, M4 Max, M5 Max) the lower one and its core count stand: the name of
// the chip does not tell the variant. The GPU reads 82 % of the peak (334 of 409.6 GB/s measured on the M1 Max,
// research/measurements/2026-11-11-mac-bandwidth.md); the other chips are scaled with the same share (ungeprueft).
struct Chip {
    const char* name;
    double peak_gbs, cores;
};
constexpr Chip CHIPS[] = {
    {"Apple M1", 68.3, 8},     {"Apple M1 Pro", 204.8, 16}, {"Apple M1 Max", 409.6, 32}, {"Apple M1 Ultra", 819.2, 64},
    {"Apple M2", 102.4, 10},   {"Apple M2 Pro", 204.8, 19}, {"Apple M2 Max", 409.6, 38}, {"Apple M2 Ultra", 819.2, 76},
    {"Apple M3", 102.4, 10},   {"Apple M3 Pro", 153.6, 18}, {"Apple M3 Max", 307.2, 30}, {"Apple M3 Ultra", 819.3, 80},
    {"Apple M4", 120, 10},     {"Apple M4 Pro", 273, 20},   {"Apple M4 Max", 410, 32},   {"Apple M5", 153, 10},
    {"Apple M5 Pro", 307, 20}, {"Apple M5 Max", 460, 32},
};
constexpr double GPU_READ_SHARE = 0.82;
constexpr const char* ANCHOR_CHIP = "Apple M1 Max";
// Expert reads from the SSD as the setup measures them (disk.cpp, measure_experts: 2 MiB blocks, 16 in flight), on the
// anchor's own M1 Max: 6.7 GB/s (the setup's drive probe on that machine,
// research/measurements/2026-11-27-mac-acceptance.md). The rule: every value of an anchor's machine comes from the same
// probe the setup runs, so the estimate on the anchor's hardware is the anchor's rate. The 13 GB/s of
// research/measurements/2026-11-11-mac-ssd.md came from another test (cached reads, probably) and is not used.
constexpr double SSD_EXPERT_GBS = 6.7;

const Chip* find_chip(const std::string& cpu) {
    const auto* c = std::find_if(std::begin(CHIPS), std::end(CHIPS), [&](const Chip& x) { return cpu == x.name; });
    return c == std::end(CHIPS) ? nullptr : c;
}

// What the speed model reads from this Mac. Unified memory has no RAM tier: the experts the arena misses come from
// the SSD.
Machine machine_of(const Hardware& h, const Settings& s, double fixed_gb) {
    Machine m;
    // `fixed_gb` holds the KV cache of the measured start (one agent, 32K); a larger cache takes the experts' room
    const double kv_gb = (s.ctx * s.seqs - 32768.0) * KV_BYTES_PER_POS / GB;
    if (const Chip* c = find_chip(h.cpu)) m.gpu_gbs = c->peak_gbs * GPU_READ_SHARE, m.units = c->cores;
    m.fast_gb = std::max(0.0, s.vram_gb - fixed_gb - kv_gb);
    if (h.disk_speed.expert_mb_s > 0) m.drive_gbs = h.disk_speed.expert_mb_s / 1000;
    return m;
}

Anchor anchor_of(const MacAnchor& a) {
    Hardware h;
    h.cpu = ANCHOR_CHIP;
    Settings s;
    s.vram_gb = ANCHOR_VRAM_GB, s.ctx = 32768, s.seqs = 1;  // the measured start
    Anchor r;
    r.machine = machine_of(h, s, a.fixed_gb);
    r.machine.drive_gbs = SSD_EXPERT_GBS;
    r.decode = a.decode, r.prefill = a.prefill, r.context_tokens = a.context_tokens;
    r.label = "M1 Max 32 GB";
    return r;
}

int macos_major(const Hardware& h) {
    const std::string v = h.distro_name.substr(std::min(h.distro_name.size(), std::string("macOS ").size()));
    return v.empty() || !std::isdigit(static_cast<unsigned char>(v[0])) ? 0 : std::stoi(v);
}

class MetalApple final : public Module {
public:
    std::string name() const override { return "metal"; }
    std::string title() const override { return "Apple silicon (M1 or newer), Metal backend (experimental)"; }

    const Gpu* pick(const Hardware& h) const override {
        for (const Gpu& g : h.gpus)
            if (h.os == "macos" && g.vendor == "Apple") return &g;
        return nullptr;
    }

    Settings recommend(const Hardware& h) const override {
        Settings s;
        // maintainer decision 2026-11-12: the Mac gets the pre-tiled file, as soon as it is published; until then the
        // installer would end in "not published yet" (models.toml, UNPUBLISHED)
        s.model = metal_apple_model([](const std::string& id) {
            const CatalogEntry* e = find_entry(id);
            return e && pinned(*e);
        });
        s.drafter = "mtp";  // for coder-dmt-164 the head cut to the same experts (models.toml, mtp = )
        s.ram_gb = std::max(8.0, std::floor(h.ram_total / GB - 6));
        if (const Gpu* g = pick(h)) s.vram_gb = std::floor(static_cast<double>(g->vram_total) / GB * 10) / 10;
        s.seqs = 1;          // the one setup measured on the Mac (limits() below holds what the pages offer)
        s.ctx = 32768;       // the context of every server measurement on the Mac
        s.ctx_auto = false;  // "auto" has no Mac plan: the measured size stays
        s.oea = false;       // shared experts of parallel conversations: reference backend only
        s.prompt_cache_gb = h.disk_free > 150 * GB ? 16 : 4;
        return s;
    }

    // Only what was measured: one agent at 32K. The server can run several sequences (TONA_SEQS up to SMAX, the KV
    // cache int8, 3.98 GB at 64K per agent) and longer contexts; the runs with 1, 2 and 4 agents at 16K to 256K that
    // raise this limit are not done yet. "auto" has no plan on the Mac.
    Limits limits() const override { return {1, {32768}, false}; }

    // The offer data's rates are the reference PC's. The Mac's own measurements, scaled to this chip, are these; a
    // model without one has no estimate.
    Estimate estimate(const Hardware& h, const Settings& s, const std::string& model) const override {
        Estimate e;
        e.model = model;
        const auto* a =
            std::find_if(std::begin(ANCHORS), std::end(ANCHORS), [&](const MacAnchor& x) { return model == x.model; });
        if (a == std::end(ANCHORS)) {
            e.setup = "this model has no measured speed on the Mac";
            return e;
        }
        const bool same_class = h.cpu == ANCHOR_CHIP && h.ram_total >= MIN_RAM_GB * GB;
        const Machine m = machine_of(h, s, a->fixed_gb);
        project(anchor_of(*a), m, a->traffic, same_class, e);
        project_load(anchor_of(*a), m, a->traffic, {s.seqs, static_cast<double>(s.ctx)}, e);
        return e;
    }

    std::vector<Note> check(const Hardware& h, const Settings& s) const override {
        std::vector<Note> n;
        if (!pick(h)) n.push_back({true, "no Apple silicon GPU found; this backend runs only there"});
        if (const int v = macos_major(h); v && v < 15)
            n.push_back({true, std::format("macOS {} is too old: the kernels need Metal 3.2 (macOS 15 or newer)", v)});
        if (h.ram_total < MIN_RAM_GB * GB)
            n.push_back({true, std::format("{:.0f} GB RAM: the Mac model needs 32 GB (20.5 GB on the GPU, the n-gram "
                                           "table through the page cache)",
                                           h.ram_total / GB)});
        if (const Gpu* g = pick(h); g && g->vram_total / GB < MODEL_GPU_GB + 2)
            n.push_back({true, std::format("GPU working set {:.1f} GB: the model's matrices take {:.1f} GB",
                                           g->vram_total / GB, MODEL_GPU_GB)});
        if (s.seqs > 1) n.push_back({false, "more than one conversation is not measured on the Mac"});
        return n;
    }

    std::vector<Package> packages(const Hardware&) const override { return {}; }

    // A checkout builds the kernels at build time with Apple's Metal toolchain; a release ships the prebuilt library
    std::vector<Step> toolchain(const std::string&) const override {
        return {{"Metal toolchain for building the kernels (Xcode component, ~700 MB)",
                 "xcodebuild -downloadComponent MetalToolchain",
                 run("xcrun --find metal").status == 0,
                 {}}};
    }
};

}  // namespace

std::string metal_apple_model(const std::function<bool(const std::string&)>& installable) {
    const bool tiled = installable("coder-dmt-164") && installable(drafter_entry_id("mtp", "coder-dmt-164"));
    return tiled ? "coder-dmt-164" : "coder-iq1_m";
}

const Module& metal_apple_module() {
    static const MetalApple m;
    return m;
}

}  // namespace tona::install
