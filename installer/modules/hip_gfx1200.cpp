// Module of the reference backend: RX 9060 XT (gfx1200) with HIP, megakernel, experts in RAM for the CPU.
// Speeds are the rates on the models' cards in the offer data (measured on the reference PC,
// docs/backends/reference-platform.md), scaled to this machine by the bandwidth model of core/speed.h.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>

#include "core/context_plan.h"
#include "core/describe.h"
#include "core/module.h"
#include "core/pkgs.h"
#include "core/speed.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::install {

namespace {

constexpr double GB = 1e9;
// int8 KV cache per conversation and context position (12 QSA layers; 0.23 GB at 16K, gpu/mega.hip)
constexpr double KV_BYTES_PER_POS = 0.23e9 / 16384;
// KV bytes read per token and position for the two models whose KV size is not in the code: fitted to their measured
// decode at 2K and 64K context (data/modeldb/offer.json rates, one agent): the extra seconds per position times the
// card's 315 GB/s. Ornith (1/114.3 -> 1/64.7 s over 62,000 positions) gives 34 KB, Bolt (1/40.1 -> 1/36.3) 13 KB.
constexpr double KV_ORNITH = 34e3, KV_BOLT = 13e3;
// The Coder: its decode is flat from 2K to 64K (70.6, 67.8, 74.1 tok/s, same file) and on the Mac from 2K to 16K, so
// the 14 KB of cache per position are not read in full (sparse attention); no read cost until a measurement shows one.
// The cache size still takes RAM (kv_gb).
constexpr double KV_READ_CODER = 0;
// Fixed VRAM before the expert slots (weights, buffers, prefill workspace, checkpoint), per slot across all 48 layers,
// and per further conversation (measured with TONA_VRAM_LOG=1). At least 10 slots are needed (experts per token).
constexpr double VRAM_BASE_GB = 6.3, VRAM_SLOT_GB = 0.0982, VRAM_SEQ_GB = 0.422;

// The setup of every rate on the models' cards (docs/backends/reference-platform.md), named here once
constexpr const char* REFERENCE_SETUP = "RX 9060 XT 16 GB, Ryzen 9 9950X, 64 GB DDR5-6000";

// Expert slots of the setup the cards were measured on: what recommend() gives the reference PC (14.8 GB VRAM limit,
// two conversations), so its estimate is the card's number and not 1 % less.
constexpr double REFERENCE_SLOTS = 82;

// From this RAM limit (KV cache already taken off) the engine pins all experts (backend/budget.h); below it they come
// through the page cache. Measured on the Coder, decode code and prefill at 31.5 GB against 58 GB: 53.5 / 89.2 and
// 772 / 1741 (research/measurements/2026-10-27-ram-limit.md). The bandwidth model does not see this step.
constexpr double PIN_ALL_GB = 31.6;
constexpr double UNPINNED_DECODE = 0.6, UNPINNED_PREFILL = 0.44;
// RAM the pinned experts do not get: PIN_ALL_GB less the 25.2 GB of the Coder's experts
constexpr double RAM_OVERHEAD_GB = 6.4;
// RAM left for experts on the reference PC: 58 GB limit, 7.4 GB KV (2 conversations at 256K), the overhead above
constexpr double REFERENCE_RAM_FOR_EXPERTS_GB = 44;

// GPUs of this backend by PCI device id: GB/s the GPU reads its memory at and compute units. 0x7590 is the RX 9060 XT,
// 16 GB and 8 GB with the same chip: 315 GB/s measured of 320 GB/s peak (research/measurements/2026-10-22-micro),
// 32 CUs. A gfx1200 card not in this table gets no estimate.
struct GpuSpec {
    uint32_t pci_device;
    double gbs, units;
};
constexpr GpuSpec GPUS[] = {{0x7590, 315, 32}};
constexpr double REFERENCE_VRAM_GB = 15;  // the 16 GB card's size class
constexpr double REFERENCE_RAM_GBS = 72;  // docs/backends/reference-platform.md: 72 GB/s read, both CCDs
// The reference PC's drive in the unit of the setup's probe (DiskSpeed::seq_mb_s, sequential 1 MiB reads, which this
// backend uses): the 2653 MB/s of DISK_POINTS below. Ungeprueft: that the measurement file of the point ran the setup's
// own probe; the estimate test on the reference hardware (tests/test_flow.cpp) holds the anchor to its own rate.
constexpr double REFERENCE_DRIVE_GBS = 2.65;

// Bytes per token of the models Tona runs here. Coder: docs/backends/reference-platform.md (3.52 GB dense, 480 experts
// of 2.046 MB per pass, 12,288 experts). The others from their parameter counts (data/modeldb/models/*.json) and file
// size with one bit width for all weights (ungeprueft: the tensor types per layer). Ornith: 36 B weights, 3 B active,
// 8 of 256 experts, so experts 34.1 B and the rest 1.9 B; at 4.82 bit/weight (21.7 GB) that is dense 1.17 GB, experts
// 20.5 GB, 0.64 GB read. Bolt: 74 B, 9 B active, 9 of 512 experts at 2.06 bit/weight: experts 66.2 B = 17.05 GB of its
// 22.0 GB, dense 4.97 GB, 0.30 GB read.
struct ModelBytes {
    const char* id;
    Traffic traffic;
};
constexpr ModelBytes MODEL_BYTES[] = {
    {"coder-iq1_m", {3.52, 0.98, 25.2, KV_READ_CODER}},
    {"ornith-1.5-35b-a3b", {1.17, 0.64, 20.5, KV_ORNITH}},
    {"nemotron-bolt-74b-a9b-24gb", {4.97, 0.30, 17.05, KV_BOLT}},
};
constexpr const char* CURVE_MODEL = "coder-iq1_m";  // the model the pin step above was measured with

// Drives, emulated with IOReadBandwidthMax (research/measurements/2026-10-27-ram-limit.md): sequential MB/s and the
// server start in s with all experts pinned (58 GB limit) and without (26 GB).
struct DiskPoint {
    double seq_mb_s, load_pinned_s, load_unpinned_s;
};
constexpr DiskPoint DISK_POINTS[] = {
    {87, 821, 444},    // HDD
    {552, 62, 34},     // SATA-SSD
    {2653, 23, 10.5},  // NVMe of the reference PC
};
constexpr double HDD_MAX_MB_S = 300;  // below this a drive is too slow for experts without pins

// Sequential MB/s: measured, else by drive kind (fresh install without a large file)
double disk_mb_s(const Hardware& h) {
    if (h.disk_speed.measured) return h.disk_speed.seq_mb_s;
    return h.disk.kind == "hdd" ? 90 : h.disk.kind == "ssd" ? 550 : 2600;
}

// Logarithmic in MB/s between the points, the first or last point outside them.
double disk_at(double mb_s, double DiskPoint::* f) {
    const auto& P = DISK_POINTS;
    if (mb_s <= P[0].seq_mb_s) return P[0].*f;
    for (size_t i = 1; i < std::size(P); i++)
        if (mb_s <= P[i].seq_mb_s) {
            const double t = std::log(mb_s / P[i - 1].seq_mb_s) / std::log(P[i].seq_mb_s / P[i - 1].seq_mb_s);
            return P[i - 1].*f + t * (P[i].*f - P[i - 1].*f);
        }
    return P[std::size(P) - 1].*f;
}

// RAM at which this model's experts are all pinned: its experts plus the overhead. The Coder gives PIN_ALL_GB
// (measured); the other models' value follows the same rule (ungeprueft: measured only for the Coder).
double pin_all_gb(const std::string& model) {
    const auto* bytes = std::find_if(std::begin(MODEL_BYTES), std::end(MODEL_BYTES),
                                     [&](const ModelBytes& x) { return model == x.id; });
    return bytes == std::end(MODEL_BYTES) ? PIN_ALL_GB : bytes->traffic.experts_gb + RAM_OVERHEAD_GB;
}

double kv_gb(const Settings& s) { return s.ctx * KV_BYTES_PER_POS * s.seqs / GB; }

const GpuSpec* gpu_spec(const Gpu* g) {
    if (g)
        for (const GpuSpec& x : GPUS)
            if (x.pci_device == g->pci_device) return &x;
    return nullptr;
}

// What the speed model reads from this machine under these settings (speed.h)
Machine machine_of(const Hardware& h, const Settings& s, const Gpu* g) {
    Machine m;
    if (const GpuSpec* spec = gpu_spec(g)) m.gpu_gbs = spec->gbs, m.units = spec->units;
    m.fast_gb = std::max(0.0, s.vram_gb - VRAM_BASE_GB - (s.seqs - 1) * VRAM_SEQ_GB);
    m.ram_gb = std::max(0.0, s.ram_gb - kv_gb(s) - RAM_OVERHEAD_GB);
    m.ram_gbs = h.ram_gbs;
    m.drive_gbs = disk_mb_s(h) / 1000;
    return m;
}

Anchor reference_anchor(const offer::Rate& r) {
    Anchor a;
    a.machine = {
        315, 32, REFERENCE_SLOTS * VRAM_SLOT_GB, REFERENCE_RAM_FOR_EXPERTS_GB, REFERENCE_RAM_GBS, REFERENCE_DRIVE_GBS,
        1,   1};
    a.decode = r.decode_tps.value_or(0), a.prefill = r.prefill_tps.value_or(0), a.context_tokens = r.context_tokens;
    a.label = REFERENCE_SETUP;
    return a;
}

// Same GPU and memory size, as much RAM as the reference and within 20 % of its bandwidth (an unknown bandwidth
// widens the range instead)
bool reference_class(const Hardware& h, const Gpu& g) {
    const bool ram_bw = h.ram_gbs <= 0 || (h.ram_gbs > 0.8 * REFERENCE_RAM_GBS && h.ram_gbs < 1.25 * REFERENCE_RAM_GBS);
    return g.pci_device == GPUS[0].pci_device && g.vram_total / GB > REFERENCE_VRAM_GB && h.ram_total > 60 * GB &&
           ram_bw;
}

class HipGfx1200 final : public Module {
public:
    std::string name() const override { return "hip-gfx1200"; }
    std::string title() const override { return "AMD RDNA4 gfx1200 (RX 9060 XT), HIP, reference backend"; }

    const Gpu* pick(const Hardware& h) const override {
        for (const Gpu& g : h.gpus)
            if (g.vendor == "AMD" && g.arch == "gfx1200" && !g.integrated) return &g;
        return nullptr;
    }

    Settings recommend(const Hardware& h) const override {
        Settings s;
        // RAM: everything except 6 GB for system and desktop (the same MemoryMax as in just and the service)
        s.ram_gb = std::max(8.0, std::floor(h.ram_total / GB - 6));
        // VRAM: what is free now (the desktop already holds its share) minus a 0.64 GB reserve
        if (const Gpu* g = pick(h)) s.vram_gb = std::floor((g->vram_total - g->vram_used) / GB * 10 - 6.4) / 10;
        s.seqs = s.ram_gb >= 24 ? 2 : 1;
        s.ctx = auto_context(s);
        s.prompt_cache_gb = h.disk_free > 150 * GB ? 16 : 4;
        return s;
    }

    // Experts first (every GB of KV cache is a GB less for pinned experts, decode 0.6x without them), then the
    // largest context for all agents in the RAM that is left
    int auto_context(const Settings& s) const override {
        return plan_context({s.ram_gb, pin_all_gb(s.model), KV_BYTES_PER_POS, s.seqs});
    }

    Estimate estimate(const Hardware& h, const Settings& s, const std::string& model) const override {
        Estimate e;
        e.model = model;
        const offer::ModelCard* card = offer::model_card(model);
        const offer::Rate* r = card ? card->rate_at(kListContext) : nullptr;
        const auto* bytes = std::find_if(std::begin(MODEL_BYTES), std::end(MODEL_BYTES),
                                         [&](const ModelBytes& x) { return model == x.id; });
        if (!r || !r->decode_tps || bytes == std::end(MODEL_BYTES)) {
            e.setup = "this model has no measured speed in the offer data";
            return e;
        }
        const Gpu* g = pick(h);
        Machine m = machine_of(h, s, g);
        const bool unpinned = model == CURVE_MODEL && s.ram_gb - kv_gb(s) < PIN_ALL_GB;
        if (unpinned) m.decode_factor = UNPINNED_DECODE, m.prefill_factor = UNPINNED_PREFILL;
        project(reference_anchor(*r), m, bytes->traffic, g && reference_class(h, *g), e);
        project_load(reference_anchor(*r), m, bytes->traffic, {s.seqs, static_cast<double>(s.ctx)}, e);
        const double mb_s = disk_mb_s(h);
        const bool pinned = s.ram_gb - kv_gb(s) >= PIN_ALL_GB;
        e.pinned = s.ram_gb - kv_gb(s) >= pin_all_gb(model);
        e.load_s = disk_at(mb_s, pinned ? &DiskPoint::load_pinned_s : &DiskPoint::load_unpinned_s);
        return e;
    }

    std::vector<Note> check(const Hardware& h, const Settings& s) const override {
        std::vector<Note> n;
        if (!pick(h)) n.push_back({true, "no RX 9060 XT (gfx1200) found; this backend runs only there"});
        if (h.cpu_vendor != "AuthenticAMD" || h.cpu_family != 26)
            n.push_back(
                {true, "the engine is built for Zen 5 (-march=znver5); it will not start on this CPU (" + h.cpu + ")"});
        if (!h.kfd_access) n.push_back({true, "no access to /dev/kfd: add the user to the group render"});
        const double need = VRAM_BASE_GB + 10 * VRAM_SLOT_GB + (s.seqs - 1) * VRAM_SEQ_GB;
        if (s.vram_gb < need)
            n.push_back(
                {true, std::format("VRAM limit {:.1f} GB too small, at least {:.1f} GB needed", s.vram_gb, need)});
        if (s.ram_gb < 20)
            n.push_back({false,
                         "below 20 GB RAM most experts come from the NVMe: much slower (measured "
                         "from 20 GB)"});
        else if (s.ram_gb < 31.6)
            n.push_back({false,
                         "below a RAM limit of ~32 GB the experts do not fit pinned in RAM: about 40 % "
                         "slower than from 32 GB"});
        if (kv_gb(s) > 0.25 * s.ram_gb)
            n.push_back(
                {false, std::format("KV cache {:.1f} GB for {} agents with {}K context takes much RAM from the experts",
                                    kv_gb(s), s.seqs, s.ctx / 1024)});
        const double mb_s = disk_mb_s(h);
        if (s.ram_gb < PIN_ALL_GB && mb_s < HDD_MAX_MB_S)
            n.push_back({true, std::format("drive too slow ({:.0f} MB/s): below a RAM limit of ~32 GB the experts "
                                           "come from the drive, measured below 0.1 tok/s. Put the model on an SSD "
                                           "or add RAM",
                                           mb_s)});
        else if (s.ram_gb >= PIN_ALL_GB && mb_s < HDD_MAX_MB_S)
            n.push_back({false,
                         "slow drive: the start takes ~14 minutes (all experts are read), then full "
                         "speed"});
        if (s.ram_gb * GB > h.ram_total - 2 * GB)
            n.push_back({true, "RAM limit above installed RAM minus 2 GB: the system gets no room"});
        return n;
    }

    std::vector<Package> packages(const Hardware& h) const override {
        Package kfd = Package{.id = "/dev/kfd",
                              .purpose = "access to the GPU (group render)",
                              .hint = "sudo usermod -aG render $USER, then log in again"};
        kfd.found = h.kfd_access;
        return {kfd};
    }

    std::vector<Step> toolchain(const std::string& repo) const override {
        return {{"Fetch and verify ROCm 10.1 into work/ (scripts/setup.sh, ~7 GB, no root)",
                 "scripts/setup.sh",
                 fs::exists(fs::path(repo) / "work/rocm-10.1/bin/python"),
                 {}}};
    }
};

}  // namespace

const Module& hip_gfx1200_module() {
    static const HipGfx1200 m;
    return m;
}

}  // namespace tona::install
