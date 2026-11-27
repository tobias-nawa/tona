// Tests of the installer core without hardware: recommendation and checks for invented machines, the configuration
// there and back. Usage: test_install REPO
#include <cmath>
#include <cstdio>
#include <filesystem>

#include "core/config.h"
#include "core/context_plan.h"
#include "core/describe.h"
#include "core/module.h"
#include "core/pkgs.h"
#include "core/steps.h"
#include "tests/test_support.h"

using namespace tona::install;

namespace {

int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "FAIL: %s\n", what), fails++;
}

Hardware ref_pc(double ram_gb) {
    Hardware h;
    h.os = "linux", h.distro = "fedora", h.cpu = "AMD Ryzen 9 9950X", h.cpu_vendor = "AuthenticAMD", h.cpu_family = 26;
    h.ram_total = (int64_t)(ram_gb * 1e9), h.ram_gbs = 72, h.kfd_access = true, h.disk_free = (int64_t)500e9;
    Gpu g;
    g.pci_vendor = 0x1002, g.pci_device = 0x7590, g.vendor = "AMD", g.arch = "gfx1200";
    g.vram_total = (int64_t)17.1e9, g.vram_used = (int64_t)1.6e9;
    h.gpus = {g};
    return h;
}

bool has_error(const Module& m, const Hardware& h, const Settings& s) {
    for (const Note& n : m.check(h, s))
        if (n.warn) return true;
    return false;
}

constexpr const char* CODER = "coder-iq1_m";

double mid(const Range& r) { return (r.lo + r.hi) / 2; }

void test_recommendation(const Module& m) {
    const Hardware big = ref_pc(66), small = ref_pc(33.6);
    const Settings sb = m.recommend(big), ss = m.recommend(small);
    expect(sb.ram_gb == 60 && ss.ram_gb == 27, "RAM limit: installed minus 6 GB, rounded down");
    expect(sb.vram_gb > 14.5 && sb.vram_gb < 15.5, "VRAM limit: free minus reserve");
    expect(sb.ctx_auto && ss.ctx_auto, "the context is auto by default");
    expect(sb.ctx == 262144 && ss.ctx == kContextFloor, "auto: 256K at 64 GB, the floor at 32 GB (no room to pin)");
    expect(sb.seqs == 2 && ss.seqs == 2, "two agents from 24 GB");
}

// "auto" keeps the experts pinned first (RAM limit above 31.6 GB), then takes the largest context for all agents
void test_auto_context(const Module& m) {
    Settings s = m.recommend(ref_pc(66));
    for (double ram : {58.0, 40.0, 38.0, 34.0, 33.0, 32.0, 27.0}) {
        s.ram_gb = ram, s.seqs = 2, s.ctx_auto = true;
        resolve_context(&m, s);
        const double kv = s.ctx * 0.23e9 / 16384 * s.seqs / 1e9;
        expect(s.ctx >= kContextFloor, "auto never goes below the floor");
        expect(ram < 31.6 || s.ctx == kContextFloor || ram - kv >= 31.6, "auto keeps the experts pinned");
        expect(s.ctx == 262144 || s.ctx == kContextFloor || ram - kv * 2 < 31.6 || s.ctx == 131072 || s.ctx == 65536,
               "auto steps");
    }
    s.ram_gb = 38, s.seqs = 2, resolve_context(&m, s);
    expect(s.ctx == 131072, "38 GB, 2 agents: 128K (256K would unpin the experts)");
    s.seqs = 1, resolve_context(&m, s);
    expect(s.ctx == 262144, "38 GB, 1 agent: 256K");
    s.seqs = 4, resolve_context(&m, s);
    expect(s.ctx == 65536 && s.ram_gb - s.ctx * 0.23e9 / 16384 * 4 / 1e9 >= 31.6, "38 GB, 4 agents: 64K");
    s.ram_gb = 27, resolve_context(&m, s);
    expect(s.ctx == kContextFloor, "below the pin limit: the floor");
    s.ctx_auto = false, s.ctx = 262144, s.ram_gb = 38, s.seqs = 2, resolve_context(&m, s);
    expect(s.ctx == 262144, "a fixed size stays");
    const Hardware h = ref_pc(66);
    expect(m.estimate(h, s, CODER).pinned == false, "a fixed 256K x 2 at 38 GB does not pin the experts");
    s.ctx_auto = true, resolve_context(&m, s);
    expect(m.estimate(h, s, CODER).pinned, "auto at 38 GB pins them");
    expect(mid(m.estimate(h, s, CODER).decode) > 0, "the speed range follows");
    expect(context_text(131072) == "128K" && parse_context("128K") == 131072 && parse_context("64k") == 65536 &&
               parse_context("auto") == 0 && parse_context("12") == 0 && parse_context("K") == 0,
           "context sizes as text");
}

void test_checks(const Module& m) {
    const Hardware big = ref_pc(66);
    const Settings sb = m.recommend(big);
    expect(!has_error(m, big, sb), "recommendation on the reference PC has no error");
    Hardware intel = big;
    intel.cpu_vendor = "GenuineIntel", intel.cpu_family = 6;
    expect(has_error(m, intel, sb), "CPU other than Zen 5: error (built with -march=znver5)");
    Settings low = sb;
    low.vram_gb = 6;
    expect(has_error(m, big, low), "VRAM limit below the minimum: error");
}

bool contains(const Range& r, double v) { return r.lo <= v && v <= r.hi; }

// The reference PC: the range of every model contains the rate on its card, plus or minus 15 %, and its anchor names
// the reference PC
void test_estimate_contains_the_card(const Module& m) {
    const Hardware big = ref_pc(66);
    const Settings s = m.recommend(big);
    for (const tona::offer::ModelCard& c : tona::offer::model_cards()) {
        const tona::offer::Rate* r = c.rate_at(kListContext);
        if (!r || !r->decode_tps) continue;
        const Estimate e = m.estimate(big, s, c.id);
        expect(e.model == c.id && e.context_tokens == r->context_tokens, "estimate names the card's model and context");
        expect(contains(e.decode, *r->decode_tps), "decode range contains the card's rate");
        expect(std::abs(mid(e.decode) / *r->decode_tps - 1) < 0.02,
               "the centre is the card's rate on the reference PC");
        expect(std::abs(e.width - 0.15) < 1e-9 && e.unknown.empty(), "same class: plus or minus 15 %, nothing unknown");
        expect(!r->prefill_tps || contains(e.prefill, *r->prefill_tps), "prefill range contains the card's rate");
        expect(e.anchor.contains("9950X") && !e.anchor.contains("9600"), "the anchor names the reference PC");
    }
}

// Less memory bandwidth, less RAM, a slower drive: a lower range
void test_estimate_follows_the_machine(const Module& m) {
    const Hardware big = ref_pc(66), small = ref_pc(33.6);
    const Settings sb = m.recommend(big), ss = m.recommend(small);
    const Estimate eb = m.estimate(big, sb, CODER), es = m.estimate(small, ss, CODER);
    expect(mid(eb.decode) > mid(es.decode), "less RAM: slower");
    Hardware slow_ram = big;
    slow_ram.ram_gbs = big.ram_gbs / 2;
    const Estimate er = m.estimate(slow_ram, sb, CODER);
    expect(mid(er.decode) < mid(eb.decode) && er.width > eb.width,
           "half the RAM bandwidth: a lower range, another class");
    Hardware low = ref_pc(26), low_sata = low;
    low_sata.disk_speed = {true, 132000, 242, 552, ""};
    expect(mid(m.estimate(low_sata, m.recommend(low_sata), CODER).decode) <
               mid(m.estimate(low, m.recommend(low), CODER).decode),
           "experts from a SATA drive: slower than from NVMe");
}

// An input the detection did not find widens the range; no GPU bandwidth, no estimate
void test_estimate_with_missing_inputs(const Module& m) {
    const Hardware big = ref_pc(66);
    const Settings s = m.recommend(big);
    const Estimate known = m.estimate(big, s, CODER);
    Hardware no_ram = big;
    no_ram.ram_gbs = 0;
    const Estimate e = m.estimate(no_ram, s, CODER);
    expect(e.width > known.width && e.unknown == std::vector<std::string>{"ram-bandwidth"},
           "unknown RAM bandwidth: wider");
    expect(contains(e.decode, mid(known.decode)), "the wider range still holds the known one's centre");
    Hardware other_gpu = big;
    other_gpu.gpus[0].pci_device = 0x1234;
    const Estimate none = m.estimate(other_gpu, s, CODER);
    expect(none.decode.hi == 0 && none.prefill.hi == 0 && !none.unknown.empty() && !none.setup.empty(),
           "a GPU of unknown bandwidth: no estimate, and why");
    expect(speed_text(none) == "unknown" || speed_text(none) == "unbekannt", "the screens say unknown");
}

// A model without a card gets no numbers
void test_estimate_without_card(const Module& m) {
    const Hardware big = ref_pc(66);
    for (const char* id : {"no-such-model", "coder-dmt-164"}) {
        const Estimate e = m.estimate(big, m.recommend(big), id);
        expect(e.decode.hi == 0 && e.prefill.hi == 0 && e.load_s == 0, "no card: no numbers");
    }
}

// Drives emulated as in research/measurements/2026-10-27-ram-limit.md: a HDD cannot serve unpinned experts, with
// pins it only starts slowly.
void test_drive(const Module& m) {
    const Hardware big = ref_pc(66), small = ref_pc(33.6);
    const Settings sb = m.recommend(big), ss = m.recommend(small);
    Hardware hdd_small = small, hdd_big = big;
    hdd_small.disk_speed = hdd_big.disk_speed = {true, 200, 160000, 87, ""};
    expect(has_error(m, hdd_small, ss), "HDD below the pin limit: error");
    expect(!has_error(m, hdd_big, sb), "HDD with all experts pinned: no error");
    expect(m.estimate(hdd_big, sb, CODER).load_s > 600, "HDD: start takes over 10 minutes");
    expect(m.estimate(big, sb, CODER).load_s < 60, "NVMe with pins: start under a minute");
}

// A model step of a checkout and of a release install names the model as the model page does
void test_step_texts(const Module& m, const std::string& repo) {
    const Hardware h = ref_pc(66);
    Settings s = m.recommend(h);
    s.service = true, s.models = {"coder-iq1_m", "ornith-1.5-35b-a3b"};
    Target checkout;
    checkout.repo = repo;
    for (const Target& t : {checkout, release_target("/nonexistent/tona")}) {
        std::string all;
        for (const Step& st : plan_steps(h, s, t)) all += step_text(st) + "\n";
        expect(all.contains("Download and verify: Coder · ") && all.contains("Expert profile: Ornith") &&
                   !all.contains("coder-iq1_m"),
               "model steps name the model as the model page does");
    }
}

void test_config(const Module& m, const std::string& repo) {
    Settings c = m.recommend(ref_pc(66));
    c.ram_gb = 26.5, c.oea = false, c.drafter = "dflash", c.port = 9000, c.role_effort = "subagent=low",
    c.clock_hold = true;
    c.model_from = "/some/download";
    const std::string text = render(c, repo);
    expect(!text.contains("/some/download"), "model_from is not written to work/tona.conf");
    const std::string tmp = (std::filesystem::temp_directory_path() / "tona-setup-test.conf").string();
    {
        FILE* f = std::fopen(tmp.c_str(), "w");
        expect(f != nullptr, "the temporary config file can be written");
        if (!f) return;
        std::fputs(text.c_str(), f);
        std::fclose(f);
    }
    const Settings r = load(tmp, Settings{});
    std::filesystem::remove(tmp);
    expect(!load(tmp, Settings{}).clock_hold && !Settings{}.clock_hold, "the GPU hold is off by default");
    expect(r.ram_gb == 26.5 && !r.oea && r.drafter == "dflash" && r.port == 9000 && r.role_effort == "subagent=low" &&
               r.clock_hold && r.model == "coder-iq1_m" && r.model_from.empty(),
           "work/tona.conf there and back");
}

void test_package_manager() {
    const Hardware big = ref_pc(66);
    Hardware deb = big;
    deb.distro = "ubuntu", deb.distro_like = "debian";
    expect(package_manager(deb) == "apt" && package_manager(big) == "dnf", "package manager by distribution");
}

}  // namespace

int main(int argc, char** argv) {
    return tona::test::guard("install", [&] {
        const std::string repo = argc > 1 ? argv[1] : ".";
        const Module* m = find_module(ref_pc(66));
        expect(m && m->name() == "hip-gfx1200", "reference PC finds the module hip-gfx1200");
        if (!m) return 1;
        test_recommendation(*m);
        test_checks(*m);
        test_auto_context(*m);
        test_estimate_contains_the_card(*m);
        test_estimate_follows_the_machine(*m);
        test_estimate_with_missing_inputs(*m);
        test_estimate_without_card(*m);
        test_drive(*m);
        test_step_texts(*m, repo);
        test_config(*m, repo);
        test_package_manager();
        if (!fails) std::printf("ok\n");
        return fails ? 1 : 0;
    });
}
