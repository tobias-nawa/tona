// The flow of the installer without a screen: pages, pre-selection per task, language and hardware class, tasks that
// are only coming, room proposals, space options, the command line and the exit codes. No network, no model files.
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#include "core/catalog.h"
#include "core/cli_args.h"
#include "core/cli_plan.h"
#include "core/config.h"
#include "core/describe.h"
#include "core/exit_codes.h"
#include "core/flow.h"
#include "core/module.h"
#include "core/pkgs.h"
#include "core/port.h"
#include "core/room.h"
#include "core/space.h"

using namespace tona::install;

namespace {

int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "FAIL: %s\n", what), fails++;
}

constexpr const char* kClass = "pc-16gb-vram-64gb-ram";

Hardware ref_pc(double ram_gib, double vram_gib) {
    Hardware h;
    h.os = "linux", h.cpu_vendor = "AuthenticAMD", h.cpu_family = 26;
    h.ram_total = static_cast<int64_t>(ram_gib * 1073741824.0), h.disk_free = static_cast<int64_t>(1.2e12);
    Gpu g;
    g.vendor = "AMD", g.arch = "gfx1200", g.vram_total = static_cast<int64_t>(vram_gib * 1073741824.0);
    h.gpus = {g};
    return h;
}

void test_hardware_class() {
    expect(match_hw_class(ref_pc(62.2, 16)).exact,
           "a 64 GB PC with 16 GB VRAM is the measured class (the system reports less)");
    const HwMatch small = match_hw_class(ref_pc(30, 12));
    expect(!small.exact && small.cls == kClass, "another machine falls back to the measured class and says so");
}

Hardware mac_hw(double ram_gib, const char* chip) {
    Hardware h;
    h.os = "macos", h.cpu = chip, h.cpu_vendor = "Apple", h.disk_free = static_cast<int64_t>(500e9);
    h.ram_total = static_cast<int64_t>(ram_gib * 1073741824.0);
    Gpu g;
    g.vendor = "Apple", g.arch = "apple-m1-max", g.vram_total = h.ram_total / 100 * 78;
    h.gpus = {g};
    return h;
}

void test_mac_class() {
    for (const auto& [ram, cls] :
         {std::pair{32.0, "mac-32gb"}, {48.0, "mac-32gb"}, {64.0, "mac-64gb"}, {128.0, "mac-64gb"}}) {
        const HwMatch m = match_hw_class(mac_hw(ram, "Apple M1 Max"));
        expect(m.exact && m.cls == cls, "a Mac gets the class of its memory");
    }
    const HwMatch small = match_hw_class(mac_hw(16, "Apple M2"));
    expect(!small.exact && small.cls == "mac-32gb", "a Mac below the smallest class gets the nearest Mac class");
    Hardware big_pc = ref_pc(128, 16);
    expect(match_hw_class(big_pc).cls == kClass, "a PC never gets a Mac class");
    const Selection sel = default_selection("mac-32gb");
    expect(sel.tasks == std::vector<std::string>{"coding"} && sel.models == std::vector<std::string>{"coder-iq1_m"},
           "the Mac offers what the Metal backend runs: the Coder for coding");
    const std::string text = plan_text(make_plan(mac_hw(16, "Apple M2"), nullptr, sel, {}, true, "/nonexistent"));
    expect(text.contains("Class:   Mac, 32 GB (nearest class; this Mac has 16 GB)") && !text.contains("this PC") &&
               !text.contains("no class of its own"),
           "the plan names the nearest Mac class");
}

Estimate coder_at(const Hardware& h, const Module& m, int agents, int ctx, double ram_gb,
                  const char* model = "coder-iq1_m") {
    Settings s = m.recommend(h);
    s.seqs = agents, s.ctx = ctx, s.ctx_auto = false, s.ram_gb = ram_gb;
    return m.estimate(h, s, model);
}

// The rate follows the agents and the context: lower for each agent with more agents (the union of experts and every
// KV cache), lower with a longer context, and it drops further where the KV cache pushes the experts out of RAM
void test_load_speed() {
    Hardware h = ref_pc(62.2, 16);
    h.ram_gbs = 72, h.disk_speed = {true, 0, 0, 2650, ""};
    h.gpus[0].pci_device = 0x7590;  // the RX 9060 XT, the card of the module's table
    const Module* m = find_module(h);
    expect(m != nullptr, "the reference PC has a module");
    if (!m) return;
    const Estimate one = coder_at(h, *m, 1, 65536, 58), four = coder_at(h, *m, 4, 65536, 58);
    expect(one.per_agent.hi > 0 && one.agents == 1 && four.agents == 4 && four.load_context == 65536,
           "the estimate names agents and context");
    expect(four.per_agent.hi < one.per_agent.hi, "each agent gets less with more agents");
    expect(four.together.hi > one.together.hi, "but all agents together get more");
    expect(four.width_load > one.width_load, "the band is wider for the unmeasured agents");
    // Ornith's decode falls with the context (114 to 65 tok/s from 2K to 64K); the Coder's is flat there
    const char* ornith = "ornith-1.5-35b-a3b";
    const Estimate short_ctx = coder_at(h, *m, 1, 32768, 58, ornith), long_ctx = coder_at(h, *m, 1, 262144, 58, ornith);
    expect(long_ctx.per_agent.hi < short_ctx.per_agent.hi, "a longer context is slower");
    // 4 agents at 256K with 36 GB: the KV cache (15 GB) takes the room of the pinned experts, the rate falls far below
    // the same agents at 64K
    const Estimate pushed = coder_at(h, *m, 4, 262144, 36), fits = coder_at(h, *m, 4, 65536, 36);
    expect(pushed.per_agent.hi < 0.8 * fits.per_agent.hi, "KV that pushes the experts out of RAM makes a step");
    expect(one.decode.hi > 0 && one.context_tokens == 16000, "the rate at the measured point stays");
}

// The reference PC with the default settings gives the card's rate (the anchor), within the stated range
void test_pc_anchor() {
    Hardware h = ref_pc(62.2, 16);
    h.ram_gbs = 72, h.disk_speed = {true, 0, 0, 2650, ""};
    h.gpus[0].pci_device = 0x7590;
    const Module* m = find_module(h);
    expect(m != nullptr, "the reference PC has a module");
    if (!m) return;
    const Estimate e = m->estimate(h, m->recommend(h), "coder-iq1_m");
    expect(e.decode.lo <= 67.8 && 67.8 <= e.decode.hi && e.width < 0.151, "the anchor machine gets the anchor's rate");
}

void test_port_and_updates() {
    set_port_probe([](int p) { return p == 8662 ? PortState::Busy : p == 8664 ? PortState::Ours : PortState::Free; });
    expect(next_free_port(8662) == 8663, "a taken port proposes the next free one");
    expect(next_free_port(8664) == 8664, "a port of our own server is free");
    set_port_probe({});
    Options o;
    std::string err;
    const char* ok[] = {"setup", "--port", "9100"};
    expect(parse_args(3, ok, o, err) && o.port == "9100", "--port is parsed");
    Options bad;
    const char* low[] = {"setup", "--port", "80"};
    expect(!parse_args(3, low, bad, err), "a port below 1024 is refused");
    Settings s;
    s.port = 9100;
    expect(render(s, "/nonexistent").contains("TONA_PORT='9100'"), "the port goes into tona.conf");
}

void test_defaults_and_toggles() {
    Selection sel = default_selection(kClass);
    expect(sel.tasks == std::vector<std::string>{"coding", "knowledge"}, "coding and knowledge are on by default");
    expect(sel.models == std::vector<std::string>{"coder-iq1_m", "nemotron-bolt-74b-a9b-24gb"},
           "the recommendations are on");
    expect(!contains(sel.models, "ornith-1.5-35b-a3b"), "the alternative stays off");
    expect(primary_model(sel) == "coder-iq1_m", "the Coder starts the server");
    expect(!toggle_task(sel, "documents", kClass) && !contains(sel.tasks, "documents"),
           "a coming task cannot be switched on");
    expect(!toggle_task(sel, "nonsense", kClass), "an unknown task cannot be switched on");
    expect(toggle_model(sel, "ornith-1.5-35b-a3b", kClass) && contains(sel.models, "ornith-1.5-35b-a3b"),
           "the alternative can be chosen");
    toggle_task(sel, "coding", kClass);
    expect(!contains(sel.models, "coder-iq1_m") && !contains(sel.models, "ornith-1.5-35b-a3b"),
           "switching coding off drops its models");
    expect(contains(sel.models, "nemotron-bolt-74b-a9b-24gb"), "and keeps the models of the other task");
    expect(!toggle_model(sel, "coder-iq1_m", kClass), "a model of a task that is off cannot be chosen");
    expect(default_drafter("coder-iq1_m") == "mtp" && default_drafter("ornith-1.5-35b-a3b") == "none",
           "only the Coder has an MTP head");
    expect(apply_selection({}, sel).model == "nemotron-bolt-74b-a9b-24gb" && apply_selection({}, sel).drafter == "none",
           "knowledge only: Bolt starts, without a drafter");
}

void test_pages() {
    Selection sel = default_selection(kClass);
    expect(page_order(sel).size() == 6 && next_page(Page::Tasks, sel) == Page::Languages,
           "coding adds the language page");
    toggle_task(sel, "coding", kClass);
    expect(page_order(sel).size() == 5 && next_page(Page::Tasks, sel) == Page::Models,
           "without coding six pages become five");
    expect(prev_page(Page::Models, sel) == Page::Tasks && next_page(Page::Go, sel) == Page::Go, "back and the end");
}

void test_languages() {
    Selection sel = default_selection(kClass);
    toggle_language(sel, "rust");
    toggle_language(sel, "python");
    const std::vector<LanguageProposal> p = language_proposals(sel, kClass);
    expect(p.size() == 2 && p[0].language == "python" && p[0].model == "coder-iq1_m", "Python: the Coder is strong");
    expect(p[1].language == "rust" && p[1].model == "ornith-1.5-35b-a3b" && p[1].level == tona::offer::Level::Strong,
           "Rust: Ornith is strong");
    expect(contains(models_to_consider(sel, kClass), "ornith-1.5-35b-a3b"),
           "Rust: Ornith is named as a hint, not switched on");
    expect(!contains(sel.models, "ornith-1.5-35b-a3b"), "the hint does not change the selection");
    toggle_language(sel, "rust");
    expect(models_to_consider(sel, kClass).empty(), "Python alone needs no hint");
    toggle_language(sel, "typescript");
    expect(language_proposals(sel, kClass).back().level == tona::offer::Level::Open,
           "a language without a measurement stays open, no guess");
}

void test_proposed_models() {
    Selection sel = default_selection(kClass);
    toggle_language(sel, "python");
    toggle_language(sel, "rust");
    toggle_language(sel, "go");
    const std::vector<std::string> p = proposed_models(sel, kClass);
    expect(p == std::vector<std::string>{"coder-iq1_m", "ornith-1.5-35b-a3b"},
           "Python and Go: the Coder, Rust: Ornith, each once");
    toggle_task(sel, "coding", kClass);
    expect(proposed_models(sel, kClass).empty(), "without the coding task nothing is proposed");
}

void test_names() {
    Selection sel;
    FlowError err;
    expect(selection_from_names({{"coding"}, {"rust"}, {}}, kClass, sel, err) && sel.models.size() == 1 &&
               sel.languages.size() == 1,
           "--task coding --lang rust: the Coder, one language");
    expect(selection_from_names({{}, {}, {"coder", "bolt"}}, kClass, sel, err) && sel.tasks.size() == 2 &&
               sel.models.size() == 2,
           "--models coder,bolt names two models and their tasks");
    expect(selection_from_names({{}, {}, {"ornith"}}, kClass, sel, err) &&
               sel.models == std::vector<std::string>{"ornith-1.5-35b-a3b"},
           "--models ornith: the alternative alone");
    expect(!selection_from_names({{"documents"}, {}, {}}, kClass, sel, err) && err.code == kExitTaskComing,
           "a coming task: exit 6");
    expect(!selection_from_names({{"coding"}, {"klingon"}, {}}, kClass, sel, err) && err.code == kExitUsage,
           "an unknown language: exit 2");
    expect(!selection_from_names({{}, {}, {"gpt"}}, kClass, sel, err) && err.code == kExitUsage,
           "an unknown model: exit 2");
    expect(resolve_model("Coder") == "coder-iq1_m" && resolve_model("nemotron-bolt") == "nemotron-bolt-74b-a9b-24gb",
           "model names resolve");
}

void test_args() {
    Options o;
    std::string error;
    const char* argv[] = {"tona setup", "--task", "coding,knowledge", "--lang", "rust", "--models", "coder,bolt",
                          "--yes",      "--json"};
    expect(parse_args(9, argv, o, error) && o.named.tasks.size() == 2 && o.named.models.size() == 2 && o.yes && o.json,
           "options parse");
    const char* bad[] = {"tona setup", "--priority", "never"};
    Options b;
    expect(!parse_args(3, bad, b, error), "a wrong priority is a usage error");
    const char *off[] = {"tona setup", "--updates", "off"}, *wrong[] = {"tona setup", "--updates", "maybe"};
    Options u, w;
    expect(parse_args(3, off, u, error) && u.updates == "off", "--updates off parses");
    expect(!parse_args(3, wrong, w, error) && error.contains("on or off"), "--updates takes on or off only");
    expect(Options{}.updates.empty() && Settings{}.update_checks, "update checks are on unless the user says no");
    const char* missing[] = {"tona setup", "--task"};
    Options m;
    expect(!parse_args(2, missing, m, error), "a missing value is a usage error");
}

void test_room() {
    const Hardware h = ref_pc(62.2, 16);
    const std::vector<RoomPlan> p = room_proposals(h, false, 14.5);
    expect(p.size() == 3 && p[1].ram_total == 64 && p[1].tona == 40 && p[1].programs == 20 && p[1].system == 4,
           "balanced on 64 GB: Tona 40, programs 20, system 4");
    expect(p[0].tona < p[1].tona && p[1].tona < p[2].tona, "work first keeps more for the user than Tona first");
    expect(p[0].tona >= 31.6, "Tona keeps enough to pin all experts at 64 GB");
    expect(room_plan(h, true, Priority::Balanced, 14.5).programs == kServerProgramsGb,
           "a server keeps little for programs");
    expect(room_plan(ref_pc(8, 16), false, Priority::Work, 14.5).tona >= kMinTonaGb, "never below the minimum");
    Settings s;
    apply_room(s, p[2], false, Priority::Tona);
    expect(s.priority == "tona" && s.use == "desktop" && s.ram_gb == p[2].tona, "the choice lands in the settings");
}

void test_space() {
    Settings s = apply_selection({}, default_selection(kClass));
    const uint64_t coder = entry_bytes("coder-iq1_m");
    expect(coder > 50'000'000'000ULL, "the catalog knows the Coder's size");
    expect(bytes_to_load(s, "/nonexistent") > coder, "both models and the drafter count");
    expect(fits(s, "/nonexistent", 1'000'000'000'000ULL) && !fits(s, "/nonexistent", 60'000'000'000ULL),
           "fits against the free space");
    SpaceContext ctx;
    ctx.free_bytes = 70'000'000'000ULL, ctx.models_dir = "/nonexistent", ctx.places = {{"/data", 900'000'000'000ULL}};
    const std::vector<SpaceOption> o = space_options(s, default_selection(kClass), kClass, ctx);
    bool shorter = false, place = false;
    for (const SpaceOption& x : o)
        shorter |= x.selection.models.size() == 1 && x.place.empty(), place |= x.place == "/data/tona";
    expect(shorter && place, "too little space: a shorter list and another place are offered");
    expect(duration_text(6000) == "≈ 1 h 40 min" && duration_text(10) == "< 1 min", "duration text");
    expect(size_text(1.2e12) == "1.2 TB" && size_text(55e9) == "55.0 GB" && size_text(62.46e9) == "62.5 GB",
           "size text");
}

// Files with the pinned size are "here" (the install step verifies them, it does not fetch them); the sizes of a row
// include the drafter, and the parts add up
void present_files_body() {
    const std::filesystem::path dir = std::filesystem::path(getenv("TMPDIR") ? getenv("TMPDIR") : "/var/tmp") /
                                      std::format("tona-test-space-{}", static_cast<long>(getpid()));
    std::filesystem::remove_all(dir);
    expect(model_bytes("coder-iq1_m") == entry_bytes("coder-iq1_m") + entry_bytes("mtp-q8_0"),
           "the size of the Coder row includes its drafter");
    const LoadSplit none = model_split("coder-iq1_m", dir.string());
    expect(none.download == model_bytes("coder-iq1_m") && none.here == 0, "nothing there: all to download");
    for (const char* id : {"coder-iq1_m", "mtp-q8_0"}) {
        const CatalogEntry* e = find_entry(id);
        for (const ModelFile& f : e->files) {
            std::filesystem::create_directories(dir / id);
            std::ofstream(dir / id / f.name()).close();
            std::filesystem::resize_file(dir / id / f.name(), f.size);  // sparse: no disk used
        }
    }
    const LoadSplit all = model_split("coder-iq1_m", dir.string());
    expect(all.download == 0 && all.here == model_bytes("coder-iq1_m"), "files of the pinned size are already here");
    Settings s = apply_selection({}, default_selection(kClass));
    const LoadSplit both = load_split(s, dir.string());
    expect(both.here == model_bytes("coder-iq1_m") && both.download == entry_bytes("nemotron-bolt-74b-a9b-24gb"),
           "only what is missing counts as download");
    expect(load_text(both).contains("already here, verified on install"), "the text says so");
    const Step model{"Fetch and verify model coder-iq1_m (SHA-256)", "", false, {}};
    const Step drafter{"Fetch and verify drafter mtp-q8_0", "", false, {}};
    const std::string here = step_text(model, dir.string());
    expect(here.contains("Verify: Coder") && here.contains("(already here)"),
           "the Go page says verify and already here for files that are there");
    expect(step_text(drafter, dir.string()).empty() &&
               step_text(model, "/nonexistent").starts_with("Download and verify: Coder"),
           "the drafter has no line of its own; missing files say download");
    std::filesystem::remove_all(dir);
}

void test_present_files() {
    try {
        present_files_body();
    } catch (const std::exception& e) {
        expect(false, e.what());
    }
}

void test_plan_and_codes() {
    const Selection sel = default_selection(kClass);
    Settings s = apply_selection({}, sel);
    s.ram_gb = 40, s.vram_gb = 14.5;
    Hardware h = ref_pc(62.2, 16);
    const PlanDoc ok = make_plan(h, nullptr, sel, s, false, "/nonexistent");
    expect(plan_exit_code(ok) == kExitNoHardware, "no backend: exit 3");
    expect(plan_json(ok).contains("\"exit_code\": 3"), "the JSON carries the exit code");
    expect(plan_text(ok).contains("Coder") && plan_text(ok).contains("decode: unknown"),
           "no backend: the text names the models and says the speed is unknown");
    expect(plan_json(ok).contains(R"("speed": "unknown")"), "no backend: the JSON says the speed is unknown");
}

// A release install builds nothing: no compiler, CMake or just may stop it (exit 7 on a Mac without Homebrew)
void test_release_needs_no_build_tools() {
    const Hardware h = ref_pc(62.2, 16);
    auto required = [](const std::vector<Package>& all, const std::string& id) {
        for (const Package& p : all)
            if (p.id == id) return p.required;
        return false;
    };
    expect(required(check_packages(h, {}, false), "cmake"), "a source install needs CMake");
    const std::vector<Package> rel = check_packages(h, {Package{.id = "/dev/kfd", .purpose = "GPU"}}, true);
    bool any_common = false;
    for (const Package& p : rel) any_common = any_common || (p.required && p.id != "/dev/kfd");
    expect(!any_common && required(rel, "/dev/kfd"), "a release install keeps only the module's checks as required");
}

void test_config_round_trip() {
    Settings s = apply_selection({}, default_selection(kClass));
    s.priority = "work", s.use = "server", s.languages = {"rust"};
    const std::string text = render(s, "/nonexistent");
    expect(text.contains("TONA_PRIORITY='work'") && text.contains("TONA_USE='server'"), "priority and use are written");
    expect(text.contains("TONA_MODELS='coder-iq1_m,nemotron-bolt-74b-a9b-24gb'"), "all chosen models are written");
}

}  // namespace

int main() {
    test_hardware_class();
    test_mac_class();
    test_load_speed();
    test_pc_anchor();
    test_port_and_updates();
    test_defaults_and_toggles();
    test_pages();
    test_languages();
    test_proposed_models();
    test_names();
    test_args();
    test_room();
    test_space();
    test_present_files();
    test_plan_and_codes();
    test_release_needs_no_build_tools();
    test_config_round_trip();
    if (fails) return 1;
    std::puts("flow: all passed");
    return 0;
}
