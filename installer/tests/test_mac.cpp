// The Mac in the installer: the Metal module on an invented Apple silicon machine, the Mac's catalog entries (a file
// that comes from another repository under the name the split loader needs, an entry not yet published is refused)
// and the drafter that follows the model. Against a local server; nothing leaves the machine. Usage: test_mac REPO
#include <cmath>
#include <filesystem>
#include <regex>
#include <set>
#include <string>

#include "core/catalog.h"
#include "core/config.h"
#include "core/describe.h"
#include "core/flow.h"
#include "core/model_fetch.h"
#include "core/module.h"
#include "core/release_steps.h"
#include "tests/test_support.h"
#include "tests/update_testkit.h"

namespace fs = std::filesystem;
namespace in = tona::install;
using namespace tona::test;

namespace {

in::Hardware mac(double ram_gib) {
    in::Hardware h;
    h.os = "macos", h.distro_name = "macOS 27.0.1", h.cpu = "Apple M1 Max", h.cpu_vendor = "Apple";
    h.ram_total = static_cast<int64_t>(ram_gib * (1LL << 30)), h.disk_free = static_cast<int64_t>(500e9);
    in::Gpu g;
    g.vendor = "Apple", g.name = "Apple M1 Max", g.arch = "apple-m1-max";
    g.vram_total = h.ram_total / 100 * 78;
    h.gpus = {g};
    return h;
}

bool has_error(const in::Module& m, const in::Hardware& h) {
    for (const in::Note& n : m.check(h, m.recommend(h)))
        if (n.warn) return true;
    return false;
}

void test_module() {
    const in::Hardware h = mac(32);
    const in::Module* m = in::find_module(h);
    EXPECT(m && m->name() == "metal");
    if (!m) return;
    const in::Settings s = m->recommend(h);
    // coder-dmt-164 is not published yet: the recommendation is the model that installs
    EXPECT(s.model == "coder-iq1_m" && s.drafter == "mtp" && s.seqs == 1 && s.ctx == 32768);
    const in::CatalogEntry* chosen = in::find_entry(s.model);
    const in::CatalogEntry* head = in::find_entry(in::drafter_entry_id(s.drafter, s.model));
    EXPECT(chosen && head && in::pinned(*chosen) && in::pinned(*head));
    // the pre-tiled pair is taken as soon as both entries can be installed, and only then
    const auto only = [](std::set<std::string> ids) {
        return [ids = std::move(ids)](const std::string& id) { return ids.contains(id); };
    };
    EXPECT(in::metal_apple_model(only({"coder-dmt-164", "mtp-164"})) == "coder-dmt-164");
    EXPECT(in::metal_apple_model(only({"coder-dmt-164"})) == "coder-iq1_m");
    EXPECT(in::metal_apple_model(only({"mtp-164"})) == "coder-iq1_m");
    EXPECT(!has_error(*m, h) && has_error(*m, mac(16)));
    in::Hardware old = h;
    old.distro_name = "macOS 14.6";
    EXPECT(has_error(*m, old));
    in::Hardware linux_box = h;  // an Apple GPU name alone is not a Mac
    linux_box.os = "linux";
    EXPECT(!in::metal_apple_module().pick(linux_box));
}

bool holds(const in::Range& r, double v) { return r.lo <= v && v <= r.hi; }

// The pages offer what the module allows and the settings follow it, so the configuration the setup writes has the
// Mac's one agent at 32K and no "auto"
void test_limits() {
    const in::Hardware h = mac(32);
    const in::Module* m = in::find_module(h);
    EXPECT(m != nullptr);
    if (!m) return;
    const in::Limits l = m->limits();
    EXPECT(l.max_agents == 1 && !l.auto_context && l.contexts == std::vector<int>{32768});
    in::Settings s;  // the generic defaults: 2 agents, 256K, auto
    in::resolve_context(m, s);
    EXPECT(s.seqs == 1 && s.ctx == 32768 && !s.ctx_auto);
    const std::string conf = in::render(s, "/nonexistent");
    EXPECT(conf.contains("TONA_CTX='32768'") && conf.contains("TONA_CTX_AUTO='0'") && conf.contains("TONA_SEQS='1'"));
}

// The M1 Max 32 GB is the anchor: its range holds the measured rates, plus or minus 15 % with the drive known
void test_estimates() {
    in::Hardware h = mac(32);
    h.disk_speed = {true, 0, 0, 7550, "", 6700};  // what the setup probe measures on the anchor (metal_apple.cpp)
    const in::Module* m = in::find_module(h);
    EXPECT(m != nullptr);
    if (!m) return;
    const in::Settings s = m->recommend(h);
    const in::Estimate iq1 = m->estimate(h, s, "coder-iq1_m");
    EXPECT(holds(iq1.decode, 17.9) && holds(iq1.prefill, 136.2) && iq1.context_tokens == 16000);
    EXPECT(std::abs(iq1.width - 0.15) < 1e-9 && iq1.unknown.empty() && iq1.anchor == "M1 Max 32 GB");
    EXPECT(std::abs((iq1.decode.lo + iq1.decode.hi) / 2 - 17.9) < 0.5);
    EXPECT(holds(m->estimate(h, s, "coder-dmt-164").decode, 35.1));
    EXPECT(std::regex_match(in::speed_text(iq1), std::regex("1[45]–2[01] (tok|Token)/s")));
    // no drive speed measured: still the anchor's centre, a wider range, and it says what is missing
    const in::Estimate blind = m->estimate(mac(32), s, "coder-iq1_m");
    EXPECT(holds(blind.decode, 17.9) && blind.width > iq1.width && blind.unknown.size() == 1);
    // The anchor's own hardware with the default settings gives the anchor's rate, whatever else the drive test says: a
    // single sequential read (here 3 GB/s, a fraction of what the expert reads reach) must not slow the estimate
    in::Hardware slow_seq = mac(32);
    slow_seq.disk_speed = {true, 0, 0, 3000, ""};
    EXPECT(holds(m->estimate(slow_seq, s, "coder-iq1_m").decode, 17.9));
    EXPECT(holds(m->estimate(h, m->recommend(h), "coder-iq1_m").decode, 17.9) && iq1.width < 0.151);
    // another chip of the backend: plus or minus 30 %, and a lower centre with less memory bandwidth
    in::Hardware pro = h;
    pro.cpu = "Apple M4 Pro";
    const in::Estimate other = m->estimate(pro, s, "coder-iq1_m");
    EXPECT(std::abs(other.width - 0.30) < 1e-9 && other.decode.hi > 0);
    in::Hardware base = h;
    base.cpu = "Apple M1";
    EXPECT(m->estimate(base, s, "coder-iq1_m").decode.hi < iq1.decode.hi);
    // a chip not in the table, a model nobody measured on the Mac: no number
    in::Hardware future = h;
    future.cpu = "Apple M9";
    EXPECT(m->estimate(future, s, "coder-iq1_m").decode.hi == 0 &&
           !m->estimate(future, s, "coder-iq1_m").setup.empty());
    EXPECT(m->estimate(h, s, "ornith-1.5-35b-a3b").decode.hi == 0);
}

// engine/model/gguf.cpp opens part i of a split as <prefix of part 1>-<i>-of-<n>.gguf (tests/test_gguf.cpp)
void test_split_names() {
    static const std::regex re(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)");
    for (const in::CatalogEntry& e : in::embedded_catalog()) {
        if (e.files.size() < 2) continue;
        std::smatch m;
        const std::string first = e.files[0].name();
        EXPECT(std::regex_match(first, m, re));
        for (size_t i = 1; i < e.files.size(); i++)
            EXPECT(e.files[i].name() == std::format("{}-{:05}-of-{:05}.gguf", m[1].str(), i + 1, e.files.size()));
    }
}

void test_mac_entries() {
    const in::CatalogEntry* d = in::find_entry("coder-dmt-164");
    const in::CatalogEntry* ref = in::find_entry("coder-iq1_m");
    EXPECT(d && ref && d->files.size() == 2 && !in::pinned(*d) && in::pinned(*ref));
    if (!d || !ref || d->files.size() != 2) return;
    const in::ModelFile& p2 = d->files[1];
    EXPECT(p2.name() == "coder-DMT-164e-00002-of-00002.gguf" && p2.sha256 == ref->files[1].sha256);
    EXPECT(in::model_url({}, *d, p2) == in::model_url({}, *ref, ref->files[1]));  // the pinned ISTA file
    EXPECT(in::drafter_entry_id("mtp", "coder-dmt-164") == "mtp-164" &&
           in::drafter_entry_id("dflash", "coder-dmt-164") == "dflash-q8_0");
    const in::CatalogEntry* mtp = in::find_entry("mtp-164");
    EXPECT(mtp && mtp->kind == "drafter" && !in::pinned(*mtp));
}

void test_unpublished_is_refused(const fs::path& dir) {
    const in::CatalogEntry* d = in::find_entry("coder-dmt-164");
    TestServer srv;
    in::ModelSource src;
    src.base = "http://127.0.0.1:" + std::to_string(srv.port());
    src.fetch.allow_http_loopback = true;
    const std::string err = d ? in::fetch_model(*d, (dir / "m").string(), src, "", [](const std::string&) {}) : "";
    EXPECT(err.contains("not published") && !fs::exists(dir / "m"));
}

std::string sha_of(const std::string& s) {
    tona::update::Sha256 h;
    h.update(s.data(), s.size());
    const auto d = h.finish();
    return tona::update::to_hex(d.data(), d.size());
}

// Part 2 from another repository and revision, saved under part 1's prefix; an existing download under its
// repository name is linked
void test_file_from_other_repo(const fs::path& dir) {
    const std::string a(5000, 'a'), b(7000, 'b'), ra(40, '1'), rb(40, '2');
    in::CatalogEntry e;
    e.id = "two", e.kind = "model", e.repo = "org/mine", e.revision = ra;
    e.files = {{"m-00001-of-00002.gguf", a.size(), sha_of(a), "", "", ""},
               {"Q/up-00002-of-00002.gguf", b.size(), sha_of(b), "up/stream", rb, "m-00002-of-00002.gguf"}};
    TestServer srv;
    srv.set("/org/mine/resolve/" + ra + "/m-00001-of-00002.gguf", Route(a));
    srv.set("/up/stream/resolve/" + rb + "/Q/up-00002-of-00002.gguf", Route(b));
    in::ModelSource src;
    src.base = "http://127.0.0.1:" + std::to_string(srv.port());
    src.fetch.allow_http_loopback = true;
    const std::string models = (dir / "models").string();
    EXPECT(in::fetch_model(e, models, src, "", [](const std::string&) {}).empty());
    EXPECT(read_file((dir / "models/two/m-00002-of-00002.gguf").string()) == b && in::model_present(models, e));
    fs::create_directories(dir / "have");
    std::ofstream(dir / "have/up-00002-of-00002.gguf", std::ios::binary) << b;
    std::ofstream(dir / "have/m-00001-of-00002.gguf", std::ios::binary) << a;
    const std::string linked = (dir / "linked").string();
    EXPECT(in::fetch_model(e, linked, src, (dir / "have").string(), [](const std::string&) {}).empty());
    EXPECT(fs::is_symlink(dir / "linked/two/m-00002-of-00002.gguf") && in::model_present(linked, e));
}

void test_mac_plan(const fs::path& dir, const std::string& repo) {
    in::Target t = in::release_target((dir / "root").string());
    const in::Settings s = in::metal_apple_module().recommend(mac(32));
    const std::vector<in::Step> st = in::plan_release_steps({}, s, t, {});
    EXPECT(st.size() > 3 && st[1].title.contains("coder-iq1_m") && st[2].title.contains("mtp-q8_0"));
    for (const in::Step& step : st) EXPECT(!step.title.contains("coder-dmt-164"));
    const in::UnitSpec u = in::unit_spec(s, t);
    EXPECT(u.drafter_flag == "--mtp" && u.drafter_file.contains("/mtp-q8_0/"));
    // a checkout reads the catalog through tools/get_model.py --list
    EXPECT(in::render(s, repo).contains("work/models/mtp-q8_0/"));
}

}  // namespace

int main(int argc, char** argv) {
    return tona::test::guard("mac", [&] {
        const std::string repo = argc > 1 ? argv[1] : ".";
        return run("mac", [&] {
            TempDir dir("mac");
            test_module();
            test_estimates();
            test_limits();
            test_split_names();
            test_mac_entries();
            test_unpublished_is_refused(dir.path());
            test_file_from_other_repo(dir.path());
            test_mac_plan(dir.path(), repo);
        });
    });
}
