#include "core/steps.h"

#include <filesystem>
#include <format>

#include "core/catalog.h"
#include "core/config.h"
#include "core/flow.h"
#include "core/module.h"
#include "core/release_steps.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::install {

namespace {

bool model_present(const std::vector<ModelInfo>& cat, const std::string& id) {
    for (const ModelInfo& m : cat)
        if (m.id == id) return m.present;
    return false;
}

std::string get_model(const std::string& id, const std::string& from) {
    std::string c = "python3 -I tools/get_model.py " + id;
    if (!from.empty()) c += " --from " + quote(from);
    return c;
}

// ~/.local/bin/tona -> <repo>/work/build/engine/tona: a program, so it works in every shell (PATH: shell_path.h)
bool tona_linked(const std::string& repo) {
    std::error_code ec;
    const fs::path link = fs::path(home()) / ".local/bin/tona";
    return fs::is_symlink(link, ec) && fs::read_symlink(link, ec) == fs::path(repo) / "work/build/engine/tona";
}

int link_tona(const std::string& repo, const LineFn& out) {
    const std::string bin = home() + "/.local/bin";
    link_commands(bin, {{"tona", repo + "/work/build/engine/tona"}}, out);
    return 0;
}

}  // namespace

std::vector<Step> plan_steps(const Hardware& h, const Settings& s, const Target& t) {
    if (t.release) return plan_release_steps(h, s, t, ReleaseSources{});
    const std::string& repo = t.repo;
    std::vector<Step> st;
    const Module* mod = find_module(h);
    if (mod)
        for (Step& tc : mod->toolchain(repo)) st.push_back(std::move(tc));
    const std::vector<ModelInfo> cat = load_catalog(repo);
    // An existing download is verified and linked instead of fetched again (--model-from, the drafter's old folder).
    // The model the server starts with comes first; the drafter belongs to it.
    for (const std::string& id : models_of(s))
        st.push_back({"Fetch and verify model " + id + " (SHA-256)",
                      get_model(id, id == s.model ? s.model_from : ""),
                      model_present(cat, id),
                      {}});
    if (s.drafter != "none") {
        const std::string id = drafter_entry_id(s.drafter, s.model);
        st.push_back(
            {"Fetch and verify drafter " + id,
             get_model(id, fs::exists(fs::path(repo) / "work/models/draft") ? repo + "/work/models/draft" : ""),
             model_present(cat, id),
             {}});
    }
    // Only the reference backend's models have an expert profile (Metal keeps every expert in the mapping)
    for (const std::string& id : models_of(s)) {
        const std::string prof = "work/profile/" + id + ".bin";
        if (fs::exists(fs::path(repo) / "toolchain/profiles" / (id + ".bin")))
            st.push_back({"Expert profile " + id,
                          std::format("mkdir -p work/profile && cp toolchain/profiles/{}.bin {}", id, prof),
                          fs::exists(fs::path(repo) / prof),
                          {}});
    }
    st.push_back({"Write settings to work/tona.conf", "(written by tona setup)", false,
                  [s = s, repo = repo](const LineFn& out) {  // copies (release_steps.cpp)
                      if (!save(s, repo)) return 1;
                      out("written: " + conf_path(repo));
                      return 0;
                  }});
    st.push_back({"Build the engine", "just build", false, {}});
    const std::string url = std::format("http://127.0.0.1:{}", s.port);
    // never touch a server that is already running (e.g. the installed service). Loading takes 1 to 3 min; after
    // 10 min (the limit of every wait in the project) the run stops the server and fails with the log's place
    st.push_back(
        {"Trial run: start the server, measure the speed (~2 min), stop it",
         std::format(
             "if curl -sf -m 1 {0}/status >/dev/null; then echo 'a server is already running on port {1}, "
             "trial run skipped'; exit 0; fi; just start && end=$(($(date +%s) + 600)) && until curl -sf -m 1 "
             "{0}/status | jq -e .ready >/dev/null; do if [ $(date +%s) -ge $end ]; then echo 'server not ready "
             "after 10 min, trial run aborted (log: just start printed it)'; just stop; exit 1; fi; sleep 1; done && "
             "TONA_API_KEY_FILE={2} python3 -I scripts/bench-set.py {0}; r=$?; just stop; exit $r",
             url, s.port, s.api_key ? quote(home() + "/.config/tona/api-key") : "none"),
         false,
         {}});
    if (s.service) st.push_back({"Set up the service (starts at login)", "just install-service", false, {}});
    if (s.shell)
        st.push_back({"Command tona in ~/.local/bin", "(link to " + repo + "/work/build/engine/tona)",
                      tona_linked(repo), [repo = repo](const LineFn& out) { return link_tona(repo, out); }});
    return st;
}

int run_step(const Step& st, const std::string& repo, const LineFn& on_line) {
    if (st.action) return st.action(on_line);
    return run("cd " + quote(repo) + " && " + st.cmd, on_line).status;
}

}  // namespace tona::install
