// The step titles are English identifiers of the plan (--print, --json, the tests); the pages show them with the model
// names of the model page.
#include <algorithm>
#include <format>

#include "core/describe.h"
#include "core/space.h"
#include "core/sys.h"

namespace tona::install {

namespace {

struct Text {
    const char* prefix;
    const char* en;
};

// A title that starts with `prefix`; where the prefix ends with a space, the rest of the title (a folder) follows the
// text
constexpr Text TEXTS[] = {
    {"Fetch and verify ROCm", "Download and verify the ROCm toolchain (~7 GB, no root)"},
    {"Metal toolchain", "Download the Metal toolchain for the kernels (~700 MB)"},
    {"Write settings to work/tona.conf", "Write the settings"},
    {"Build the engine", "Build the engine"},
    {"Trial run", "Trial run: start the server, measure the speed (~2 min), stop it"},
    {"Set up the service", "Set up the service (starts at login)"},
    {"Command tona in", "Command tona in ~/.local/bin"},
    {"API key", "API key"},
    {"Settings for the command tona", "Settings for the command tona"},
    {"Write the service and enable autostart", "Write the service and start it at login"},
    {"Write the service (no autostart", "Write the service (no autostart: tona start)"},
    {"Start the server at every login", "Start the server at every login (and now)"},
    {"No autostart: start the server", "No autostart (tona start)"},
    {"Update checks: on", "Update checks: on, once a day"},
    {"Update checks: off", "Update checks: off"},
    {"Fetch the signed release", "Download, verify and install the signed release"},
};

// "Fetch and verify [model|drafter] <id> [(SHA-256)]" and "Expert profile <id>": the id, else empty
std::string id_after(const std::string& title, const std::string& prefix) {
    if (!title.starts_with(prefix)) return {};
    std::string id = title.substr(prefix.size());
    if (const size_t p = id.find(" ("); p != std::string::npos) id.resize(p);
    return id;
}

// "/home/me/.local/bin" as "~/.local/bin"
std::string short_path(const std::string& path) {
    const std::string h = home();
    return !h.empty() && path.starts_with(h + "/") ? "~" + path.substr(h.size()) : path;
}

std::string model_label(const std::string& id, bool with_size) {
    const offer::ModelCard* c = offer::model_card(id);
    if (!c) return id;
    return with_size ? std::format("{} · {}", c->name, size_text(static_cast<double>(entry_bytes(c->id)))) : c->name;
}

}  // namespace

std::string step_text(const Step& st, const std::string& models_dir) {
    const auto* t =
        std::find_if(std::begin(TEXTS), std::end(TEXTS), [&](const Text& x) { return st.title.starts_with(x.prefix); });
    if (t != std::end(TEXTS)) {
        const std::string prefix = t->prefix;
        return std::string(t->en) + (prefix.ends_with(' ') ? short_path(st.title.substr(prefix.size())) : "");
    }
    for (const std::string prefix : {"Fetch and verify model ", "Fetch and verify "})
        if (const std::string id = id_after(st.title, prefix); !id.empty()) {
            const bool model = offer::model_card(id) != nullptr;
            if (models_dir.empty())
                return std::string(model ? "Download and verify: " : "Download and verify the drafter: ") +
                       model_label(id, true);
            if (!model) return "";
            const bool here = model_split(id, models_dir).download == 0;
            return std::string(here ? "Verify: " : "Download and verify: ") + offer::model_card(id)->name + " · " +
                   size_text(static_cast<double>(model_bytes(id))) + (here ? " (already here)" : "");
        }
    if (const std::string id = id_after(st.title, "Expert profile "); !id.empty())
        return std::string("Expert profile: ") + model_label(id, false);
    return st.title;
}

}  // namespace tona::install
