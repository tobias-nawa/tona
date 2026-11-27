#include "core/plan.h"

#include "core/sys.h"
#include "nlohmann/json.hpp"

namespace tona::install {

std::vector<std::string> models_of(const Settings& s) {
    std::vector<std::string> out = {s.model};
    for (const std::string& m : s.models)
        if (m != s.model) out.push_back(m);
    return out;
}

std::vector<ModelInfo> load_catalog(const std::string& repo) {
    // one source for catalog and checksums: toolchain/models.toml through tools/get_model.py
    const RunResult r = run("python3 -I " + quote(repo + "/tools/get_model.py") + " --list");
    std::vector<ModelInfo> out;
    if (r.status != 0) return out;
    const auto j = nlohmann::json::parse(r.out, nullptr, false);
    if (!j.is_array()) return out;
    for (const auto& e : j) {
        ModelInfo m;
        m.id = e.value("id", ""), m.title = e.value("title", ""), m.kind = e.value("kind", "model");
        m.license = e.value("license", ""), m.quality = e.value("quality", "");
        m.first_file = e.value("first_file", "");
        m.size_gb = e.value("size_gb", 0.0);
        m.tested = e.value("tested", false), m.present = e.value("present", false);
        out.push_back(m);
    }
    return out;
}

}  // namespace tona::install
