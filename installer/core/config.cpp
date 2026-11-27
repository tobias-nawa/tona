#include "core/config.h"

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <sstream>
#include <type_traits>

#include "core/catalog.h"
#include "core/flow.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::install {

namespace {

std::string model_file(const std::string& repo, const std::string& id) {
    for (const ModelInfo& m : load_catalog(repo))
        if (m.id == id) return (fs::path(repo) / "work/models" / id / m.first_file).string();
    return {};
}

bool write_text_atomic(const std::string& path, const std::string& text) {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    const std::string tmp = path + ".new";
    {
        std::ofstream f(tmp);
        if (!(f << text)) return false;
    }
    fs::rename(tmp, path, ec);
    return !ec;
}

std::string unquote(std::string v) {
    if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front())
        v = v.substr(1, v.size() - 2);
    return v;
}

Values parse(const std::string& text) {
    std::istringstream in(text);
    Values v;
    for (std::string line; std::getline(in, line);) {
        line = trim(line);
        const auto p = line.find('=');
        if (line.empty() || line[0] == '#' || p == std::string::npos) continue;
        v[line.substr(0, p)] = unquote(line.substr(p + 1));
    }
    return v;
}

template <typename T>
void read_number(const Values& v, const char* key, T& out) {
    const auto it = v.find(key);
    if (it == v.end() || it->second.empty()) return;
    char* end = nullptr;
    const double d = std::strtod(it->second.c_str(), &end);
    if (end != it->second.c_str()) out = static_cast<T>(d);  // not a number: keep the default
}

// What the pages asked (flow.h, room.h)
void read_page_choices(const Values& v, Settings& s) {
    if (v.contains("TONA_MODELS")) s.models = split_list(v.at("TONA_MODELS"));
    if (v.contains("TONA_LANGS")) s.languages = split_list(v.at("TONA_LANGS"));
    if (v.contains("TONA_USE")) s.use = v.at("TONA_USE");
    if (v.contains("TONA_PRIORITY")) s.priority = v.at("TONA_PRIORITY");
    if (v.contains("TONA_CTX_AUTO")) s.ctx_auto = v.at("TONA_CTX_AUTO") != "0";
}

void read_choices(const Values& v, Settings& s) {
    read_page_choices(v, s);
    if (v.contains("TONA_EFFORT")) s.effort = v.at("TONA_EFFORT");
    if (v.contains("TONA_OEA_K0")) s.oea = v.at("TONA_OEA_K0") != "0";
    if (v.contains("TONA_CLOCK_HOLD")) s.clock_hold = v.at("TONA_CLOCK_HOLD") == "1" || v.at("TONA_CLOCK_HOLD") == "2";
    if (v.contains("TONA_ROLE_EFFORT")) s.role_effort = v.at("TONA_ROLE_EFFORT");
    if (v.contains("TONA_API_KEY_FILE")) s.api_key = v.at("TONA_API_KEY_FILE") != "none";
    if (v.contains("TONA_DRAFTER")) {
        const std::string& d = v.at("TONA_DRAFTER");
        // "--mtp <file>" in a checkout; the plain mode in a release installation, whose unit holds the files
        s.drafter = d.starts_with("--mtp") || d == "mtp"         ? "mtp"
                    : d.starts_with("--dflash") || d == "dflash" ? "dflash"
                                                                 : "none";
    }
    if (v.contains("TONA_MODEL_ID")) s.model = v.at("TONA_MODEL_ID");
    if (v.contains("TONA_MODEL"))  // work/models/<id>/<file>
        if (const fs::path m = v.at("TONA_MODEL"); m.has_parent_path()) s.model = m.parent_path().filename().string();
}

}  // namespace

std::string conf_path(const std::string& repo) { return (fs::path(repo) / "work/tona.conf").string(); }

std::string render(const Settings& s, const std::string& repo) {
    std::string o = "# written by tona setup; to change: run tona setup, or edit here and restart the server\n";
    auto kv = [&](const std::string& k, const std::string& v) { o += k + "=" + quote(v) + "\n"; };
    kv("TONA_MODEL", model_file(repo, s.model));
    const std::string drafter = drafter_entry_id(s.drafter, s.model);
    kv("TONA_DRAFTER", drafter.empty() ? "" : "--" + s.drafter + " " + model_file(repo, drafter));
    kv("TONA_PROFILE", (fs::path(repo) / "work/profile" / (s.model + ".bin")).string());
    kv("TONA_PORT", std::to_string(s.port));
    kv("TONA_CTX", std::to_string(s.ctx));
    kv("TONA_CTX_AUTO", s.ctx_auto ? "1" : "0");
    kv("TONA_SEQS", std::to_string(s.seqs));
    kv("TONA_RAM_GB", std::format("{:g}", s.ram_gb));
    kv("TONA_VRAM_GB", std::format("{:g}", s.vram_gb));
    kv("TONA_EFFORT", s.effort);
    kv("TONA_OEA_K0", s.oea ? "7" : "0");
    if (!s.role_effort.empty()) kv("TONA_ROLE_EFFORT", s.role_effort);
    kv("TONA_CLOCK_HOLD", s.clock_hold ? "1" : "0");
    kv("TONA_API_KEY_FILE", s.api_key ? home() + "/.config/tona/api-key" : "none");
    kv("TONA_PROMPT_CACHE_GB", std::format("{:g}", s.prompt_cache_gb));
    kv("TONA_MODELS", join_list(models_of(s)));
    kv("TONA_LANGS", join_list(s.languages));
    kv("TONA_USE", s.use);
    kv("TONA_PRIORITY", s.priority);
    return o;
}

Values parse_values(const std::string& text) { return parse(text); }
Values read_values(const std::string& path) { return parse(read_file(path)); }

std::string set_values(const std::string& path, const Values& changes) {
    std::string out;
    Values pending = changes;
    std::istringstream in(read_file(path));
    for (std::string line; std::getline(in, line);) {
        const auto p = line.find('=');
        const std::string key = p == std::string::npos || line.empty() || line[0] == '#' ? "" : trim(line.substr(0, p));
        if (const auto it = pending.find(key); !key.empty() && it != pending.end()) {
            out += key + "=" + quote(it->second) + "\n";
            pending.erase(it);
        } else {
            out += line + "\n";
        }
    }
    for (const auto& [k, v] : pending) out += k + "=" + quote(v) + "\n";
    return write_text_atomic(path, out) ? "" : path + ": cannot write";
}

Settings load(const std::string& path, Settings s) {
    const Values v = parse(read_file(path));
    read_number(v, "TONA_PORT", s.port);
    read_number(v, "TONA_CTX", s.ctx);
    read_number(v, "TONA_SEQS", s.seqs);
    read_number(v, "TONA_RAM_GB", s.ram_gb);
    read_number(v, "TONA_VRAM_GB", s.vram_gb);
    read_number(v, "TONA_PROMPT_CACHE_GB", s.prompt_cache_gb);
    read_choices(v, s);
    return s;
}

bool save(const Settings& s, const std::string& repo) {
    const std::string path = conf_path(repo);
    // Keys the installer does not own (the client's own, e.g. TONA_PRIORITY) survive a save
    const Values old = read_values(path), mine = parse(render(s, repo));
    std::string text = render(s, repo);
    for (const auto& [k, v] : old)
        if (!mine.contains(k)) text += k + "=" + quote(v) + "\n";
    return write_text_atomic(path, text);
}

}  // namespace tona::install
