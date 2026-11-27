#include "update/config.h"

#include <cstdlib>

#include "backend/env.h"
#include "update/url.h"

namespace tona::update {

std::optional<UpdateBase> parse_base(std::string_view text) {
    const auto url = parse_url(text);
    if (!url || text.contains('?') || text.ends_with('/')) return std::nullopt;
    const bool loopback = url->scheme == "http" && url->host == "127.0.0.1" && text.starts_with("http://127.0.0.1:");
    if (url->scheme != "https" && !loopback) return std::nullopt;
    return UpdateBase{std::string(text), loopback};
}

std::optional<UpdateBase> resolve_base(std::string& error) {
    if (const char* env = tona::env::text("TONA_UPDATE_BASE")) {
        auto base = parse_base(env);
        if (!base) error = "TONA_UPDATE_BASE must be https://host/path or http://127.0.0.1:port/path";
        return base;
    }
    return parse_base("https://github.com/" + std::string(kReleaseRepo) + "/releases");
}

std::string install_root() {
    if (const char* dir = tona::env::text("TONA_INSTALL_DIR")) return dir;
    if (const char* data = tona::env::text("XDG_DATA_HOME")) return std::string(data) + "/tona";
    const char* home = tona::env::text("HOME");
    return home ? std::string(home) + "/.local/share/tona" : std::string();
}

std::string state_dir() {
    if (const char* state = tona::env::text("XDG_STATE_HOME")) return std::string(state) + "/tona";
    const char* home = tona::env::text("HOME");
    return home ? std::string(home) + "/.local/state/tona" : std::string();
}

}  // namespace tona::update
