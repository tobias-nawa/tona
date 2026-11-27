// Where releases come from and where they are installed. The repository is named in one place only.
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace tona::update {

// The one place that names the repository. While it is private, downloads without a login answer 404.
inline constexpr std::string_view kReleaseRepo = "tobias-nawa/tona";

struct UpdateBase {
    std::string url;                   // "<base>/latest/download/manifest.json" and "<base>/download/v<ver>/<asset>"
    bool allow_http_loopback = false;  // test servers on 127.0.0.1 only

    std::string manifest_url() const { return url + "/latest/download/manifest.json"; }
    std::string signature_url() const { return manifest_url() + ".minisig"; }
    std::string asset_url(const std::string& version, const std::string& asset) const {
        return url + "/download/v" + version + "/" + asset;
    }
};

// Accepts https://host[:port]/path, or http://127.0.0.1:port/path (tests). No user info, query, fragment or
// trailing slash; anything else is nullopt.
std::optional<UpdateBase> parse_base(std::string_view text);

// TONA_UPDATE_BASE if set (an invalid value is an error, never a silent fallback to something else), else GitHub.
// `error` receives the reason when the variable is set but unusable.
std::optional<UpdateBase> resolve_base(std::string& error);

// Install root: TONA_INSTALL_DIR, else $XDG_DATA_HOME/tona, else ~/.local/share/tona. Empty if none can be determined.
// The one definition: tona uses it to find out whether it runs from a release installation.
std::string install_root();

// State directory of tona (server.pid, server.log): $XDG_STATE_HOME/tona, else ~/.local/state/tona
std::string state_dir();

}  // namespace tona::update
