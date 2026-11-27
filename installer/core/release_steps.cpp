#include "core/release_steps.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>

#include "core/autostart.h"
#include "core/catalog.h"
#include "core/flow.h"
#include "core/shell_path.h"
#include "core/sys.h"
#include "update/config.h"
#include "update/crypto.h"
#include "update/fsutil.h"
#include "update/install.h"
#include "update/keys.h"
#include "update/state.h"

namespace fs = std::filesystem;

namespace tona::install {

std::string write_atomic(const fs::path& path, const std::string& text, mode_t mode) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) return path.parent_path().string() + ": " + ec.message();
    const fs::path tmp = path.string() + ".new";
    ::unlink(tmp.c_str());
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, mode);
    if (fd < 0) return tmp.string() + ": cannot create";
    const bool ok = ::write(fd, text.data(), text.size()) == static_cast<ssize_t>(text.size()) && ::fsync(fd) == 0;
    ::close(fd);
    if (!ok) return tmp.string() + ": cannot write";
    fs::rename(tmp, path, ec);
    return ec ? path.string() + ": " + ec.message() : "";
}

namespace {

std::string key_path() { return home() + "/.config/tona/api-key"; }

std::optional<update::UpdateEnv> default_update_env(const Target& t, std::string& error) {
    update::UpdateEnv env;
    env.layout.root = t.root;
    env.keys = update::release_keys();
    const auto base = update::resolve_base(error);
    if (!base) return std::nullopt;
    env.base = *base;
    env.fetch.allow_http_loopback = base->allow_http_loopback;
    return env;
}

int install_signed_release(const ReleaseSources& src, const Target& t, const LineFn& out) {
    std::string error;
    const auto env = src.update ? src.update : default_update_env(t, error);
    if (!env) {
        out("Nothing installed: " + error);
        return 1;
    }
    const update::CheckOutcome check = update::check_for_update(*env);
    if (check.kind == update::CheckOutcome::Kind::Failed) {
        out("Nothing installed: " + check.error);
        return 1;
    }
    const update::Manifest& m = check.release.manifest;
    out(std::format("release {} (sequence {}), {} MB, signature and manifest verified", m.version, m.seq,
                    m.total_bytes() / 1000000));
    update::InstallHooks hooks;
    int shown = -1;
    hooks.progress = [&](uint64_t done, uint64_t total) {
        const int pct = total ? static_cast<int>(done * 20 / total) : 0;  // a line per 5 %
        if (pct != shown) out(std::format("  downloaded {} of {} MB", done / 1000000, total / 1000000)), shown = pct;
    };
    const update::InstallOutcome in = update::install_release(*env, check.release, hooks);
    if (!in.ok) {
        out("Nothing installed: " + in.error);
        return 1;
    }
    out("installed version " + m.version + " as " + (t.root + "/current"));
    return 0;
}

// Init-captures make non-const copies: a closure holding a const object copies it when moved, and that copy may
// throw (bugprone-exception-escape, move constructors)
Step model_step(const Target& t, const Settings& s, const std::string& id, const ReleaseSources& src) {
    const CatalogEntry* e = find_entry(id);
    const std::string title = "Fetch and verify " + id + " (SHA-256)";
    if (!e)
        return {title, "(not in the catalog of this installer)", false, [id = id](const LineFn& out) {
                    out("unknown model: " + id);
                    return 1;
                }};
    return {title, std::format("(download of {:.1f} GB at revision {}, resumable)", e->size_gb(), e->revision),
            model_present(t.models_dir, *e),
            [e, dir = t.models_dir, from = src.models, local = s.model_from](const LineFn& out) {
                const std::string err = fetch_model(*e, dir, from, local, out);
                if (!err.empty()) out("Failed: " + err);
                return err.empty() ? 0 : 1;
            }};
}

Step profile_step(const Target& t, const std::string& model) {
    const ProfileBlob* p = find_profile(model);
    const fs::path dest = fs::path(t.root) / "profiles" / (model + ".bin");
    std::error_code ec;
    const bool done = !p || (fs::exists(dest) && fs::file_size(dest, ec) == p->bytes.size());
    return {"Expert profile " + model, "(written from the installer to " + dest.string() + ")", done,
            [p, dest = dest](const LineFn& out) {
                const std::string err = write_atomic(dest, std::string(p->bytes.begin(), p->bytes.end()), 0644);
                if (!err.empty()) out("Failed: " + err);
                return err.empty() ? 0 : 1;
            }};
}

Step key_step(const Settings& s) {
    std::error_code ec;
    const bool done = !s.api_key || (fs::exists(key_path()) && fs::file_size(key_path(), ec) > 0);
    return {"API key", "(random key, mode 0600, " + key_path() + ")", done, [](const LineFn& out) {
                const std::string err = ensure_api_key(key_path());
                if (!err.empty()) out("Failed: " + err);
                return err.empty() ? 0 : 1;
            }};
}

Step unit_step(const Settings& s, const Target& t) {
    const std::string file = t.unit_dir + "/" + kUnitName + ".service";
    const std::string title = s.service ? "Write the service and enable autostart (starts now and at every login)"
                                        : "Write the service (no autostart: start it with tona start)";
    return {title, "(" + file + ", then systemctl --user daemon-reload" + (s.service ? ", enable --now)" : ")"), false,
            [s = s, t = t, file = file](const LineFn& out) {
                const std::string err = write_unit_file(t.root, t.unit_dir);
                if (!err.empty()) {
                    out("Failed: " + err);
                    return 1;
                }
                out("written: " + file);
                const std::string enable =
                    s.service ? std::string(" && systemctl --user enable --now ") + kUnitName : "";
                return run("systemctl --user daemon-reload" + enable, out).status;
            }};
}

// macOS: no systemd. With "At login" a LaunchAgent runs `tona start --foreground` at every login (autostart.h);
// without it an agent of an earlier installation is removed.
Step login_agent_step(const Settings& s, const Target& t) {
    const std::string title =
        s.service ? "Start the server at every login (and now)" : "No autostart: start the server with tona start";
    return {title, "(" + launch_agent_path() + ")", false, [s = s, t = t](const LineFn& out) {
                const std::string err = set_autostart(true, t.root, s.service, true, out);
                if (!err.empty()) out("Failed: " + err);
                return err.empty() ? 0 : 1;
            }};
}

// ~/.local/bin/tona points into current/, so they follow every update
Step links_step(const Target& t) {
    return {"Command tona in " + t.bin_dir, "(links to " + t.root + "/current/bin/)", false,
            [t = t](const LineFn& out) {
                std::vector<CommandLink> links;
                links.push_back({"tona", t.root + "/current/bin/tona"});
                link_commands(t.bin_dir, links, out);
                return 0;
            }};
}

Step conf_step(const Settings& s) {
    const std::string path = home() + "/.config/tona/tona.conf";
    return {"Settings for the command tona", "(" + path + ")", false, [s = s, path = path](const LineFn& out) {
                const std::string err = write_atomic(path, render_release_conf(s), 0644);
                out(err.empty() ? "written: " + path : "Failed: " + err);
                return err.empty() ? 0 : 1;
            }};
}

// The switch key c of tona sets (Updater::set_checks): checks_enabled in state.json, the highest accepted seq kept
Step update_checks_step(const Settings& s, const Target& t) {
    const bool on = s.update_checks;
    return {std::string("Update checks: ") + (on ? "on, once a day" : "off"), "(" + t.root + "/state.json)", false,
            [on = on, layout = update::Layout{t.root}](const LineFn& out) {
                update::StateResult st = update::load_state(layout);
                if (!st.ok) {
                    out("Failed: " + st.error);
                    return 1;
                }
                st.state.checks_enabled = on;
                std::string err;
                try {
                    update::ensure_private_root(layout);
                    err = update::save_state(layout, st.state);
                } catch (const std::exception& e) {
                    err = e.what();
                }
                out(err.empty() ? std::string("update checks ") + (on ? "on" : "off") : "Failed: " + err);
                return err.empty() ? 0 : 1;
            }};
}

}  // namespace

UnitSpec unit_spec(const Settings& s, const Target& t) {
    UnitSpec u;
    u.root = t.root;
    if (const CatalogEntry* m = find_entry(s.model)) u.model_file = (model_dir(t.models_dir, *m) / m->files[0].name());
    if (const CatalogEntry* d = find_entry(drafter_entry_id(s.drafter, s.model))) {
        u.drafter_flag = "--" + s.drafter;
        u.drafter_file = model_dir(t.models_dir, *d) / d->files[0].name();
    }
    u.profile = t.root + "/profiles/" + s.model + ".bin";
    u.prompt_cache = t.cache_dir;
    u.api_key_file = s.api_key ? key_path() : "";
    // other PCs only with a key: a server open to the network without one would answer anyone
    if (s.network && s.api_key) u.host = "0.0.0.0";
    u.port = s.port, u.ctx = s.ctx, u.seqs = s.seqs;
    u.ram_gb = s.ram_gb, u.vram_gb = s.vram_gb, u.prompt_cache_gb = s.prompt_cache_gb;
    u.oea = s.oea, u.role_effort = s.role_effort, u.clock_hold = s.clock_hold;
    return u;
}

std::string write_unit_file(const std::string& root, const std::string& unit_dir) {
    return write_atomic(fs::path(unit_dir) / (std::string(kUnitName) + ".service"), render_unit(root), 0644);
}

std::string render_release_conf(const Settings& s) {
    // The whole of Settings: the command tona reads the port, effort and key file, and `tona start` starts the server
    // from the rest
    return std::format(
        "# written by tona setup; the one file of the settings: tona start reads it and starts the server with them.\n"
        "TONA_PORT={}\nTONA_EFFORT={}\nTONA_API_KEY_FILE={}\n"
        "TONA_MODEL_ID={}\nTONA_DRAFTER={}\nTONA_CTX={}\nTONA_CTX_AUTO={}\nTONA_SEQS={}\nTONA_RAM_GB={:g}\nTONA_VRAM_"
        "GB={:g}\n"
        "TONA_OEA_K0={}\nTONA_CLOCK_HOLD={}\nTONA_PROMPT_CACHE_GB={:g}\nTONA_MODELS={}\nTONA_LANGS={}\n"
        "TONA_USE={}\nTONA_PRIORITY={}\n{}",
        s.port, s.effort, s.api_key ? key_path() : "none", s.model, s.drafter, s.ctx, s.ctx_auto ? 1 : 0, s.seqs,
        s.ram_gb, s.vram_gb, s.oea ? 7 : 0, s.clock_hold ? 1 : 0, s.prompt_cache_gb, join_list(models_of(s)),
        join_list(s.languages), s.use, s.priority,
        s.role_effort.empty() ? "" : "TONA_ROLE_EFFORT=" + s.role_effort + "\n");
}

void link_commands(const std::string& bin_dir, const std::vector<CommandLink>& links, const LineFn& out) {
    std::error_code ec;
    fs::create_directories(bin_dir, ec);
    for (const CommandLink& l : links) {
        const fs::path link = fs::path(bin_dir) / l.name;
        if (fs::exists(fs::symlink_status(link, ec)) && !fs::is_symlink(link)) {
            out(link.string() + " exists and is not a link: left alone");
            continue;
        }
        fs::remove(link, ec);
        fs::create_symlink(l.target, link, ec);
        out(ec ? link.string() + ": " + ec.message() : "linked  " + link.string());
    }
}

std::vector<std::string> path_hint(const std::string& bin_dir, const std::string& path_env) {
    if (on_path(path_env, bin_dir)) return {};
    return hint_lines(bin_dir, home());
}

std::string ensure_api_key(const std::string& path) {
    std::error_code ec;
    if (fs::exists(path) && fs::file_size(path, ec) > 0) return "";
    uint8_t raw[32];
    update::random_bytes(raw, sizeof raw);
    return write_atomic(path, update::to_hex(raw, sizeof raw), 0600);
}

Target release_target(const std::string& root) {
    Target t;
    t.release = true;
    t.root = root.empty() ? update::install_root() : root;
    t.models_dir = t.root + "/models";
    t.cache_dir = t.root + "/prompt-cache";
    t.unit_dir = home() + "/.config/systemd/user";
#ifdef __APPLE__
    t.macos = true;
#endif
    t.bin_dir = home() + "/.local/bin";
    return t;
}

std::vector<Step> plan_release_steps(const Hardware&, const Settings& s, const Target& t, const ReleaseSources& src) {
    std::vector<Step> st;
    const update::Layout layout{t.root};
    st.push_back({"Fetch the signed release, verify it and install it into " + t.root,
                  "(minisign signature, then SHA-256 of every file; installs versions/<v> and current)",
                  fs::exists(layout.current()),
                  [src = src, t = t](const LineFn& out) { return install_signed_release(src, t, out); }});
    for (const std::string& id : models_of(s)) st.push_back(model_step(t, s, id, src));
    if (s.drafter != "none") st.push_back(model_step(t, s, drafter_entry_id(s.drafter, s.model), src));
    for (const std::string& id : models_of(s)) st.push_back(profile_step(t, id));
    st.push_back(key_step(s));
    st.push_back(conf_step(s));
    st.push_back(t.macos ? login_agent_step(s, t) : unit_step(s, t));
    if (s.shell) st.push_back(links_step(t));
    return st;
}

// The tasks after the installation that the user ticks on the Finish screen (or names on the command line)
std::vector<Step> finish_steps(const Settings& s, const Target& t, const PathEnv& env) {
    std::vector<Step> st;
    const std::string bin = t.bin_dir.empty() ? env.home + "/.local/bin" : t.bin_dir;
    if (s.shell && s.modify_path && !on_path(env.path, bin)) st.push_back(path_step(bin, env));
    if (t.release) st.push_back(update_checks_step(s, t));
    return st;
}

}  // namespace tona::install
