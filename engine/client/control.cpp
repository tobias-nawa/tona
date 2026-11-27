#include "client/control.h"

#include <filesystem>
#include <format>

#include "client/settings.h"
#include "core/catalog.h"
#include "core/config.h"
#include "core/model_fetch.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::client {

namespace {

std::string last_line(const std::string& text) {
    std::string t = install::trim(text);
    const size_t nl = t.rfind('\n');
    return nl == std::string::npos ? t : t.substr(nl + 1);
}

class SystemControl : public Control {
public:
    explicit SystemControl(Env env) : env_(std::move(env)) {}

    std::string start() override {
        if (const std::string err = env_.refresh_unit(); !err.empty()) return err;
        return shell(env_.start_command(), "start");
    }
    std::string stop() override { return shell(env_.stop_command(), "stop"); }

    std::string restart() override {
        if (const std::string err = env_.refresh_unit(); !err.empty()) return err;
        if (const std::string err = stop(); !err.empty()) return err;
        return start();
    }

    std::string download(const std::string& model_id, const std::string& drafter, const LineFn& line) override {
        for (const std::string& id : {model_id, drafter}) {
            if (id.empty()) continue;
            const std::string err = env_.release ? fetch_release(id, line) : fetch_checkout(id, line);
            if (!err.empty()) return err;
        }
        return {};
    }

    std::string remove(const std::string& model_id) override {
        if (model_id.empty() || model_id.contains('/') || model_id.contains("..")) return "not a model id: " + model_id;
        std::error_code ec;
        const fs::path dir = fs::path(env_.models_dir) / model_id;
        if (!fs::is_directory(dir, ec)) return {};
        fs::remove_all(dir, ec);
        return ec ? dir.string() + ": " + ec.message() : "";
    }

private:
    std::string shell(const std::string& cmd, const char* what) const {
        const install::RunResult r = install::run(cmd);
        return r.status == 0 ? "" : std::format("{} failed ({}): {}", what, r.status, last_line(r.out));
    }

    std::string fetch_release(const std::string& id, const LineFn& line) const {
        const install::CatalogEntry* e = install::find_entry(id);
        if (!e) return "unknown model: " + id;
        return install::fetch_model(*e, env_.models_dir, install::ModelSource{}, "", line);
    }

    std::string fetch_checkout(const std::string& id, const LineFn& line) const {
        const install::RunResult r = install::run(
            "cd " + install::quote(env_.repo) + " && python3 -I tools/get_model.py " + install::quote(id), line);
        return r.status == 0 ? "" : "download of " + id + " failed: " + last_line(r.out);
    }

    Env env_;
};

const install::ModelInfo* find_model(const std::vector<install::ModelInfo>& catalog, const std::string& id) {
    for (const install::ModelInfo& m : catalog)
        if (m.id == id) return &m;
    return nullptr;
}

}  // namespace

std::unique_ptr<Control> system_control(const Env& env) { return std::make_unique<SystemControl>(env); }

std::vector<install::ModelInfo> installed_catalog(const Env& env) { return install::catalog_of(env.target()); }

std::string model_id_of_file(const std::vector<install::ModelInfo>& catalog, const std::string& file) {
    for (const install::ModelInfo& m : catalog)
        if (m.kind == "model" && !file.empty() && m.first_file == file) return m.id;
    return {};
}

std::string prepare_switch(const Env& env, const std::vector<install::ModelInfo>& catalog,
                           const std::string& model_id) {
    const install::ModelInfo* m = find_model(catalog, model_id);
    if (!m || m->kind != "model") return "unknown model: " + model_id;
    install::Settings s = load_settings(env);
    const std::string drafter = install::drafter_entry_id(s.drafter, model_id);
    if (!m->present) return model_id + " is not installed (verified) yet: download it first";
    if (!drafter.empty()) {
        const install::ModelInfo* d = find_model(catalog, drafter);
        if (!d || !d->present)
            return "the drafter " + drafter + " of " + model_id + " is not installed: download it first";
    }
    s.model = model_id;
    // "auto" is planned per model (its own pinning limit): the context of the old model does not carry over
    if (s.ctx_auto) local_resolver(env)(s);
    return save_settings(env, s);
}

std::string switch_model(const Env& env, Control& control, const std::vector<install::ModelInfo>& catalog,
                         const std::string& model_id) {
    if (const std::string err = prepare_switch(env, catalog, model_id); !err.empty()) return err;
    return control.restart();
}

}  // namespace tona::client
