// What the client does to the machine: start, stop and restart the server, fetch and remove models, switch the model.
// Behind an interface so the screens and the commands are tested with a fake; `system_control` is the real one.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "client/env.h"
#include "core/plan.h"

namespace tona::client {

using LineFn = std::function<void(const std::string&)>;

class Control {
public:
    virtual ~Control() = default;
    // Each returns an error text, empty on success. They block: callers run them in a thread.
    virtual std::string start() = 0;
    virtual std::string stop() = 0;
    virtual std::string restart() = 0;  // stop, then start (a release installation also reloads the unit)
    // Fetches a model and then its drafter (empty: none); `line` gets progress lines
    virtual std::string download(const std::string& model_id, const std::string& drafter_id, const LineFn& line) = 0;
    virtual std::string remove(const std::string& model_id) = 0;
};

std::unique_ptr<Control> system_control(const Env& env);

// Models of the installation: every catalog entry of kind "model" with `present`
std::vector<install::ModelInfo> installed_catalog(const Env& env);

// Catalog id of the model a server runs, from the file name /status reports ("" if the catalog does not know it)
std::string model_id_of_file(const std::vector<install::ModelInfo>& catalog, const std::string& file);

// Writes the model into the settings and restarts the server. Checks first that the model and its drafter are on the
// disk and verified (a restart onto a missing file would leave the user without a server). Returns an error text.
// Conversations that run at that moment are cut: the caller asks the user before (see switching in the views).
std::string switch_model(const Env& env, Control& control, const std::vector<install::ModelInfo>& catalog,
                         const std::string& model_id);

// The same without the restart: for the command `tona use --no-restart` and the tests
std::string prepare_switch(const Env& env, const std::vector<install::ModelInfo>& catalog, const std::string& model_id);

}  // namespace tona::client
