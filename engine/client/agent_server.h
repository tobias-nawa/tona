// What the agent launcher asks the running server: is it up, what model and context does it have, is it idle. Plain
// HTTP on 127.0.0.1 with short timeouts (a server that does not answer within a second counts as not running).
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

#include "client/env.h"

namespace tona::client {

// GET /status answered with a 2xx
bool server_answers(int port);

// The engine state of /status: "idle", "prefill", "decode"; empty when no server answers
std::string server_engine_state(int port);

// n_ctx of GET /health, 0 when it is not there
int64_t server_context_size(int port);

// Id and display name of the loaded model from GET /v1/models (the key is sent in case the server asks for one);
// both stay as they are when the server does not answer
void server_model_names(int port, const std::string& key, std::string& id, std::string& name);

// Waits until the model is loaded and shows the current phase on `out`. False after an error while loading, when the
// service failed, or after `wait_s` seconds without an answer; the reason is on `err`.
bool wait_until_ready(int port, int wait_s, std::ostream& out, std::ostream& err);

}  // namespace tona::client
