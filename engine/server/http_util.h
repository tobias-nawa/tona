// Small helpers shared by the HTTP routes and the engine thread
#pragma once

#include <functional>
#include <string>

#include "server/chat.h"
#include "server/job.h"

namespace tona::server {

// Id and display name of the Coder: clients (tona, harness configs) address it by this id, so it never changes
inline constexpr const char* kModelName = "qwen3.8-flash-next-coder";
inline constexpr const char* kCoderGgufName = "Qwen3.8-Flash-Next GSQ-RCO Coder";

struct ModelIdentity {
    std::string id, display_name;
};
// The names the API reports for a model whose GGUF says general.name = gguf_name. The Coder (and a file without a
// name) keeps its fixed id; any other model gets its name as a lowercase id and "<name> (local)" as display name.
ModelIdentity identity_of(const std::string& gguf_name);
// Set once while the model loads, before the routes answer; read by every response that names the model
void set_model_identity(ModelIdentity id);
std::string model_id();
std::string model_display_name();

std::string rand_id(const char* prefix);
// Wall-clock time for the log (HH:MM:SS), to line up with scripts/gpu-watch.sh
std::string hms();
// One server-sent event; an empty event name gives a bare data line
std::string sse(const std::string& event, const json& data);
json anthropic_error(const std::string& type, const std::string& msg);

// Output of a job piece by piece until it is done; on_piece(text) returning false cancels.
// alive: is the client still there? Asked every 0.5 s at most, also when nothing is emitted (without streaming, or
// while the parser holds back a long tool call): otherwise the engine keeps computing for minutes after ESC.
void drain(Job& j, const std::function<bool(const std::string&)>& on_piece, const std::function<bool()>& alive);

// Waits until the prefill is done. ping() every 5 s: clients give up after 300 s without bytes and resend the request
// (Claude Code, Codex, OpenCode; research/07-harnesses.md). If ping returns false the client is gone: mark the job
// cancelled; the prefill still runs to the end and leaves its checkpoint behind.
bool wait_started(Job& j, const std::function<bool()>& ping);

}  // namespace tona::server
