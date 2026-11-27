// The coding agents `tona claude|opencode|junie|pi` start against the server. One module per agent (agent_claude.cpp
// and the others) turns a context (port, context size, key, model, project) into the command to run: program,
// arguments and the environment of this one invocation. The modules never start a process, so each is tested without
// any. The shared part (server, folder, wait, start, exit question) is agent_launch.h.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tona::client {

// What the environment variables and the settings file say (docs/use/environment.md, "the tona command")
struct AgentSettings {
    std::string tools;                 // TONA_TOOLS: "" all tools, "lean", or a list of Claude Code tools
    std::string compact = "128k";      // TONA_COMPACT
    std::string mode = "acceptEdits";  // TONA_MODE
    std::string effort = "medium";     // TONA_EFFORT, else TONA_EFFORT of tona.conf
    std::string pi_dir;                // PI_CODING_AGENT_DIR, else ~/.pi/agent
};

struct AgentContext {
    int port = 8662;
    int64_t n_ctx = 0;  // context size of the server (/health); 0 when it did not say
    std::string key = "local";
    // The Coder's values are the fallback when the server does not name its model (/v1/models)
    std::string model_id = "qwen3.8-flash-next-coder";
    std::string model_name = "Qwen3.8 Flash Next Coder (tona)";
    std::string project;    // folder name of the working directory, sent as X-Tona-Project
    std::string hint_file;  // the note every agent gets in its instructions (write_hint)
    std::string hint_text;  // the same text
    std::string state_dir;  // junie-models/ lives here
    AgentSettings settings;
};

struct AgentCommand {
    std::string program;
    std::vector<std::string> args;
    std::vector<std::pair<std::string, std::string>> env;  // set for this agent only
};

// Each fills `out` and returns an error text, empty on success. `user_args` are the agent's own arguments as the user
// typed them (after the project folder); they come last, so they win.
std::string claude_command(const AgentContext& c, const std::vector<std::string>& user_args, AgentCommand& out);
std::string opencode_command(const AgentContext& c, const std::vector<std::string>& user_args, AgentCommand& out);
std::string junie_command(const AgentContext& c, const std::vector<std::string>& user_args, AgentCommand& out);
std::string pi_command(const AgentContext& c, const std::vector<std::string>& user_args, AgentCommand& out);

// "claude", "opencode", "junie" or "pi"
bool is_agent(const std::string& word);
// The command of the agent called `agent` (error text for an unknown name)
std::string agent_command(const std::string& agent, const AgentContext& c, const std::vector<std::string>& user_args,
                          AgentCommand& out);

// The note in the instructions: never stop the server or take its port. An agent once took the server for a leftover
// process of its own app and killed it (2026-11-03).
std::string hint_text(int port);

// True if an argument of the user starts with `prefix` ("--effort" also covers "--effort=high")
bool has_arg_with_prefix(const std::vector<std::string>& args, const std::string& prefix);

// Replaces `path` with `content` atomically (written next to it first); creates the directory. Error text or "".
std::string write_file_atomic(const std::string& path, const std::string& content);

}  // namespace tona::client
