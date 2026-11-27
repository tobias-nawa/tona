// `tona claude|opencode|junie|pi [folder] [agent options]`: makes sure a server runs, runs the agent in the folder
// with the settings of this one invocation (agent.h) and, if it started the server itself, asks on exit whether to
// stop it again (about 25 GB RAM pinned, almost all of the VRAM).
#pragma once

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#include "client/agent.h"
#include "client/control.h"
#include "client/env.h"

namespace tona::client {

struct LaunchIo {
    std::ostream* out = nullptr;
    std::ostream* err = nullptr;
    // The answer to the exit question; the function shows `prompt` itself when there is a terminal. Empty answer
    // (also at the end of input) means yes: the server was started for this session.
    std::function<std::string(const std::string& prompt)> ask;
};

struct LaunchOptions {
    int wait_s = 120;  // how long a server that does not answer at all may take to come up
};

// The settings from the environment variables and the settings file of the installation
AgentSettings agent_settings(const Env& env);

// args: the agent, then the options of the command line. Returns the exit code of the agent (1 when it could not be
// started).
int run_agent(const Env& env, Control& control, const std::vector<std::string>& args, LaunchIo& io,
              const LaunchOptions& options = {});

// Runs a command and waits for it; SIGINT and SIGQUIT are ignored meanwhile (the agent gets them, tona stays to ask
// its question). Returns its exit code, 128 + signal after a signal, 127 when the program is not found.
int spawn_agent(const AgentCommand& cmd, const std::string& cwd, std::ostream& err);

}  // namespace tona::client
