#include "client/agent.h"

#include <unistd.h>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace tona::client {

bool is_agent(const std::string& word) {
    return word == "claude" || word == "opencode" || word == "junie" || word == "pi";
}

std::string agent_command(const std::string& agent, const AgentContext& c, const std::vector<std::string>& user_args,
                          AgentCommand& out) {
    if (agent == "claude") return claude_command(c, user_args, out);
    if (agent == "opencode") return opencode_command(c, user_args, out);
    if (agent == "junie") return junie_command(c, user_args, out);
    if (agent == "pi") return pi_command(c, user_args, out);
    return "unknown agent: " + agent;
}

std::string hint_text(int port) {
    const std::string p = std::to_string(port);
    return "The language model you are talking to is served by tona-server on 127.0.0.1:" + p +
           " on this machine.\nNever stop, kill or restart that process (shown as tona-server), and do not bind your "
           "own apps or tests to\nport " +
           p + "; pick another port for them.";
}

bool has_arg_with_prefix(const std::vector<std::string>& args, const std::string& prefix) {
    for (const std::string& a : args)
        if (a.starts_with(prefix)) return true;
    return false;
}

std::string write_file_atomic(const std::string& path, const std::string& content) {
    std::error_code ec;
    const fs::path target(path);
    fs::create_directories(target.parent_path(), ec);
    if (ec) return target.parent_path().string() + ": " + ec.message();
    // The pid in the name: two tona invocations never write the same temporary file
    const fs::path tmp = target.string() + "." + std::to_string(::getpid()) + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << content;
        if (!f) return tmp.string() + ": cannot write";
    }
    fs::rename(tmp, target, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return path + ": cannot replace";
    }
    return {};
}

}  // namespace tona::client
