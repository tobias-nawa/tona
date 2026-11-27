// What each agent module makes of a context: program, arguments, environment and the files it writes. No process, no
// server (the launcher with a fake agent is test_agent_launch.cpp).
#include <unistd.h>

#include <algorithm>

#include "client/agent.h"
#include "client/env.h"
#include "tests/client_testkit.h"

using namespace tona::client;
using namespace tona::test;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

AgentContext context(const std::string& dir) {
    AgentContext c;
    c.port = 8676, c.n_ctx = 262144, c.key = "secretkey", c.project = "shop";
    c.hint_text = hint_text(c.port), c.hint_file = dir + "/agent-hint.md", c.state_dir = dir;
    c.settings.pi_dir = dir + "/pi";
    return c;
}

std::string env_of(const AgentCommand& cmd, const std::string& name) {
    for (const auto& [k, v] : cmd.env)
        if (k == name) return v;
    return "<unset>";
}

bool has_arg(const AgentCommand& cmd, const std::string& a) {
    return std::find(cmd.args.begin(), cmd.args.end(), a) != cmd.args.end();
}

void test_claude_settings_and_order() {
    TempDir dir("agent-claude");
    AgentContext c = context(dir.path().string());
    AgentCommand cmd;
    EXPECT(claude_command(c, {"-p", "hi there"}, cmd).empty());
    EXPECT(cmd.program == "claude");
    EXPECT(env_of(cmd, "ANTHROPIC_BASE_URL") == "http://127.0.0.1:8676");
    EXPECT(env_of(cmd, "ANTHROPIC_AUTH_TOKEN") == "secretkey");
    EXPECT(env_of(cmd, "ANTHROPIC_MODEL") == "qwen3.8-flash-next-coder");
    EXPECT(env_of(cmd, "ANTHROPIC_CUSTOM_HEADERS") == "X-Tona-Project: shop");
    EXPECT(env_of(cmd, "CLAUDE_CODE_MAX_CONTEXT_TOKENS") == "262144");
    EXPECT(env_of(cmd, "CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "128k");
    const std::vector<std::string> head = {
        "--permission-mode", "acceptEdits", "--append-system-prompt", c.hint_text, "--effort", "medium", "-p",
        "hi there"};
    EXPECT(cmd.args == head);  // the user's own arguments come last and arrive unsplit

    c.n_ctx = 65536;  // below 128K compacting at 128k would never happen
    AgentCommand small;
    EXPECT(claude_command(c, {}, small).empty() && env_of(small, "CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "<unset>");
}

void test_claude_effort_tools_and_mode() {
    TempDir dir("agent-claude2");
    AgentContext c = context(dir.path().string());
    c.settings.effort = "low", c.settings.mode = "auto", c.settings.compact = "200k";
    AgentCommand cmd;
    EXPECT(claude_command(c, {}, cmd).empty());
    EXPECT(has_arg(cmd, "auto") && has_arg(cmd, "low") && env_of(cmd, "CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "200k");
    EXPECT(!has_arg(cmd, "--tools"));  // all tools and MCP servers unless asked

    AgentCommand own;
    EXPECT(claude_command(c, {"--effort", "high"}, own).empty());
    EXPECT(!has_arg(own, "low") && has_arg(own, "high"));  // the call names a level: tona adds none
    AgentCommand own_eq;
    EXPECT(claude_command(c, {"--effort=high"}, own_eq).empty() && !has_arg(own_eq, "--effort"));

    c.settings.tools = "lean";
    AgentCommand lean;
    EXPECT(claude_command(c, {}, lean).empty());
    EXPECT(has_arg(lean, "--strict-mcp-config") && has_arg(lean, "Bash,Read,Edit,Write,Glob,Grep"));
    c.settings.tools = "Bash,Read";
    AgentCommand named;
    EXPECT(claude_command(c, {}, named).empty() && has_arg(named, "Bash,Read"));
}

void test_opencode_inline_config() {
    TempDir dir("agent-opencode");
    AgentContext c = context(dir.path().string());
    c.model_id = "other-model", c.model_name = "Other (local)";
    AgentCommand cmd;
    EXPECT(opencode_command(c, {"run", "do it"}, cmd).empty());
    EXPECT(cmd.program == "opencode" && cmd.args == std::vector<std::string>({"run", "do it"}));
    EXPECT(env_of(cmd, "TONA_API_KEY") == "secretkey" && env_of(cmd, "TONA_PROJECT") == "shop");
    EXPECT(env_of(cmd, "OPENCODE_DISABLE_EXTERNAL_SKILLS") == "<unset>");
    const json cfg = json::parse(env_of(cmd, "OPENCODE_CONFIG_CONTENT"));
    const json& p = cfg["provider"]["tona"];
    EXPECT(p["options"]["baseURL"] == "http://127.0.0.1:8676/v1");
    EXPECT(p["options"]["apiKey"] == "{env:TONA_API_KEY}");  // the key is in the environment, not in the config text
    EXPECT(p["options"]["headers"]["X-Tona-Project"] == "{env:TONA_PROJECT}");
    EXPECT(p["options"]["timeout"] == false && p["options"]["chunkTimeout"] == false);
    EXPECT(p["models"]["other-model"]["limit"]["context"] == 262144 &&
           p["models"]["other-model"]["limit"]["output"] == 8192);
    EXPECT(p["models"]["other-model"]["name"] == "Other (local)");
    EXPECT(cfg["model"] == "tona/other-model" && cfg["small_model"] == "tona/other-model");
    EXPECT(cfg["agent"]["title"]["disable"] == true);
    EXPECT(cfg["instructions"][0] == c.hint_file);
    EXPECT(!has(env_of(cmd, "OPENCODE_CONFIG_CONTENT"), "secretkey"));

    c.settings.tools = "lean";
    AgentCommand lean;
    EXPECT(opencode_command(c, {}, lean).empty());
    EXPECT(env_of(lean, "OPENCODE_DISABLE_EXTERNAL_SKILLS") == "1" &&
           env_of(lean, "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS") == "1");
}

void test_junie_writes_its_model_profile() {
    TempDir dir("agent-junie");
    AgentContext c = context(dir.path().string());
    AgentCommand cmd;
    EXPECT(junie_command(c, {"--task", "x y"}, cmd).empty());
    const std::string models = dir.path().string() + "/junie-models";
    const std::vector<std::string> want = {"--model-location", models,      "--model", "custom:tona",
                                           "--ide-guidelines", c.hint_text, "--task",  "x y"};
    EXPECT(cmd.program == "junie" && cmd.args == want);
    EXPECT(env_of(cmd, "TONA_API_KEY") == "secretkey" && env_of(cmd, "TONA_PROJECT") == "shop");
    const json p = json::parse(read_text(models + "/tona.json"));
    EXPECT(p["baseUrl"] == "http://127.0.0.1:8676/v1/chat/completions" && p["apiType"] == "OpenAICompletion");
    EXPECT(p["apiKey"] == "${TONA_API_KEY}" && p["extraHeaders"]["X-Tona-Project"] == "${TONA_PROJECT}");
    EXPECT(p["extraBody"]["max_tokens"] == 8192 && p["maxContextLength"] == 262144);
}

void test_pi_keeps_the_other_providers() {
    TempDir dir("agent-pi");
    AgentContext c = context(dir.path().string());
    fs::create_directories(c.settings.pi_dir);
    const std::string file = c.settings.pi_dir + "/models.json";
    std::ofstream(file) << R"({"providers": {"mine": {"baseUrl": "http://x"}}, "other": 1})";
    AgentCommand cmd;
    EXPECT(pi_command(c, {"-p", "go"}, cmd).empty());
    const json doc = json::parse(read_text(file));
    EXPECT(doc["providers"]["mine"]["baseUrl"] == "http://x" && doc["other"] == 1);
    const json& t = doc["providers"]["tona"];
    EXPECT(t["api"] == "anthropic-messages" && t["baseUrl"] == "http://127.0.0.1:8676");
    EXPECT(t["apiKey"] == "$TONA_API_KEY" && t["headers"]["X-Tona-Project"] == "$TONA_PROJECT");
    EXPECT(t["models"][0]["contextWindow"] == 262144 && t["models"][0]["maxTokens"] == 16384);
    const std::vector<std::string> want = {
        "--model", "tona/qwen3.8-flash-next-coder", "--append-system-prompt", c.hint_text, "--thinking", "medium", "-p",
        "go"};
    EXPECT(cmd.program == "pi" && cmd.args == want);
    AgentCommand own;
    EXPECT(pi_command(c, {"--thinking", "high"}, own).empty() && !has_arg(own, "medium"));

    std::ofstream(file) << "not json";  // a file that cannot be read is left alone
    AgentCommand broken;
    EXPECT(has(pi_command(c, {}, broken), "could not update") && read_text(file) == "not json");
}

void test_pi_creates_its_directory_and_file() {
    TempDir dir("agent-pi2");
    AgentContext c = context(dir.path().string());
    AgentCommand cmd;
    EXPECT(pi_command(c, {}, cmd).empty());
    EXPECT(json::parse(read_text(c.settings.pi_dir + "/models.json"))["providers"].contains("tona"));
}

void test_files_that_cannot_be_written_are_errors() {
    TempDir dir("agent-write");
    const std::string d = dir.path().string();
    fs::create_directories(d + "/target/inside");
    EXPECT(has(write_file_atomic(d + "/target", "x"), "cannot replace"));  // a directory is not replaced by a file
    EXPECT(fs::exists(d + "/target/inside"));
    EXPECT(!fs::exists(d + "/target." + std::to_string(getpid()) + ".tmp"));  // the temporary file is gone again
    std::ofstream(d + "/file") << "a file where a directory is needed";
    EXPECT(!write_file_atomic(d + "/file/sub/x", "x").empty());
    AgentContext c = context(d + "/file");  // junie keeps its profile below the state directory
    AgentCommand cmd;
    EXPECT(!junie_command(c, {}, cmd).empty());
}

void test_launcher_program_by_layout() {
    EXPECT(Env::for_checkout("/x").launcher() == "/x/work/build/engine/tona");
    EXPECT(Env::for_release("/r").launcher() == "/r/current/bin/tona");
}

void test_agent_names() {
    for (const char* a : {"claude", "opencode", "junie", "pi"}) EXPECT(is_agent(a));
    EXPECT(!is_agent("top") && !is_agent("start") && !is_agent(""));
    AgentCommand cmd;
    EXPECT(has(agent_command("frobnicate", AgentContext{}, {}, cmd), "unknown agent"));
    EXPECT(has(hint_text(8676), "127.0.0.1:8676") && has(hint_text(8676), "Never stop, kill or restart"));
}

}  // namespace

int main() {
    return run("agent_commands", [] {
        test_claude_settings_and_order();
        test_claude_effort_tools_and_mode();
        test_opencode_inline_config();
        test_junie_writes_its_model_profile();
        test_pi_keeps_the_other_providers();
        test_pi_creates_its_directory_and_file();
        test_files_that_cannot_be_written_are_errors();
        test_launcher_program_by_layout();
        test_agent_names();
    });
}
