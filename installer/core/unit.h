// The systemd user unit of a release installation and the command line of the server. The unit holds one line that
// matters: `<root>/current/bin/tona start --foreground`. That command reads the settings from tona.conf and execs
// tona-server (server_launch), so the unit never changes with a setting or a version, and macOS runs the same start
// from its LaunchAgent.
//
// scripts/tona-service.sh writes the unit of a checkout (the program in work/build/); it keeps the same restart rule:
// no swap, restart on failure at most three times in ten minutes.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace tona::install {

struct UnitSpec {
    std::string root;          // install root; the program is <root>/current/bin/tona-server
    std::string model_file;    // first file of the model
    std::string drafter_flag;  // "--mtp" or "--dflash", empty: no drafter
    std::string drafter_file;
    std::string profile;       // expert profile (TONA_EXPERT_PROFILE)
    std::string prompt_cache;  // folder of the prompt checkpoints (TONA_PROMPT_CACHE)
    std::string api_key_file;  // empty: no key
    std::string host;          // address to listen on; empty: the server default (this PC only)
    int port = 8662;
    int ctx = 262144;
    int seqs = 2;
    double ram_gb = 0;  // TONA_RAM_GB and, as bytes, MemoryMax
    double vram_gb = 0;
    double prompt_cache_gb = 16;
    bool oea = true;
    bool clock_hold = false;  // TONA_CLOCK_HOLD=1; not set when off
    std::string role_effort;  // empty: off
};

constexpr const char* kUnitName = "tona-server";

// One word of ExecStart or one Environment assignment, quoted where systemd would split or expand it
std::string unit_quote(const std::string& s);

// Text of ~/.config/systemd/user/tona-server.service for the installation in `root`
std::string render_unit(const std::string& root);

// How the server is started: the folder, the environment and the argument list (argv[0] is the program)
struct ServerLaunch {
    std::string workdir;
    std::vector<std::pair<std::string, std::string>> env;
    std::vector<std::string> argv;
};

// `macos` picks the platform part of the environment: the GPU selection on Linux, the thread count on a Mac
ServerLaunch server_launch(const UnitSpec& u, bool macos);

// MemoryMax in bytes for a RAM limit in GB (decimal, as TONA_RAM_GB)
long long memory_max_bytes(double ram_gb);

}  // namespace tona::install
