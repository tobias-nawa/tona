// Shared pieces of tona: JSON and ANSI helpers, the terminal, the state fed by the status poller and the updater
// keys. The pages themselves are in engine/client/. Terminal output is drawn with ANSI sequences only (no library).
#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "third_party/nlohmann/json.hpp"

namespace tona::update {
struct UpdateView;
class Updater;
}  // namespace tona::update

namespace tona::top {

using nlohmann::json;

double jnum(const json& j, const char* key, double fallback = 0);
std::string jstr(const json& j, const char* key, const std::string& fallback = "");

// ANSI drawing
inline constexpr const char* kReset = "\x1b[0m";
inline constexpr const char* kBold = "\x1b[1m";
inline constexpr const char* kDim = "\x1b[2m";
inline constexpr const char* kInverse = "\x1b[7m";
inline constexpr const char* kRed = "\x1b[31m";
inline constexpr const char* kGreen = "\x1b[32m";
inline constexpr const char* kYellow = "\x1b[33m";
inline constexpr const char* kBlue = "\x1b[34m";
inline constexpr const char* kMagenta = "\x1b[35m";
inline constexpr const char* kCyan = "\x1b[36m";

std::string paint(const char* color, const std::string& text);
// Visible width: escape sequences count zero, every UTF-8 character one column (all used here are narrow)
int width(const std::string& s);
// Cut to width w or pad with spaces
std::string fit(const std::string& s, int w);
std::string right(const std::string& s, int w);
std::string repeat(const char* glyph, int n);

// Terminal
void term_setup();
void term_restore();
// Output of a shell command (stderr included); returns its last line, the whole output in `all`.
std::string run(const std::string& cmd, std::string* all = nullptr);

// State shared between the poller thread and the screen
struct Shared {
    std::mutex mu;
    json status;        // last /status
    bool up = false;    // server reachable
    std::string error;  // why not
    // Decode tok/s per poll (moving over 1 s), split by project (index into `projects`); one part per running
    // conversation
    std::deque<std::vector<std::pair<int, float>>> rate;
    std::vector<std::string> projects;  // in the order of first appearance
    int selected = 0;                   // view: 0 all, else projects[selected - 1]
    std::string message;                // footer message
    std::vector<std::string> log;
    bool busy = false;  // start or stop is running

    int project(const std::string& name);
};

// Polls GET /status four times a second until `quit`, keeps `sh` current. `start` is called once if --start was
// given and no server answers.
struct PollerConfig {
    std::string url;
    bool autostart;
    bool local;
    std::atomic<bool>* quit;
    std::atomic<bool>* show_log;
    std::function<void()> start;
    // Seconds on the clock the decode rate is taken from; empty: the steady clock. A test passes its own so that the
    // rate does not depend on how long a sleep of 250 ms takes on a loaded machine
    std::function<double()> now{};
};
void poll_status(Shared& sh, const PollerConfig& cfg);

// Keys of the updater: u install (or check), c turn checks on/off, r roll back. Turning on and rolling back ask for
// the same key again within a few seconds. Returns the footer message ("" for none). Never blocks.
struct KeyConfirm {
    char key = 0;
    std::chrono::steady_clock::time_point at{};
};
std::string update_key(tona::update::Updater* updater, char key, KeyConfirm& confirm);

}  // namespace tona::top
