#include "client/history.h"

#include <filesystem>
#include <fstream>

#include "third_party/nlohmann/json.hpp"

namespace fs = std::filesystem;

namespace tona::client {

using nlohmann::json;

namespace {

std::string history_path(const std::string& state_dir) { return state_dir + "/launches.json"; }

}  // namespace

std::vector<LaunchEntry> load_launches(const std::string& state_dir) {
    std::vector<LaunchEntry> out;
    std::ifstream f(history_path(state_dir));
    const json j = json::parse(f, nullptr, false);
    if (!j.is_array()) return out;
    for (const json& e : j) {
        if (!e.is_object()) continue;
        LaunchEntry l;
        l.agent = e.value("agent", ""), l.project = e.value("project", ""), l.model = e.value("model", "");
        l.time = e.value("time", 0LL);
        if (!l.agent.empty()) out.push_back(l);
        if (out.size() >= kMaxLaunches) break;
    }
    return out;
}

std::string record_launch(const std::string& state_dir, const LaunchEntry& entry) {
    std::vector<LaunchEntry> all = load_launches(state_dir);
    all.insert(all.begin(), entry);
    if (all.size() > kMaxLaunches) all.resize(kMaxLaunches);
    json j = json::array();
    for (const LaunchEntry& l : all)
        j.push_back({{"agent", l.agent}, {"project", l.project}, {"model", l.model}, {"time", l.time}});
    std::error_code ec;
    fs::create_directories(state_dir, ec);
    const std::string tmp = history_path(state_dir) + ".new";
    {
        std::ofstream f(tmp);
        if (!(f << j.dump(1))) return tmp + ": cannot write";
    }
    fs::rename(tmp, history_path(state_dir), ec);
    return ec ? history_path(state_dir) + ": " + ec.message() : "";
}

}  // namespace tona::client
