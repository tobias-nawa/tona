#include "update/state.h"

#include <fcntl.h>
#include <unistd.h>

#include <fstream>
#include <sstream>

#include "third_party/nlohmann/json.hpp"
#include "update/crypto.h"
#include "update/fsutil.h"

namespace tona::update {

namespace {

constexpr size_t kMaxStateBytes = 4096;

StateResult bad(const std::string& why) {
    StateResult r;
    r.error = "state.json: " + why + " (fix or delete it)";
    return r;
}

}  // namespace

StateResult load_state(const Layout& layout) {
    StateResult r;
    std::ifstream in(layout.state(), std::ios::binary);
    if (!in) {
        r.ok = true;  // first run
        return r;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    if (text.size() > kMaxStateBytes) return bad("too large");
    const auto j = nlohmann::json::parse(text, nullptr, false);
    if (!j.is_object() || !j.contains("seq") || !j.contains("checks") || j.size() != 2 ||
        !j["seq"].is_number_unsigned() || !j["checks"].is_boolean())
        return bad("unexpected content");
    r.state.seq = j["seq"].get<uint64_t>();
    r.state.checks_enabled = j["checks"].get<bool>();
    r.ok = true;
    return r;
}

std::string save_state(const Layout& layout, const State& state) {
    try {
        const nlohmann::json j = {{"seq", state.seq}, {"checks", state.checks_enabled}};
        const Dir root = Dir::open(layout.root.string());
        uint8_t rnd[6];
        random_bytes(rnd, sizeof rnd);
        const std::string tmp = "state.json.tmp-" + to_hex(rnd, sizeof rnd);
        {
            FileWriter w(root, tmp);
            const std::string body = j.dump() + "\n";
            if (!w.write(body.data(), body.size())) throw FsError("cannot write state.json");
            w.finish(0600);
        }
        if (::renameat(root.fd(), tmp.c_str(), root.fd(), "state.json") != 0) {
            ::unlinkat(root.fd(), tmp.c_str(), 0);
            throw FsError("cannot replace state.json");
        }
        fsync_dir(root);
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

}  // namespace tona::update
