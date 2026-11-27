// Parsers for the text the installer reads from the system (/proc, /sys, /etc/os-release, nvidia-smi). They take the
// text, not a path, so that tests can feed them invented machines; the callers in hw.cpp and disk.cpp read the files.
#pragma once

#include <cstdint>
#include <format>
#include <sstream>
#include <string>
#include <vector>

#include "core/sys.h"

namespace tona::install::parse {

// value of a key in "key<sep>value" lines (/proc/cpuinfo, /etc/os-release, KFD properties); quotes around the value
// are removed, a missing key gives an empty string
inline std::string field(const std::string& text, const std::string& key, char sep) {
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        const auto p = line.find(sep);
        if (p == std::string::npos || trim(line.substr(0, p)) != key) continue;
        std::string v = trim(line.substr(p + 1));
        if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
        return v;
    }
    return {};
}

// number at the start of the text (decimal, 0x or octal as stoll with base 0); 0 if there is none
inline int64_t num(const std::string& s) {
    try {
        return std::stoll(s, nullptr, 0);
    } catch (...) {
        return 0;
    }
}

// whether the space-separated flag list ("flags" of /proc/cpuinfo) contains the flag
inline bool has_flag(const std::string& flags, const std::string& flag) {
    return (" " + flags + " ").contains(" " + flag + " ");
}

// source of the file system that contains path (longest matching mount point) in the text of /proc/self/mountinfo;
// path must be canonical
inline std::string mount_source(const std::string& mountinfo, const std::string& path) {
    std::istringstream in(mountinfo);
    std::string best_mp, best_src;
    for (std::string line; std::getline(in, line);) {
        // ID PARENT MAJ:MIN ROOT MOUNTPOINT ... - FSTYPE SOURCE ...
        std::istringstream f(line);
        std::string id, parent, mm, root, mp, w;
        f >> id >> parent >> mm >> root >> mp;
        while (f >> w && w != "-") {
        }
        std::string fstype, src;
        f >> fstype >> src;
        const bool inside = path == mp || path.starts_with(mp == "/" ? "/" : mp + "/");
        if (inside && mp.size() >= best_mp.size()) best_mp = mp, best_src = src;
    }
    return best_src;
}

// gfx_target_version of a KFD node (120000 → gfx1200)
inline std::string gfx_name(int64_t v) { return std::format("gfx{}{:x}{:x}", v / 10000, (v / 100) % 100, v % 100); }

struct NvidiaRow {
    std::string name;
    int64_t total_mib = 0, used_mib = 0;
};

// output of nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
inline std::vector<NvidiaRow> nvidia_rows(const std::string& csv) {
    std::vector<NvidiaRow> rows;
    std::istringstream in(csv);
    for (std::string line; std::getline(in, line);) {
        std::istringstream cols(line);
        std::string name, total, used;
        std::getline(cols, name, ',');
        std::getline(cols, total, ',');
        std::getline(cols, used, ',');
        rows.push_back({trim(name), num(trim(total)), num(trim(used))});
    }
    return rows;
}

}  // namespace tona::install::parse
