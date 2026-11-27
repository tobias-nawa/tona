#include "gpu/cu_mask.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <system_error>
#include <vector>

namespace tona::gpu {

long long kfd_property(std::string_view properties, std::string_view key) {
    size_t pos = 0;
    while (pos < properties.size()) {
        size_t end = properties.find('\n', pos);
        if (end == std::string_view::npos) end = properties.size();
        const std::string_view line = properties.substr(pos, end - pos);
        pos = end + 1;
        if (line.size() <= key.size() || !line.starts_with(key) || line[key.size()] != ' ') continue;
        long long v = -1;
        const std::string_view num = line.substr(key.size() + 1);
        return std::from_chars(num.data(), num.data() + num.size(), v).ec == std::errc() ? v : -1;
    }
    return -1;
}

int kfd_node_cus(std::string_view properties) {
    const long long simds = kfd_property(properties, "simd_count"), per_cu = kfd_property(properties, "simd_per_cu");
    return simds > 0 && per_cu > 0 ? (int)(simds / per_cu) : 0;
}

int first_gpu_cus(const std::string& root) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<int> ids;
    for (const auto& e : fs::directory_iterator(root, ec)) {
        int id = -1;
        const std::string name = e.path().filename().string();
        if (std::from_chars(name.data(), name.data() + name.size(), id).ec == std::errc()) ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    for (const int id : ids) {
        std::ifstream in(std::format("{}/{}/properties", root, id));
        std::stringstream text;
        text << in.rdbuf();
        if (const int cus = kfd_node_cus(text.str()); cus > 0) return cus;
    }
    return 0;
}

std::string cu_mask_string(int cus, int free) {
    if (free <= 0 || free >= cus) return {};
    return std::format("0:0-{}", cus - 1 - free);
}

}  // namespace tona::gpu
