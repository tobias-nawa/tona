#include "gpu/pcie_link.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <system_error>

#include "backend/env.h"

namespace tona::gpu {

namespace {

// PCIe 5.0 x16: the link of the reference platform, where the 15 us of the planner were tuned (the result is flat
// between 8 and 25 us, research/measurements/2026-10-24-misses.md) and where the GPU-direct reads reach the link rate
// (55.7 GB/s H2D, research/measurements/2026-10-22-micro.md; 43 to 61 GB/s in the expert step,
// 2026-11-16-expert-bandwidth.md)
constexpr double kRefGts = 32.0;
constexpr int kRefWidth = 16;

std::string read_line(const std::filesystem::path& p) {
    std::ifstream in(p);
    std::string s;
    std::getline(in, s);
    return s;
}

int read_int(const std::filesystem::path& p) {
    const std::string s = read_line(p);
    int v = 0;
    return std::from_chars(s.data(), s.data() + s.size(), v).ec == std::errc() ? v : 0;
}

PcieLink read_one(const std::filesystem::path& dir) {
    return {parse_link_speed(read_line(dir / "max_link_speed")), read_int(dir / "current_link_width")};
}

}  // namespace

double parse_link_speed(std::string_view text) {
    double v = 0;
    if (!env::read_double(text.data(), text.data() + text.size(), v)) return 0;  // from_chars, also on macOS 15
    return text.contains("GT/s") ? v : 0;
}

PcieLink read_pcie_link(const std::string& device_dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dev = fs::canonical(device_dir, ec);
    if (ec) return {};
    PcieLink l = read_one(dev);
    if (!l.known()) return {};
    // The upstream port: a card in a slot that the board wires with fewer lanes or a lower generation
    if (const PcieLink up = read_one(dev.parent_path()); up.known()) {
        l.gts = std::min(l.gts, up.gts);
        l.width = std::min(l.width, up.width);
    }
    return l;
}

double gd_cost_us(double reference_us, const PcieLink& link) {
    if (!link.known()) return reference_us;
    return reference_us * (kRefGts * kRefWidth) / (link.gts * link.width);
}

}  // namespace tona::gpu
