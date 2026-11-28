// gpu/pcie_link.h: link speed parsing, the link of a device from a fake sysfs tree, and the GPU-direct price
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include "gpu/pcie_link.h"
#include "tests/test_support.h"

using namespace tona::gpu;

namespace {

void write_link(const std::filesystem::path& dir, const std::string& max_speed, int width) {
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "max_link_speed") << max_speed << "\n";
    std::ofstream(dir / "current_link_width") << width << "\n";
}

bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

}  // namespace

int main() {
    EXPECT(near(parse_link_speed("32.0 GT/s PCIe"), 32.0));
    EXPECT(near(parse_link_speed("16.0 GT/s PCIe"), 16.0));
    EXPECT(near(parse_link_speed("2.5 GT/s PCIe"), 2.5));
    EXPECT(parse_link_speed("Unknown") == 0);
    EXPECT(parse_link_speed("") == 0);
    EXPECT(parse_link_speed("32.0") == 0);  // no unit

    // The reference link gives exactly the tuned 15 us (the result of today's build must not move)
    EXPECT(gd_cost_us(15.0, {32.0, 16}) == 15.0);
    EXPECT(near(gd_cost_us(15.0, {32.0, 8}), 30.0));   // x8: half the rate (28.8 vs 55.6 GB/s measured: 1.93)
    EXPECT(near(gd_cost_us(15.0, {16.0, 16}), 30.0));  // PCIe 4.0 x16
    EXPECT(near(gd_cost_us(15.0, {16.0, 8}), 60.0));
    EXPECT(near(gd_cost_us(15.0, {32.0, 4}), 60.0));
    EXPECT(gd_cost_us(15.0, {}) == 15.0);  // unknown link: the reference price
    EXPECT(gd_cost_us(15.0, {32.0, 0}) == 15.0);

    // sysfs: bridge/endpoint tree, the endpoint reached through a symlink like /sys/bus/pci/devices
    const tona::test::TempDir dir("pcie_link");
#ifdef _WIN32
    // NTFS forbids ':' in path components; read_pcie_link inspects the files, not the directory names
    const auto port = dir.path() / "tree" / "0000_00_01.1";
    const auto gpu = port / "0000_03_00.0";
    const auto dev_name = "0000_03_00.0";
#else
    const auto port = dir.path() / "tree" / "0000:00:01.1";
    const auto gpu = port / "0000:03:00.0";
    const auto dev_name = "0000:03:00.0";
#endif
    write_link(port, "32.0 GT/s PCIe", 16);
    write_link(gpu, "32.0 GT/s PCIe", 16);
    std::filesystem::create_directories(dir.path() / "devices");
    std::filesystem::path dev_path = dir.path() / "devices" / dev_name;
#ifdef _WIN32
    std::error_code sym_ec;
    std::filesystem::create_directory_symlink(gpu, dev_path, sym_ec);
    if (sym_ec) dev_path = gpu;  // unprivileged Windows cannot create symlinks without Developer Mode
#else
    std::filesystem::create_directory_symlink(gpu, dev_path);
#endif
    PcieLink l = read_pcie_link(dev_path.string());
    EXPECT(l.known() && near(l.gts, 32.0) && l.width == 16);

    write_link(gpu, "32.0 GT/s PCIe", 8);  // trained at x8
    l = read_pcie_link(dev_path.string());
    EXPECT(l.width == 8 && near(gd_cost_us(15.0, l), 30.0));

    write_link(gpu, "32.0 GT/s PCIe", 16);
    write_link(port, "16.0 GT/s PCIe", 16);  // the port limits the generation
    l = read_pcie_link(dev_path.string());
    EXPECT(near(l.gts, 16.0) && l.width == 16);

    EXPECT(!read_pcie_link((dir.path() / "nothing").string()).known());
    return tona::test::finish("pcie_link");
}
