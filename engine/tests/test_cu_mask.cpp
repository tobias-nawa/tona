// gpu/cu_mask.h: CU count from the KFD topology and the HSA_CU_MASK string, on the CPU with a fake topology
#include <filesystem>
#include <fstream>
#include <string>

#include "gpu/cu_mask.h"
#include "tests/test_support.h"

using namespace tona::gpu;

namespace {

void write_node(const std::filesystem::path& root, int id, const std::string& text) {
    std::filesystem::create_directories(root / std::to_string(id));
    std::ofstream(root / std::to_string(id) / "properties") << text;
}

}  // namespace

int main() {
    // The reference card: 64 SIMDs, 2 per CU
    const std::string gpu32 = "cpu_cores_count 0\nsimd_count 64\nsimd_per_cu 2\nmax_waves_per_simd 16\n";
    const std::string gpu64 = "simd_count 128\nsimd_per_cu 2\n";
    EXPECT(kfd_property(gpu32, "simd_count") == 64);
    EXPECT(kfd_property(gpu32, "simd_per_cu") == 2);
    EXPECT(kfd_property(gpu32, "simd") == -1);  // a prefix is not a key
    EXPECT(kfd_property(gpu32, "missing") == -1);
    EXPECT(kfd_node_cus(gpu32) == 32);
    EXPECT(kfd_node_cus(gpu64) == 64);
    EXPECT(kfd_node_cus("cpu_cores_count 16\nsimd_count 0\nsimd_per_cu 0\n") == 0);  // CPU node
    EXPECT(kfd_node_cus("") == 0);

    // 32 CUs, 4 free: CUs 0-27 stay; 64 CUs: the same count of free CUs frees the top of the card
    EXPECT(cu_mask_string(32, 4) == "0:0-27");
    EXPECT(cu_mask_string(32, 8) == "0:0-23");
    EXPECT(cu_mask_string(64, 4) == "0:0-59");
    EXPECT(cu_mask_string(64, 16) == "0:0-47");
    EXPECT(cu_mask_string(32, 0).empty());
    EXPECT(cu_mask_string(32, 32).empty());  // nothing would be left
    EXPECT(cu_mask_string(0, 4).empty());    // unknown device

    const tona::test::TempDir dir("cu_mask");
    const auto root = dir.path() / "nodes";
    write_node(root, 0, "simd_count 0\nsimd_per_cu 0\n");
    write_node(root, 2, "simd_count 4\nsimd_per_cu 2\n");  // the iGPU comes after the dGPU
    write_node(root, 1, gpu32);
    EXPECT(first_gpu_cus(root.string()) == 32);
    EXPECT(first_gpu_cus((root / "none").string()) == 0);
    return tona::test::finish("cu_mask");
}
