// The expert layout (experts per layer, backend/expert_layout.h) and the profile file in both versions. The version 1
// bytes are written here by hand, independent of the writer, so that a profile made by an older build must still load.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "backend/expert_layout.h"
#include "backend/expert_profile.h"
#include "tests/test_support.h"

namespace {

using namespace tona;
using backend::ExpertLayout;

void write_words(const std::string& path, const std::vector<uint32_t>& words) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    EXPECT(f != nullptr);
    if (!f) return;
    std::fwrite(words.data(), sizeof(uint32_t), words.size(), f);
    std::fclose(f);
}

std::vector<uint32_t> words_of(const std::string& path) {
    std::vector<uint32_t> w(std::filesystem::file_size(path) / sizeof(uint32_t));
    std::FILE* f = std::fopen(path.c_str(), "rb");
    EXPECT(f && std::fread(w.data(), sizeof(uint32_t), w.size(), f) == w.size());
    if (f) std::fclose(f);
    return w;
}

void test_layout() {
    const ExpertLayout u = ExpertLayout::uniform(3, 256);
    EXPECT(u.layers() == 3 && u.experts(2) == 256 && u.max_experts() == 256 && u.total() == 768);
    EXPECT(u.is_uniform() && u.offset(0) == 0 && u.offset(2) == 512);
    const ExpertLayout v({240, 496, 300});
    EXPECT(!v.is_uniform() && v.max_experts() == 496 && v.total() == 1036);
    EXPECT(v.offset(0) == 0 && v.offset(1) == 240 && v.offset(2) == 736);
    EXPECT(!(u == v) && u == ExpertLayout::uniform(3, 256));
}

// A profile of the Coder's kind: version 1 on disk, as before
void test_version1(const std::string& dir) {
    const std::string path = dir + "/v1.bin";
    const std::vector<uint32_t> counts = {5, 9, 9, 1, 0, 3, 3, 3};
    write_words(path, {0x50584549u, 2, 4, 5, 9, 9, 1, 0, 3, 3, 3});  // "IEXP", layers, experts, counts
    EXPECT(backend::read_expert_profile(path.c_str(), ExpertLayout::uniform(2, 4), "t") == counts);
    // another layout: not this model's profile, also one with the same number of entries
    EXPECT(test::throws([&] { backend::read_expert_profile(path.c_str(), ExpertLayout({3, 5}), "t"); }));
    // the writer keeps the format for a uniform layout, byte for byte
    const std::string out = dir + "/v1-out.bin";
    EXPECT(backend::write_expert_profile(out.c_str(), ExpertLayout::uniform(2, 4), counts));
    EXPECT(words_of(out) == words_of(path));
}

void test_version2(const std::string& dir) {
    const std::string path = dir + "/v2.bin";
    const ExpertLayout lay({3, 5, 2});
    const std::vector<uint32_t> counts = {1, 2, 3, 9, 8, 7, 6, 5, 4, 3};
    EXPECT(backend::write_expert_profile(path.c_str(), lay, counts));
    // header, offsets (layers + 1), counts
    EXPECT(words_of(path) == (std::vector<uint32_t>{0x32584549u, 3, 10, 0, 3, 8, 10, 1, 2, 3, 9, 8, 7, 6, 5, 4, 3}));
    EXPECT(backend::read_expert_profile(path.c_str(), lay, "t") == counts);
    EXPECT(test::throws([&] { backend::read_expert_profile(path.c_str(), ExpertLayout({3, 4, 3}), "t"); }));
    EXPECT(test::throws([&] { backend::read_expert_profile(path.c_str(), ExpertLayout::uniform(3, 4), "t"); }));
    // offsets that do not match the layout, same total
    write_words(path, {0x32584549u, 3, 10, 0, 4, 8, 10, 1, 2, 3, 9, 8, 7, 6, 5, 4, 3});
    EXPECT(test::throws([&] { backend::read_expert_profile(path.c_str(), lay, "t"); }));
    std::filesystem::resize_file(path, 30);  // truncated inside the counts
    EXPECT(test::throws([&] { backend::read_expert_profile(path.c_str(), lay, "t"); }));
}

}  // namespace

int main() {
    return tona::test::run("test_expert_layout", [] {
        test::TempDir dir("expert-layout");
        test_layout();
        test_version1(dir.path().string());
        test_version2(dir.path().string());
    });
}
