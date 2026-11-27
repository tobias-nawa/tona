// Row reads through io_uring (the PLE table). Exit code 77 (skipped) where the kernel or the sandbox has no io_uring.
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "io/uring.h"
#include "tests/test_support.h"

namespace {

constexpr uint32_t ROW = 90;  // bytes per row of the real table
constexpr uint64_t BASE = 1000;
constexpr int64_t ROWS = 600;

// File with a header of BASE bytes and ROWS rows whose bytes are derived from the row number
void write_table(const std::filesystem::path& path) {
    std::vector<uint8_t> bytes(BASE + ROWS * ROW, 0xEE);
    for (int64_t r = 0; r < ROWS; r++)
        for (uint32_t i = 0; i < ROW; i++) bytes[BASE + r * ROW + i] = (uint8_t)(r * 7 + i);
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
}

bool row_ok(const uint8_t* got, int64_t r) {
    for (uint32_t i = 0; i < ROW; i++)
        if (got[i] != (uint8_t)(r * 7 + i)) return false;
    return true;
}

void test_reads(const std::filesystem::path& file) {
    tona::DirectRows rows(file.string(), BASE, ROW);
    EXPECT(rows.row_bytes() == ROW);
    // More rows than the ring is deep (depth 8): requests wait for free slots
    std::vector<int64_t> ask;
    ask.reserve(100);
    for (int64_t r = 0; r < 100; r++) ask.push_back((r * 37) % ROWS);
    std::vector<uint8_t> out(ask.size() * (size_t)ROW);
    tona::DirectRows small(file.string(), BASE, ROW, false, 8);
    small.read(ask.data(), ask.size(), out.data());
    for (size_t i = 0; i < ask.size(); i++) EXPECT(row_ok(&out[i * (size_t)ROW], ask[i]));
    const int64_t one[1] = {ROWS - 1};
    std::vector<uint8_t> last(ROW);
    rows.read(one, 1, last.data());
    EXPECT(row_ok(last.data(), ROWS - 1));
    rows.read(one, 0, last.data());  // nothing to read
    rows.evict();
    rows.read(one, 1, last.data());
    EXPECT(row_ok(last.data(), ROWS - 1));
}

void test_missing_file(const std::filesystem::path& dir) {
    EXPECT(tona::test::throws([&] { tona::DirectRows((dir / "missing.bin").string(), 0, ROW); }));
}

}  // namespace

int main() {
    bool skipped = false;
    const int rc = tona::test::run("test_uring", [&skipped] {
        tona::test::TempDir tmp("uring");
        const auto file = tmp.path() / "table.bin";
        write_table(file);
        try {
            test_reads(file);
        } catch (const std::exception& e) {
            if (!std::string(e.what()).contains("io_uring_setup")) throw;
            std::fprintf(stderr, "test_uring: skipped (%s)\n", e.what());
            skipped = true;
        }
        test_missing_file(tmp.path());
    });
    return skipped ? 77 : rc;
}
