// Read-ahead of the n-gram rows: the page ranges a pass's rows touch (rounded to pages, each page once, neighbours
// merged), and the hint itself on a real file mapping
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <vector>

#include "backend/ple_prefetch.h"
#include "tests/test_support.h"

namespace pf = tona::backend::ple_prefetch;

namespace {

void test_rows_round_to_their_pages() {
    const int32_t rows[] = {0, 1, 11};  // 90-byte rows in 1000-byte pages: page 0, page 0 again, page 0 to 1
    const auto r = pf::ranges(0, 90, rows, 3, 1000);
    EXPECT(r.size() == 1 && r[0].begin == 0 && r[0].end == 2000);
}

void test_a_base_off_the_page_rounds_down() {
    const int32_t rows[] = {0};
    const auto r = pf::ranges(1500, 90, rows, 1, 1000);
    EXPECT(r.size() == 1 && r[0].begin == 1000 && r[0].end == 2000);
}

void test_duplicates_once_and_gaps_split() {
    const int32_t rows[] = {50, 0, 50, 0};  // 100-byte rows: pages 5 and 0, each twice
    const auto r = pf::ranges(0, 100, rows, 4, 1000);
    EXPECT(r.size() == 2);
    EXPECT(r[0].begin == 0 && r[0].end == 1000);
    EXPECT(r[1].begin == 5000 && r[1].end == 6000);
}

void test_the_hint_succeeds_on_a_file_mapping() {
#ifdef _WIN32
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const auto page = static_cast<size_t>(si.dwPageSize);
    tona::test::TempDir tmp("ple-prefetch");
    const std::string path = (tmp.path() / "map.bin").string();
    const std::vector<char> data(page * 3, 'x');
    std::ofstream(path, std::ios::binary).write(data.data(), static_cast<std::streamsize>(data.size()));
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                           nullptr);
    EXPECT(f != INVALID_HANDLE_VALUE);
    if (f == INVALID_HANDLE_VALUE) return;
    HANDLE map = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    EXPECT(map != nullptr);
    void* m = map ? MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0) : nullptr;
    EXPECT(m != nullptr);
    if (m != nullptr) {
        const int32_t rows[] = {0, 2};
        EXPECT(pf::prefetch(static_cast<const uint8_t*>(m), page, rows, 2) == 0);
        UnmapViewOfFile(m);
    }
    if (map) CloseHandle(map);
    CloseHandle(f);
#else
    FILE* f = std::tmpfile();
    EXPECT(f != nullptr);
    if (f == nullptr) return;
    const auto page = (size_t)sysconf(_SC_PAGESIZE);
    const std::vector<char> data(page * 3, 'x');
    EXPECT(std::fwrite(data.data(), 1, data.size(), f) == data.size());
    std::fflush(f);
    void* m = mmap(nullptr, data.size(), PROT_READ, MAP_SHARED, fileno(f), 0);
    EXPECT(m != MAP_FAILED);
    if (m != MAP_FAILED) {
        const int32_t rows[] = {0, 2};
        EXPECT(pf::prefetch(static_cast<const uint8_t*>(m), page, rows, 2) == 0);
        munmap(m, data.size());
    }
    std::fclose(f);
#endif
}

}  // namespace

int main() {
    return tona::test::run("ple prefetch", [] {
        test_rows_round_to_their_pages();
        test_a_base_off_the_page_rounds_down();
        test_duplicates_once_and_gaps_split();
        test_the_hint_succeeds_on_a_file_mapping();
    });
}
