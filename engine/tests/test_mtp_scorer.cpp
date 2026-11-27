// The scorer cache of the dynamic MTP rows: fingerprint of the head, rejection of a file of another head, of another
// rank and of a truncated one, no temporary file left behind (host only, no model, no GPU).
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "backend/mtp_scorer.h"
#include "tests/test_guard.h"

using namespace tona::backend;

namespace {

int fails = 0;
#define EXPECT(c)                                                                    \
    do {                                                                             \
        if (!(c)) std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c), fails++; \
    } while (0)

constexpr int D = 16, RANK = 16;
constexpr size_t VOCAB = 300, ROW = 24;

MtpScorer make_fake() {
    MtpScorer s;
    s.rank = RANK;
    s.basis_t.resize((size_t)RANK * D), s.scale.resize(VOCAB), s.rows.resize(VOCAB * RANK);
    for (size_t i = 0; i < s.basis_t.size(); i++) s.basis_t[i] = (float)i * 0.5f;
    for (size_t i = 0; i < VOCAB; i++) s.scale[i] = 1.0f + (float)i;
    for (size_t i = 0; i < s.rows.size(); i++) s.rows[i] = (int8_t)(i % 100);
    return s;
}

uint64_t fingerprint_of(const std::vector<uint8_t>& head, uint32_t type = 7) {
    return head_fingerprint(type, head.size(), VOCAB,
                            [&](size_t row, uint8_t* dst, size_t n) { std::copy_n(&head[row * ROW], n, dst); });
}

void test_fingerprint() {
    std::vector<uint8_t> head(VOCAB * ROW);
    for (size_t i = 0; i < head.size(); i++) head[i] = (uint8_t)(i * 31 + 7);
    const uint64_t fp = fingerprint_of(head);
    EXPECT(fp == fingerprint_of(head));
    EXPECT(fp != fingerprint_of(head, 8));  // type
    for (size_t row : {size_t{0}, size_t{64}, size_t{128}, VOCAB - 1}) {
        std::vector<uint8_t> other = head;
        other[row * ROW + 3] ^= 1;
        EXPECT(fp != fingerprint_of(other));  // a sampled row
    }
    EXPECT(fp !=
           head_fingerprint(7, head.size() + ROW, VOCAB, [&](size_t, uint8_t* d, size_t n) { std::fill_n(d, n, 0); }));
}

// Files of another head, rank or vocabulary are not read
void test_mismatch(const std::string& path, const MtpScorer& s, uint64_t fp) {
    const MtpScorer ok = read_mtp_scorer(path, RANK, D, VOCAB, fp);
    EXPECT(ok.rank == RANK && ok.rows == s.rows && ok.scale == s.scale && ok.basis_t == s.basis_t);
    EXPECT(read_mtp_scorer(path, RANK, D, VOCAB, fp + 1).rank == 0);             // another head
    EXPECT(read_mtp_scorer(path, RANK, D, VOCAB, fp ^ (1ULL << 40)).rank == 0);  // high half of the fingerprint
    EXPECT(read_mtp_scorer(path, 32, D, VOCAB, fp).rank == 0);                   // another rank
    EXPECT(read_mtp_scorer(path, RANK, D, VOCAB + 1, fp).rank == 0);             // another vocabulary
}

// Truncated, too long, and the magic of format version 1
void test_damaged(const std::string& path, const MtpScorer& s, uint64_t fp) {
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    EXPECT(read_mtp_scorer(path, RANK, D, VOCAB, fp).rank == 0);
    EXPECT(write_mtp_scorer(path, s, D, VOCAB, fp));
    std::ofstream(path, std::ios::binary | std::ios::app).put('x');
    EXPECT(read_mtp_scorer(path, RANK, D, VOCAB, fp).rank == 0);
    EXPECT(write_mtp_scorer(path, s, D, VOCAB, fp));
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    const uint32_t old_magic = 0x31435354;
    f.write(reinterpret_cast<const char*>(&old_magic), 4);
    f.close();
    EXPECT(read_mtp_scorer(path, RANK, D, VOCAB, fp).rank == 0);
}

void test_file(const std::filesystem::path& dir) {
    const std::string path = (dir / "scorer").string();
    const MtpScorer s = make_fake();
    const uint64_t fp = 0x1122334455667788ULL;
    EXPECT(write_mtp_scorer(path, s, D, VOCAB, fp));
    EXPECT(!std::filesystem::exists(path + ".tmp"));
    test_mismatch(path, s, fp);
    EXPECT(read_mtp_scorer((dir / "missing").string(), RANK, D, VOCAB, fp).rank == 0);
    test_damaged(path, s, fp);
    // not writable: false, nothing left behind
    EXPECT(!write_mtp_scorer((dir / "no" / "such" / "scorer").string(), s, D, VOCAB, fp));
    EXPECT(!std::filesystem::exists(dir / "no"));
}

}  // namespace

int main() {
    return tona::test::guard("mtp_scorer", [&] {
        const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / ("tona-test-mtp-scorer-" + std::to_string(getpid()));
        std::filesystem::create_directories(dir);
        test_fingerprint();
        test_file(dir);
        std::filesystem::remove_all(dir);
        if (fails) return 1;
        std::printf("test_mtp_scorer: ok\n");
        return 0;
    });
}
