// Checkpoint files of the prompt cache: write and read back, and what a
// damaged, foreign or truncated file must do.
#ifndef _WIN32
#include <sys/stat.h>
#endif

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "server/checkpoint_file.h"
#include "tests/test_support.h"

using tona::backend::HostSnap;
using namespace tona::server;
namespace fs = std::filesystem;

namespace {

bool g_has_logits = false;  // out parameter of read_checkpoint_tokens, read where a test cares
constexpr uint64_t FP = 0x1234abcd5678ef01ull;
constexpr size_t VOCAB = 50, STATE = 40;

HostSnap make_snap() {
    HostSnap h;
    h.tokens = {5, 6, 7, 8, 9, 10};
    h.pos = 6, h.commit = 3, h.parity = 1;
    for (size_t i = 0; i < STATE; i++) h.state.push_back((uint8_t)(i + 1));
    for (size_t i = 0; i < 100; i++) h.kv.push_back((uint8_t)(i * 3));
    for (size_t i = 0; i < 7; i++) h.pool.push_back((uint8_t)(200 + i));
    h.dring = {};  // an empty part is legal
    return h;
}

std::vector<float> make_logits() {
    std::vector<float> l(VOCAB);
    for (size_t i = 0; i < VOCAB; i++) l[i] = (float)i * 0.5f - 7.0f;
    return l;
}

std::vector<char> slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

void spit(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream(path, std::ios::binary).write(bytes.data(), (std::streamsize)bytes.size());
}

bool read_back(const std::string& path, const HostSnap& want, HostSnap& got, std::vector<float>& logits) {
    return read_checkpoint(path, {FP, want.tokens, VOCAB, STATE}, logits, got);
}

void test_round_trip(const std::string& dir) {
    const HostSnap h = make_snap();
    const auto logits = make_logits();
    const std::string path = dir + "/a.bin";
    EXPECT(write_checkpoint(path, FP, logits, h));
    EXPECT(fs::file_size(path) == sizeof(DiskHead) + sizeof(int32_t) * 6 + sizeof(float) * VOCAB + STATE + 100 + 7);
    HostSnap got;
    std::vector<float> l2;
    EXPECT(read_back(path, h, got, l2));
    EXPECT(got.tokens == h.tokens && got.pos == 6 && got.commit == 3 && got.parity == 1);
    EXPECT(got.state == h.state && got.kv == h.kv && got.pool == h.pool && got.dring.empty() && l2 == logits);
    std::vector<int32_t> tokens;
    EXPECT(read_checkpoint_tokens(path, FP, tokens, g_has_logits) && tokens == h.tokens && g_has_logits);
}

// A checkpoint inside a prompt has no logits: the file holds none, the head says so, and it reads back as such
void test_without_logits(const std::string& dir) {
    const HostSnap h = make_snap();
    const std::string path = dir + "/nologits.bin";
    EXPECT(write_checkpoint(path, FP, {}, h));
    EXPECT(fs::file_size(path) == sizeof(DiskHead) + sizeof(int32_t) * 6 + STATE + 100 + 7);
    std::vector<int32_t> tokens;
    EXPECT(read_checkpoint_tokens(path, FP, tokens, g_has_logits) && tokens == h.tokens && !g_has_logits);
    HostSnap got;
    std::vector<float> l2(3, 1.0f);
    EXPECT(read_back(path, h, got, l2) && l2.empty() && got.state == h.state);
    // Another logits count than none or the vocabulary is still refused
    EXPECT(write_checkpoint(path, FP, std::vector<float>(VOCAB - 1), h));
    EXPECT(!read_back(path, h, got, l2));
    fs::remove(path);
}

#ifndef _WIN32
// A checkpoint is a function of the prompt text: only the owner may read it, whatever the umask
void test_file_is_private(const std::string& dir) {
    const mode_t old = umask(0);
    const std::string path = dir + "/private.bin";
    EXPECT(write_checkpoint(path, FP, make_logits(), make_snap()));
    umask(old);
    EXPECT((fs::status(path).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
    EXPECT((fs::status(path).permissions() & fs::perms::owner_read) != fs::perms::none);
    fs::remove(path);
}

// A link planted at the temporary name (a directory others can write to) must not make the writer truncate or
// overwrite the file it points to
void test_link_is_not_followed(const std::string& dir) {
    const std::string victim = dir + "/victim.txt", path = dir + "/linked.bin";
    spit(victim, {'k', 'e', 'e', 'p'});
    fs::create_symlink(victim, path);
    EXPECT(write_checkpoint(path, FP, make_logits(), make_snap()));
    EXPECT((slurp(victim) == std::vector<char>{'k', 'e', 'e', 'p'}));
    EXPECT(fs::is_regular_file(fs::symlink_status(path)));  // replaced by a file of its own
    HostSnap got;
    std::vector<float> l2;
    EXPECT(read_back(path, make_snap(), got, l2));
    // A dangling link, too
    fs::remove(path);
    fs::create_symlink(dir + "/does-not-exist.txt", path);
    EXPECT(write_checkpoint(path, FP, make_logits(), make_snap()));
    EXPECT(!fs::exists(dir + "/does-not-exist.txt"));
    fs::remove(path), fs::remove(victim);
}
#endif

// Others may not be able to write into the checkpoint directory
void test_directory_is_secured(const std::string& base) {
    std::string why;
    const std::string made = base + "/made", ours = base + "/ours", foreign = base + "/foreign";
    for (const auto& d : {made, ours, foreign}) fs::create_directories(d);
#ifdef _WIN32
    EXPECT(secure_checkpoint_dir(made, true, why));
    EXPECT(!secure_checkpoint_dir(base + "/missing", false, why));
#else
    for (const auto& d : {made, ours, foreign}) fs::permissions(d, fs::perms::all);  // 0777
    // just created by the server: tightened
    EXPECT(secure_checkpoint_dir(made, true, why));
    EXPECT((fs::status(made).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
    // ours with nothing but checkpoint files: tightened
    spit(ours + "/x.bin", {'1'});
    EXPECT(secure_checkpoint_dir(ours, false, why));
    EXPECT((fs::status(ours).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
    // a foreign file in it: it may not be touched, and a directory others can write to is refused
    spit(foreign + "/notes.txt", {'1'});
    EXPECT(!secure_checkpoint_dir(foreign, false, why) && !why.empty());
    EXPECT((fs::status(foreign).permissions() & fs::perms::others_write) != fs::perms::none);
    // the same directory without write access for others is fine
    fs::permissions(foreign, fs::perms::owner_all | fs::perms::group_read | fs::perms::others_read,
                    fs::perm_options::replace);
    EXPECT(secure_checkpoint_dir(foreign, false, why));
    EXPECT(!secure_checkpoint_dir(base + "/missing", false, why));
#endif
}

void test_file_name() {
    const std::vector<int32_t> a{1, 2, 3}, b{1, 2, 4};
    const std::string pa = checkpoint_path("/d", FP, a);
    EXPECT(pa.rfind("/d/1234abcd5678ef01-", 0) == 0 && pa.size() == 3 + 16 + 1 + 16 + 4 && pa.ends_with(".bin"));
    EXPECT(pa == checkpoint_path("/d", FP, a) && pa != checkpoint_path("/d", FP, b));
    EXPECT(pa != checkpoint_path("/d", FP + 1, a));
    // FNV-1a of the empty input is the basis
    EXPECT(fnv(FNV_BASIS, nullptr, 0) == FNV_BASIS);
    const char x = 'a';
    EXPECT(fnv(FNV_BASIS, &x, 1) == 0xaf63dc4c8601ec8cull);  // FNV-1a 64 of "a"
}

void test_foreign_files_are_refused(const std::string& dir) {
    const HostSnap h = make_snap();
    const std::string path = dir + "/b.bin";
    write_checkpoint(path, FP, make_logits(), h);
    HostSnap got;
    std::vector<float> l;
    std::vector<int32_t> tokens;
    EXPECT(!read_checkpoint(path, {FP + 1, h.tokens, VOCAB, STATE}, l,
                            got));  // another model
    EXPECT(!read_checkpoint_tokens(path, FP + 1, tokens, g_has_logits));
    EXPECT(!read_checkpoint(path, {FP, h.tokens, VOCAB + 1, STATE}, l,
                            got));  // another vocabulary
    EXPECT(!read_checkpoint(path, {FP, h.tokens, VOCAB, STATE + 1}, l,
                            got));  // another state layout
    auto other = h.tokens;
    other[2] = 99;
    EXPECT(!read_checkpoint(path, {FP, other, VOCAB, STATE}, l,
                            got));  // other tokens of the same length
    other.pop_back();
    EXPECT(!read_checkpoint(path, {FP, other, VOCAB, STATE}, l,
                            got));  // another number of tokens
    auto bytes = slurp(path);
    for (size_t at : {size_t(0), size_t(4)}) {  // magic, version
        auto bad = bytes;
        bad[at] ^= 1;
        spit(dir + "/c.bin", bad);
        EXPECT(!read_checkpoint(dir + "/c.bin", {FP, h.tokens, VOCAB, STATE}, l, got));
        EXPECT(!read_checkpoint_tokens(dir + "/c.bin", FP, tokens, g_has_logits));
    }
    EXPECT(!read_checkpoint(dir + "/missing.bin", {FP, h.tokens, VOCAB, STATE}, l, got));
    EXPECT(!read_checkpoint_tokens(dir + "/missing.bin", FP, tokens, g_has_logits));
}

void test_truncated_and_extended_files(const std::string& dir) {
    const HostSnap h = make_snap();
    const std::string path = dir + "/d.bin";
    write_checkpoint(path, FP, make_logits(), h);
    const auto bytes = slurp(path);
    HostSnap got;
    std::vector<float> l;
    std::vector<int32_t> tokens;
    const std::string cut = dir + "/cut.bin";
    // Header only, inside the tokens, inside the logits, inside the state, one
    // byte short
    for (size_t keep : {size_t(0), size_t(10), sizeof(DiskHead), sizeof(DiskHead) + 8, sizeof(DiskHead) + 24 + 60,
                        sizeof(DiskHead) + 24 + 200 + 10, bytes.size() - 1}) {
        spit(cut, std::vector<char>(bytes.begin(), bytes.begin() + (std::ptrdiff_t)keep));
        EXPECT(!read_checkpoint(cut, {FP, h.tokens, VOCAB, STATE}, l, got));
    }
    // The tokens alone can still be read from a file that lost only its tail; the
    // full read refuses it
    spit(cut, std::vector<char>(bytes.begin(), bytes.end() - 1));
    EXPECT(read_checkpoint_tokens(cut, FP, tokens, g_has_logits) && tokens == h.tokens);
    spit(cut, std::vector<char>(bytes.begin(), bytes.begin() + (std::ptrdiff_t)sizeof(DiskHead) + 8));
    EXPECT(!read_checkpoint_tokens(cut, FP, tokens, g_has_logits));
    auto longer = bytes;
    longer.push_back(0);
    spit(cut, longer);
    EXPECT(!read_checkpoint(cut, {FP, h.tokens, VOCAB, STATE}, l,
                            got));  // size must match exactly
}

void test_write_failure(const std::string& dir) {
    EXPECT(!write_checkpoint(dir + "/no/such/dir/x.bin", FP, make_logits(), make_snap()));
}

// A head that announces more tokens than the file holds must not make the scan
// allocate for them
void test_absurd_token_count(const std::string& dir) {
    const HostSnap h = make_snap();
    const std::string path = dir + "/e.bin";
    write_checkpoint(path, FP, make_logits(), h);
    auto bytes = slurp(path);
    const uint64_t huge = 1ull << 50;
    std::memcpy(&bytes[offsetof(DiskHead, ntok)], &huge, sizeof huge);
    spit(path, bytes);
    std::vector<int32_t> tokens;
    EXPECT(!read_checkpoint_tokens(path, FP, tokens, g_has_logits));
}

}  // namespace

int main() {
    return tona::test::run("test_checkpoint_file", [] {
        tona::test::TempDir tmp("checkpoint");
        const std::string dir = tmp.path().string();
        test_round_trip(dir);
        test_without_logits(dir);
#ifndef _WIN32
        test_file_is_private(dir);
        test_link_is_not_followed(dir);
#endif
        test_directory_is_secured(dir);
        test_file_name();
        test_foreign_files_are_refused(dir);
        test_truncated_and_extended_files(dir);
        test_write_failure(dir);
        test_absurd_token_count(dir);
    });
}
