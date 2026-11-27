// Tests for the GGUF loader with hand-written files: values, tensors, split files, broken files.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "model/gguf.h"
#include "tests/test_support.h"

namespace fs = std::filesystem;

using tona::test::GgufWriter;
using Writer = GgufWriter;
using tona::test::throws;

static void write_values_file(const fs::path& path) {
    Writer w;
    w.header(2, 5);
    w.kv_str("general.architecture", "qwen4exp");
    w.kv_u32("qwen4exp.block_count", 48);
    w.kv_f32("qwen4exp.rope.freq_base", 1e7f);
    w.kv_arr_i32("qwen4exp.attention.compress_ratios", {0, 0, 0, 4});
    w.kv_u16("tiny", 7);
    w.tensor("a.weight", {4, 2}, 0, 0);    // F32, 32 bytes
    w.tensor("b.weight", {32, 1}, 8, 32);  // Q8_0, 34 bytes
    w.align();
    for (int i = 0; i < 8; i++) w.put<float>((float)i + 0.5f);
    for (int i = 0; i < 34; i++) w.put<uint8_t>((uint8_t)(100 + i));
    w.save(path);
}

static void check_values(const tona::Gguf& g) {
    EXPECT(g.version() == 3);
    EXPECT(g.get_string("general.architecture") == "qwen4exp");
    EXPECT(g.get_uint("qwen4exp.block_count") == 48);
    EXPECT(g.get_uint("tiny") == 7);
    EXPECT(g.get_float("qwen4exp.rope.freq_base") == 1e7);
    EXPECT((g.get_int_array("qwen4exp.attention.compress_ratios") == std::vector<int64_t>{0, 0, 0, 4}));
    EXPECT(throws([&] { g.get_uint("general.architecture"); }));
    EXPECT(throws([&] { g.get_uint("missing"); }));
}

static void check_tensors(const tona::Gguf& g) {
    EXPECT(g.tensors().size() == 2);
    const tona::TensorInfo* a = g.find("a.weight");
    EXPECT(a && a->bytes == 32 && a->elements() == 8);
    float f[8];
    std::memcpy(f, a->data, sizeof f);
    EXPECT(f[0] == 0.5f && f[7] == 7.5f);
    const tona::TensorInfo* b = g.find("b.weight");
    EXPECT(b && b->bytes == 34 && b->data[0] == 100 && b->data[33] == 133);
    EXPECT(g.find("c.weight") == nullptr);
}

static void test_values_and_tensors(const fs::path& dir) {
    write_values_file(dir / "a.gguf");
    tona::Gguf g((dir / "a.gguf").string());
    check_values(g);
    check_tensors(g);
}

// Split: part 1 with metadata and one tensor, part 2 with a tensor only
static void test_split(const fs::path& dir) {
    Writer w1;
    w1.header(1, 3);
    w1.kv_str("general.architecture", "qwen4exp");
    w1.kv_u16("split.count", 2);
    w1.kv_u16("split.tensors.count", 2);
    w1.tensor("x", {8}, 0, 0);
    w1.align();
    for (int i = 0; i < 8; i++) w1.put<float>(1.0f);
    w1.save(dir / "m-00001-of-00002.gguf");
    Writer w2;
    w2.header(1, 1);
    w2.kv_u16("split.no", 1);
    w2.tensor("y", {8}, 0, 0);
    w2.align();
    for (int i = 0; i < 8; i++) w2.put<float>(2.0f);
    w2.save(dir / "m-00002-of-00002.gguf");

    tona::Gguf g((dir / "m-00001-of-00002.gguf").string());
    EXPECT(g.files() == 2);
    EXPECT(g.tensors().size() == 2);
    const tona::TensorInfo* y = g.find("y");
    float v;
    std::memcpy(&v, y->data, 4);
    EXPECT(y && y->file == 1 && v == 2.0f);
    EXPECT(!g.has("split.no"));  // metadata only from part 1
    EXPECT(throws([&] { tona::Gguf((dir / "m-00002-of-00002.gguf").string()); }));
}

static void test_broken_files(const fs::path& dir) {
    Writer w;
    w.header(1, 0);
    w.tensor("z", {8}, 0, 0);
    w.align();
    w.put<float>(1.0f);  // 4 instead of 32 bytes of data
    w.save(dir / "short.gguf");
    EXPECT(throws([&] { tona::Gguf((dir / "short.gguf").string()); }));

    Writer u;
    u.header(1, 0);
    u.tensor("z", {8}, 99, 0);
    u.align();
    u.save(dir / "type.gguf");
    EXPECT(throws([&] { tona::Gguf((dir / "type.gguf").string()); }));

    Writer h;
    h.header(0, 1);
    h.str("truncated");
    h.save(dir / "header.gguf");
    EXPECT(throws([&] { tona::Gguf((dir / "header.gguf").string()); }));
    EXPECT(throws([&] { tona::Gguf((dir / "missing.gguf").string()); }));
}

// The alignment's type comes from the file: an integer of any width loads, anything else is rejected
static void test_alignment_type(const fs::path& dir) {
    Writer f;
    f.header(0, 1);
    f.kv_f32("general.alignment", 32.0f);
    f.save(dir / "align-f32.gguf");
    EXPECT(throws([&] { tona::Gguf((dir / "align-f32.gguf").string()); }));

    Writer i;
    i.header(0, 1);
    i.str("general.alignment");
    i.put<uint32_t>(11);  // GGUF type INT64
    i.put<int64_t>(32);
    i.save(dir / "align-i64.gguf");
    EXPECT(!throws([&] { tona::Gguf((dir / "align-i64.gguf").string()); }));
}

// TONA_Q6_K_T interleaves 8 rows: only 2-D layer matrices may have it, never tables read by row
static void test_tile_type_only_for_layer_matrices(const fs::path& dir) {
    auto one = [&](const char* name, uint64_t rows) {
        Writer w;
        w.header(1, 0);
        w.tensor(name, {256, rows}, 1019, 0);
        w.align();
        for (uint64_t i = 0; i < rows * 210; i++) w.put<uint8_t>(0);
        w.save(dir / "tile.gguf");
        return !throws([&] { tona::Gguf((dir / "tile.gguf").string()); });
    };
    EXPECT(one("blk.0.attn_q.weight", 8));
    EXPECT(!one("blk.0.attn_q.weight", 4));
    for (const char* name : {"token_embd.weight", "output.weight", "per_layer_token_embd.weight"})
        EXPECT(!one(name, 8));
}

int main() {
    return tona::test::guard("gguf", [&] {
        tona::test::TempDir tmp("gguf");
        const fs::path& dir = tmp.path();
        test_values_and_tensors(dir);
        test_split(dir);
        test_broken_files(dir);
        test_alignment_type(dir);
        test_tile_type_only_for_layer_matrices(dir);
        return tona::test::finish("test_gguf");
    });
}
