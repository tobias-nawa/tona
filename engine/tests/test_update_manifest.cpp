// Manifest parsing of the updater: a valid manifest, and each way the schema, the paths, the limits or the context
// (platform, seq, expiry, trusted comment) can be violated.
#include <functional>
#include <string>

#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "update/manifest.h"

using namespace tona::test;
namespace {

// 2026-11-05T00:00:00Z
constexpr std::time_t kNow = 1793836800;

up::ManifestResult parse(const json& manifest, uint64_t last_seq = 0) {
    // The trusted comment a signer would have written; fields of the wrong type fall back to the valid ones
    const std::string version = manifest["version"].is_string() ? manifest["version"].get<std::string>() : "0.3.0";
    const uint64_t seq = manifest["seq"].is_number_unsigned() ? manifest["seq"].get<uint64_t>() : 3;
    up::ManifestContext ctx;
    ctx.now = kNow;
    ctx.last_seq = last_seq;
    return up::parse_manifest(manifest.dump(), comment_for(seq, version), ctx);
}

up::ManifestResult parse_with(const json& manifest, const up::ManifestContext& ctx) {
    return up::parse_manifest(manifest.dump(), comment_for(3, "0.3.0"), ctx);
}

json base() { return make_release("0.3.0", 3).manifest; }

// The manifest after `edit` must be rejected with a message containing `why`
void rejects(const std::string& label, const std::function<void(json&)>& edit, const std::string& why) {
    json m = base();
    edit(m);
    const auto r = parse(m);
    if (r.ok || !r.error.contains(why))
        std::fprintf(stderr, "case \"%s\": ok=%d error=\"%s\" (wanted \"%s\")\n", label.c_str(), r.ok, r.error.c_str(),
                     why.c_str());
    EXPECT(!r.ok);
    EXPECT(r.error.contains(why));
}

void path_of_first_file(json& m, const std::string& path) { m["files"][0]["path"] = path; }

void test_valid() {
    const auto r = parse(base());
    EXPECT(r.ok);
    EXPECT(r.manifest.version == "0.3.0");
    EXPECT(r.manifest.seq == 3);
    EXPECT(r.manifest.files.size() == 3);
    EXPECT(r.manifest.lock_id() == "888888888888");
    EXPECT(r.manifest.total_bytes() > 0);
    EXPECT(r.manifest.files[0].mode == 0755 || r.manifest.files[0].mode == 0644);
    EXPECT(!r.manifest.security);
}

void test_paths() {
    rejects("dotdot", [](json& m) { path_of_first_file(m, "bin/.."); }, "bad path");
    rejects("dotdot dir", [](json& m) { path_of_first_file(m, "../tona-server"); }, "bad path");
    rejects("dot", [](json& m) { path_of_first_file(m, "bin/."); }, "bad path");
    rejects("hidden", [](json& m) { path_of_first_file(m, "bin/.x"); }, "bad path");
    rejects("absolute", [](json& m) { path_of_first_file(m, "/bin/tona-server"); }, "two segments");
    rejects("three segments", [](json& m) { path_of_first_file(m, "bin/sub/tona-server"); }, "two segments");
    rejects("one segment", [](json& m) { path_of_first_file(m, "tona-server"); }, "two segments");
    rejects("other area", [](json& m) { path_of_first_file(m, "etc/tona-server"); }, "bad path");
    rejects("space", [](json& m) { path_of_first_file(m, "bin/tona server"); }, "bad path");
    rejects("semicolon", [](json& m) { path_of_first_file(m, "bin/a;b"); }, "bad path");
    rejects("backslash", [](json& m) { path_of_first_file(m, "bin/a\\b"); }, "bad path");
    rejects("non-ascii", [](json& m) { path_of_first_file(m, "bin/\xc3\xa4"); }, "bad path");
    rejects("empty name", [](json& m) { path_of_first_file(m, "bin/"); }, "bad path");
    rejects("long name", [](json& m) { path_of_first_file(m, "bin/" + std::string(101, 'a')); }, "bad path");
    rejects("duplicate path", [](json& m) { m["files"][1]["path"] = m["files"][0]["path"]; }, "duplicate path");
    rejects("duplicate asset", [](json& m) { m["files"][1]["asset"] = m["files"][0]["asset"]; }, "duplicate asset");
    rejects("asset with slash", [](json& m) { m["files"][0]["asset"] = "a/b"; }, "asset name");
    rejects(
        "no server",
        [](json& m) {
            m["files"][0]["path"] = "bin/other";
            m["files"][1]["path"] = "bin/other2";
        },
        "bin/tona-server is missing");
}

void test_platforms() {
    rejects(
        "unknown platform", [](json& m) { m["files"][0]["platform"] = "linux-x86_64-gfx1100"; }, "unknown platform");
    rejects("platform number", [](json& m) { m["files"][0]["platform"] = 1; }, "not a string");
    rejects("missing platform", [](json& m) { m["files"][0].erase("platform"); }, "missing key");
    rejects(
        "any and single platform share a path", [](json& m) { m["files"][3]["platform"] = "any"; },
        "both for any platform");
    rejects(
        "duplicate path on one platform",
        [](json& m) {
            m["files"][3]["platform"] = m["files"][1]["platform"];
            m["files"][3]["path"] = m["files"][1]["path"];
        },
        "duplicate path");
    // A client sees the files of its own platform only; the build's own platform always has a server
    up::ManifestContext ctx;
    ctx.now = kNow;
    const auto own = up::parse_manifest(base().dump(), comment_for(3, "0.3.0"), ctx);
    EXPECT(own.ok);
    for (const auto& f : own.manifest.files) EXPECT(f.platform == up::kPlatform || f.platform == "any");
    // The other platform's file is the one with the server: that client sees exactly it
    ctx.platform = up::kPlatform == up::kPlatformLinux ? "macos-arm64" : "linux-x86_64-gfx1200";
    const auto other = up::parse_manifest(base().dump(), comment_for(3, "0.3.0"), ctx);
    EXPECT(other.ok);
    EXPECT(other.manifest.files.size() == 1 && other.manifest.files[0].asset == "other-platform-tona-server");
    ctx.platform = "linux-x86_64-gfx1100";
    EXPECT(parse_with(base(), ctx).error.contains("bin/tona-server is missing"));
}

void test_file_fields() {
    rejects("mode 4755", [](json& m) { m["files"][0]["mode"] = "4755"; }, "mode");
    rejects("mode 0777", [](json& m) { m["files"][0]["mode"] = "0777"; }, "mode");
    rejects("mode number", [](json& m) { m["files"][0]["mode"] = 755; }, "not a string");
    rejects("size zero", [](json& m) { m["files"][0]["size"] = 0; }, "out of range");
    rejects("size over limit", [](json& m) { m["files"][0]["size"] = (5ull << 30); }, "out of range");
    rejects(
        "total over limit",
        [](json& m) {
            m["files"][0]["size"] = (3ull << 30);
            m["files"][1]["size"] = (3ull << 30);
        },
        "total size");
    rejects("negative size", [](json& m) { m["files"][0]["size"] = -1; }, "non-negative");
    rejects("float size", [](json& m) { m["files"][0]["size"] = 1.5; }, "non-negative");
    rejects("hash upper case", [](json& m) { m["files"][0]["sha256"] = std::string(64, 'A'); }, "sha256");
    rejects("hash short", [](json& m) { m["files"][0]["sha256"] = "ab"; }, "sha256");
    rejects("unknown file key", [](json& m) { m["files"][0]["extra"] = 1; }, "unknown key");
    rejects("missing file key", [](json& m) { m["files"][0].erase("mode"); }, "missing key");
    rejects("no files", [](json& m) { m["files"] = json::array(); }, "1 to 256");
    rejects("files not array", [](json& m) { m["files"] = "x"; }, "1 to 256");
    rejects(
        "too many files",
        [](json& m) {
            for (int i = 0; i < 300; i++)
                m["files"].push_back({{"asset", "a" + std::to_string(i)},
                                      {"path", "bin/a" + std::to_string(i)},
                                      {"platform", "any"},
                                      {"size", 1},
                                      {"sha256", std::string(64, 'a')},
                                      {"mode", "0644"}});
        },
        "1 to 256");
}

void test_top_level_fields() {
    rejects("unknown key", [](json& m) { m["extra"] = 1; }, "unknown key");
    rejects("missing key", [](json& m) { m.erase("commit"); }, "missing key");
    rejects("schema 2", [](json& m) { m["schema"] = 2; }, "schema");
    rejects("schema string", [](json& m) { m["schema"] = "1"; }, "schema");
    rejects("product", [](json& m) { m["product"] = "other"; }, "product");
    rejects("top-level platform", [](json& m) { m["platform"] = "linux-x86_64-gfx1200"; }, "unknown key");
    rejects("version with slash", [](json& m) { m["version"] = "../1.0.0"; }, "version");
    rejects("version two parts", [](json& m) { m["version"] = "1.0"; }, "version");
    rejects("version suffix", [](json& m) { m["version"] = "1.0.0-rc1"; }, "version");
    rejects("version empty part", [](json& m) { m["version"] = "1..0"; }, "version");
    rejects("seq zero", [](json& m) { m["seq"] = 0; }, "out of range");
    rejects("seq huge", [](json& m) { m["seq"] = (1ull << 60); }, "out of range");
    rejects("seq string", [](json& m) { m["seq"] = "3"; }, "non-negative");
    rejects("commit short", [](json& m) { m["commit"] = "abc"; }, "commit");
    rejects("lock upper", [](json& m) { m["rocm_lock_sha256"] = std::string(64, 'F'); }, "rocm_lock");
    rejects("severity", [](json& m) { m["severity"] = "critical"; }, "severity");
    rejects("changelog escape", [](json& m) { m["changelog"] = "x\x1b[31mred"; }, "control character");
    rejects("changelog huge", [](json& m) { m["changelog"] = std::string(17000, 'x'); }, "too long");
    rejects("released bad", [](json& m) { m["released"] = "2026-02-30T00:00:00Z"; }, "UTC time");
    rejects("expires zone", [](json& m) { m["expires"] = "2027-05-02T00:00:00+01:00"; }, "UTC time");
    rejects("released after expires", [](json& m) { m["released"] = "2028-01-01T00:00:00Z"; }, "after it expires");
}

void test_freshness_and_context() {
    EXPECT(!parse(base(), 4).ok);  // rollback
    EXPECT(parse(base(), 4).error.contains("older"));
    EXPECT(parse(base(), 3).ok);  // same seq is parsed; the caller reports "up to date"
    json expired = base();
    expired["expires"] = "2026-11-04T00:00:00Z";
    EXPECT(parse(expired).error.contains("expired"));
    up::ManifestContext ctx;
    ctx.now = kNow;
    const std::string text = base().dump();
    EXPECT(!up::parse_manifest(text, comment_for(4, "0.3.0"), ctx).ok);  // trusted comment of another seq
    EXPECT(!up::parse_manifest(text, comment_for(3, "0.3.1"), ctx).ok);
    EXPECT(!up::parse_manifest(text, "", ctx).ok);
    EXPECT(up::parse_manifest(text, comment_for(3, "0.3.0") + " ", ctx).error.contains("trusted comment"));
    json security = base();
    security["severity"] = "security";
    EXPECT(parse(security).manifest.security);
}

void test_json_strictness() {
    up::ManifestContext ctx;
    ctx.now = kNow;
    const std::string good = base().dump();
    const std::string comment = comment_for(3, "0.3.0");
    const auto bad = [&](const std::string& text, const std::string& why) {
        const auto r = up::parse_manifest(text, comment, ctx);
        EXPECT(!r.ok);
        EXPECT(r.error.contains(why));
    };
    // The same key twice: nlohmann alone would keep the last one silently
    std::string dup = good;
    dup.insert(1, "\"seq\":2,");
    bad(dup, "duplicate key");
    std::string dup_nested = good;
    dup_nested.insert(dup_nested.find("\"asset\""), R"("asset":"x",)");
    bad(dup_nested, "duplicate key");
    bad(good + "x", "not valid JSON");
    bad(good + " {}", "not valid JSON");
    bad("[]", "not an object");
    bad("", "not valid JSON");
    bad("{\"a\":" + std::string(50, '[') + std::string(50, ']') + "}", "nests too deeply");
    bad(std::string(2 << 20, ' '), "larger than 1 MiB");
    std::string bad_utf8 = good;
    bad_utf8.replace(bad_utf8.find("Test"), 4, "\xff\xfe\xfd\xfc");
    bad(bad_utf8, "not valid JSON");
}

void test_time_parser() {
    std::time_t t = 0;
    EXPECT(up::parse_utc_time("2027-05-02T00:00:00Z", t));
    EXPECT(t == 1809216000);
    EXPECT(up::parse_utc_time("2024-02-29T23:59:59Z", t));
    for (const char* s :
         {"", "2027-05-02T00:00:00", "2027-05-02 00:00:00Z", "2027-13-02T00:00:00Z", "2027-02-29T00:00:00Z",
          "2027-05-02T24:00:00Z", " 027-05-02T00:00:00Z", "+027-05-02T00:00:00Z", "2027-05-02T00:00:00z",
          "2027-05-02T00:00:00ZZ", "2027-5-02T000:00:00Z"})
        EXPECT(!up::parse_utc_time(s, t));
}

// scripts/release_manifest.py wrote this file (tools/tests/test_release_manifest.py checks it is current)
void test_manifest_written_by_the_release_script(const std::string& data_dir) {
    up::ManifestContext ctx;
    ctx.now = kNow;
    ctx.platform = "linux-x86_64-gfx1200";
    const std::string text = read_file(data_dir + "/created-manifest.json");
    const auto r = up::parse_manifest(text, comment_for(3, "0.3.0"), ctx);
    if (!r.ok) std::fprintf(stderr, "%s\n", r.error.c_str());
    EXPECT(r.ok);
    EXPECT(r.manifest.files.size() == 3);
    EXPECT(r.manifest.lock_id() == "888888888888");
    EXPECT(r.manifest.changelog == "Test release.");
    EXPECT(r.manifest.expires > r.manifest.released);
    // The same file read by a Mac: its own three files, no runtime
    ctx.platform = "macos-arm64";
    const auto mac = up::parse_manifest(text, comment_for(3, "0.3.0"), ctx);
    EXPECT(mac.ok);
    EXPECT(mac.manifest.files.size() == 3 && !mac.manifest.has_runtime());
}

}  // namespace

int main(int argc, char** argv) {
    return tona::test::guard("update_manifest", [&] {
        if (argc < 2) return 2;
        const std::string data_dir = argv[1];
        return run("update_manifest", [&data_dir] {
            test_valid();
            test_paths();
            test_platforms();
            test_file_fields();
            test_top_level_fields();
            test_freshness_and_context();
            test_json_strictness();
            test_time_parser();
            test_manifest_written_by_the_release_script(data_dir);
        });
    });
}
