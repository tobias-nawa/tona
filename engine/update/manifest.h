// The signed release manifest (schema 1). Parsed only after the signature check; every deviation from the schema is an
// error: unknown or duplicate keys, wrong types, bad paths, sizes beyond the limits.
#pragma once

#include <cstdint>
#include <ctime>
#include <string>
#include <string_view>
#include <vector>

namespace tona::update {

constexpr size_t kMaxManifestBytes = 1 << 20;
constexpr size_t kMaxFiles = 256;
constexpr uint64_t kMaxTotalBytes = 4ull << 30;
constexpr std::string_view kProduct = "tona";
constexpr std::string_view kPlatformLinux = "linux-x86_64-gfx1200";
constexpr std::string_view kPlatformMacos = "macos-arm64";
constexpr std::string_view kPlatformAny = "any";  // a file that is the same on every platform (the `tona` script)
#if defined(__APPLE__)
constexpr std::string_view kPlatform = kPlatformMacos;
#else
constexpr std::string_view kPlatform = kPlatformLinux;
#endif

struct ManifestFile {
    std::string asset;     // flat name of the release asset
    std::string platform;  // kPlatformLinux, kPlatformMacos or kPlatformAny
    std::string area;      // "bin" or "runtime"
    std::string name;      // file name inside the area
    uint64_t size = 0;
    std::string sha256;  // 64 lowercase hex
    int mode = 0;        // 0755 or 0644
};

struct Manifest {
    std::string version;  // digits and dots, safe as a directory name
    uint64_t seq = 0;
    std::time_t released = 0;
    std::time_t expires = 0;
    std::string commit;
    std::string rocm_lock_sha256;
    bool security = false;
    std::string changelog;
    std::vector<ManifestFile> files;  // only the files of the platform asked for and of "any"

    // Whether this platform has a shared runtime directory (the ROCm libraries); the Mac release has none
    bool has_runtime() const;
    // First 12 hex characters of rocm_lock_sha256: names the shared runtime directory
    std::string lock_id() const { return rocm_lock_sha256.substr(0, 12); }
    uint64_t total_bytes() const;
};

// What the caller already knows about the world; the manifest is checked against it
struct ManifestContext {
    std::string platform = std::string(kPlatform);
    std::time_t now = 0;
    uint64_t last_seq = 0;  // highest seq accepted so far; a lower one is a rollback
};

struct ManifestResult {
    bool ok = false;
    std::string error;
    Manifest manifest;
};

// The trusted comment every signature of schema 1 carries: exactly "tona schema=1 seq=<n> version=<v>"
std::string expected_trusted_comment(const Manifest& m);

// Parses and validates `text`. `trusted_comment` is the one the signature covers; it must match the manifest.
ManifestResult parse_manifest(std::string_view text, std::string_view trusted_comment, const ManifestContext& ctx);

// Parses "YYYY-MM-DDTHH:MM:SSZ" (UTC, nothing else); false for any other shape or an impossible date
bool parse_utc_time(std::string_view s, std::time_t& out);

}  // namespace tona::update
