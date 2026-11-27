// libFuzzer: untrusted bytes of the update path -> signature parser, manifest parser, URL and time parsers.
// Input layout: two bytes length of the signature part, the signature file text, then the manifest text. A rejected
// input is the expected result; crashes, sanitizer reports and escaping exceptions are findings.
// Build and run: scripts/fuzz.sh
#include <cstdint>
#include <cstring>
#include <ctime>
#include <format>
#include <string>
#include <string_view>

#include "third_party/nlohmann/json.hpp"
#include "update/crypto.h"
#include "update/manifest.h"
#include "update/minisig.h"
#include "update/url.h"

namespace {

using namespace tona::update;

// Valid trusted comment for whatever seq and version the text claims, so the checks behind it get reached
std::string matching_comment(std::string_view manifest) {
    const auto j = nlohmann::json::parse(manifest.begin(), manifest.end(), nullptr, false);
    if (!j.is_object() || !j.contains("seq") || !j["seq"].is_number_unsigned() || !j.contains("version") ||
        !j["version"].is_string())
        return "";
    return std::format("tona schema=1 seq={} version={}", j["seq"].get<uint64_t>(), j["version"].get<std::string>());
}

void fuzz_signature(std::string_view sig, std::string_view manifest) {
    TrustedKey key;
    std::memset(key.id.data(), 0x11, key.id.size());
    std::memset(key.pub.data(), 0x22, key.pub.size());
    (void)verify_minisig(manifest, sig, {key});
    (void)parse_public_key(sig);
    (void)base64_decode_strict(sig);
}

void fuzz_manifest(std::string_view manifest) {
    ManifestContext ctx;
    ctx.now = 1;
    (void)parse_manifest(manifest, matching_comment(manifest), ctx);
    std::time_t t = 0;
    (void)parse_utc_time(manifest.substr(0, 20), t);
    (void)parse_url(manifest);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) return 0;
    const size_t sig_len = std::min<size_t>(static_cast<size_t>(data[0]) << 8 | data[1], size - 2);
    const std::string_view sig(reinterpret_cast<const char*>(data) + 2, sig_len);
    const std::string_view manifest(reinterpret_cast<const char*>(data) + 2 + sig_len, size - 2 - sig_len);
    fuzz_signature(sig, manifest);
    fuzz_manifest(manifest);
    return 0;
}
