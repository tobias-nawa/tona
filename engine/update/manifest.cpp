#include "update/manifest.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <ctime>
#include <format>
#include <set>
#include <stdexcept>

#include "third_party/nlohmann/json.hpp"
#include "update/crypto.h"

namespace tona::update {

namespace {

using nlohmann::json;

constexpr int kMaxJsonDepth = 6;
constexpr size_t kMaxChangelogBytes = 16384;
constexpr uint64_t kMaxSeq = 1ull << 53;  // exact in a JSON double, so no parser can round it
constexpr size_t kMaxNameLength = 100;

struct Invalid : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// nlohmann::json keeps the last of two equal keys silently; a signed document must not be ambiguous, so the parse
// callback sees every key and rejects a repeat within one object.
json parse_no_duplicates(std::string_view text) {
    std::vector<std::set<std::string>> seen;
    const json::parser_callback_t cb = [&](int depth, json::parse_event_t event, json& parsed) {
        if (depth > kMaxJsonDepth) throw Invalid("manifest nests too deeply");
        if (event == json::parse_event_t::object_start) {
            seen.emplace_back();
        } else if (event == json::parse_event_t::object_end) {
            seen.pop_back();
        } else if (event == json::parse_event_t::key && !seen.back().insert(parsed.get<std::string>()).second) {
            throw Invalid("duplicate key \"" + parsed.get<std::string>() + "\"");
        }
        return true;
    };
    try {
        return json::parse(text.begin(), text.end(), cb, true, false);
    } catch (const json::exception& e) {
        throw Invalid(std::string("not valid JSON: ") + e.what());
    }
}

// The object must have exactly these keys
void expect_keys(const json& obj, std::initializer_list<const char*> keys, const char* what) {
    if (!obj.is_object()) throw Invalid(std::string(what) + " is not an object");
    for (const auto& [k, v] : obj.items()) {
        if (std::find_if(keys.begin(), keys.end(), [&](const char* w) { return k == w; }) == keys.end())
            throw Invalid(std::string("unknown key \"") + k + "\" in " + what);
    }
    for (const char* k : keys)
        if (!obj.contains(k)) throw Invalid(std::string("missing key \"") + k + "\" in " + what);
}

const std::string& text_of(const json& obj, const char* key) {
    const json& v = obj.at(key);
    if (!v.is_string()) throw Invalid(std::string("\"") + key + "\" is not a string");
    return v.get_ref<const std::string&>();
}

uint64_t number_of(const json& obj, const char* key, uint64_t min, uint64_t max) {
    const json& v = obj.at(key);
    if (!v.is_number_unsigned()) throw Invalid(std::string("\"") + key + "\" is not a non-negative integer");
    const uint64_t n = v.get<uint64_t>();
    if (n < min || n > max) throw Invalid(std::string("\"") + key + "\" is out of range");
    return n;
}

bool name_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
           c == '+' || c == '-';
}

// One path segment or asset name: [A-Za-z0-9._+-], no leading dot (which also excludes "." and ".."), bounded length
bool valid_name(std::string_view s) {
    return !s.empty() && s.size() <= kMaxNameLength && s[0] != '.' && std::all_of(s.begin(), s.end(), name_char);
}

bool valid_version(std::string_view v) {
    int groups = 1;
    size_t digits = 0;
    for (const char c : v) {
        if (c == '.') {
            if (digits == 0) return false;
            groups++;
            digits = 0;
        } else if (c >= '0' && c <= '9') {
            if (++digits > 6) return false;
        } else {
            return false;
        }
    }
    return groups == 3 && digits > 0;
}

std::time_t time_of(const json& obj, const char* key) {
    std::time_t t = 0;
    if (!parse_utc_time(text_of(obj, key), t)) throw Invalid(std::string("\"") + key + "\" is not a UTC time");
    return t;
}

ManifestFile file_of(const json& entry) {
    expect_keys(entry, {"asset", "path", "platform", "size", "sha256", "mode"}, "file entry");
    ManifestFile f;
    f.asset = text_of(entry, "asset");
    f.platform = text_of(entry, "platform");
    if (f.platform != kPlatformLinux && f.platform != kPlatformMacos && f.platform != kPlatformAny)
        throw Invalid("unknown platform \"" + f.platform + "\" of " + f.asset);
    if (!valid_name(f.asset)) throw Invalid("bad asset name \"" + f.asset + "\"");
    const std::string& path = text_of(entry, "path");
    const size_t slash = path.find('/');
    if (slash == std::string::npos || path.find('/', slash + 1) != std::string::npos)
        throw Invalid("path must have exactly two segments: " + path);
    f.area = path.substr(0, slash);
    f.name = path.substr(slash + 1);
    if ((f.area != "bin" && f.area != "runtime") || !valid_name(f.name)) throw Invalid("bad path: " + path);
    f.size = number_of(entry, "size", 1, kMaxTotalBytes);
    f.sha256 = text_of(entry, "sha256");
    if (!is_lower_hex(f.sha256, 64)) throw Invalid("sha256 of " + path + " is not 64 lowercase hex digits");
    const std::string& mode = text_of(entry, "mode");
    if (mode != "0755" && mode != "0644") throw Invalid("mode of " + path + " must be 0755 or 0644");
    f.mode = mode == "0755" ? 0755 : 0644;
    return f;
}

// A path is unique within a platform; a file of "any" takes the path on every platform. All platforms are checked,
// not only the client's own, so that the manifest means the same to every client.
void check_paths(const std::vector<ManifestFile>& files) {
    std::set<std::string> linux_paths, macos_paths, any_paths;
    for (const ManifestFile& f : files) {
        const std::string path = f.area + "/" + f.name;
        std::set<std::string>& own = f.platform == kPlatformLinux   ? linux_paths
                                     : f.platform == kPlatformMacos ? macos_paths
                                                                    : any_paths;
        if (!own.insert(path).second) throw Invalid("duplicate path " + path + " for " + f.platform);
    }
    for (const std::string& path : any_paths)
        if (linux_paths.contains(path) || macos_paths.contains(path))
            throw Invalid("path " + path + " is both for any platform and for a single one");
}

void check_files(const json& list, Manifest& m, const ManifestContext& ctx) {
    if (!list.is_array() || list.empty() || list.size() > kMaxFiles)
        throw Invalid("\"files\" must hold 1 to 256 entries");
    std::vector<ManifestFile> all;
    std::set<std::string> assets;
    uint64_t total = 0;
    for (const json& entry : list) {
        ManifestFile f = file_of(entry);
        if (!assets.insert(f.asset).second) throw Invalid("duplicate asset " + f.asset);
        total += f.size;
        if (total > kMaxTotalBytes) throw Invalid("total size exceeds 4 GiB");
        all.push_back(std::move(f));
    }
    check_paths(all);
    std::set<std::string> paths;
    for (ManifestFile& f : all) {
        if (f.platform != ctx.platform && f.platform != kPlatformAny) continue;
        paths.insert(f.area + "/" + f.name);
        m.files.push_back(std::move(f));
    }
    // The switch starts bin/tona-server; a release without it for this platform cannot be right
    if (!paths.contains("bin/tona-server")) throw Invalid("bin/tona-server is missing for " + ctx.platform);
}

void check_text_fields(const json& j, Manifest& m) {
    m.version = text_of(j, "version");
    if (!valid_version(m.version)) throw Invalid("version must be three dot-separated numbers");
    m.commit = text_of(j, "commit");
    if (!is_lower_hex(m.commit, 40)) throw Invalid("commit is not 40 lowercase hex digits");
    m.rocm_lock_sha256 = text_of(j, "rocm_lock_sha256");
    if (!is_lower_hex(m.rocm_lock_sha256, 64)) throw Invalid("rocm_lock_sha256 is not 64 lowercase hex digits");
    const std::string& severity = text_of(j, "severity");
    if (severity != "normal" && severity != "security") throw Invalid("severity must be normal or security");
    m.security = severity == "security";
    m.changelog = text_of(j, "changelog");
    if (m.changelog.size() > kMaxChangelogBytes) throw Invalid("changelog is too long");
    // The changelog is drawn in a terminal: escape sequences must not get through
    for (const char c : m.changelog) {
        const auto u = static_cast<unsigned char>(c);
        if ((u < 0x20 && c != '\n') || u == 0x7f) throw Invalid("changelog contains a control character");
    }
}

void check_header(const json& j) {
    if (number_of(j, "schema", 0, 1000) != 1) throw Invalid("unsupported schema");
    if (text_of(j, "product") != kProduct) throw Invalid("manifest is for another product");
}

void check_freshness(const Manifest& m, const ManifestContext& ctx) {
    if (m.seq < ctx.last_seq)
        throw Invalid(std::format("seq {} is older than the last accepted {}", m.seq, ctx.last_seq));
    if (m.expires <= ctx.now) throw Invalid("manifest has expired");
    if (m.released > m.expires) throw Invalid("manifest is released after it expires");
}

Manifest build(std::string_view text, std::string_view trusted_comment, const ManifestContext& ctx) {
    const json j = parse_no_duplicates(text);
    expect_keys(j,
                {"schema", "product", "version", "seq", "released", "expires", "commit", "rocm_lock_sha256", "severity",
                 "changelog", "files"},
                "manifest");
    check_header(j);
    Manifest m;
    check_text_fields(j, m);
    m.seq = number_of(j, "seq", 1, kMaxSeq);
    m.released = time_of(j, "released");
    m.expires = time_of(j, "expires");
    if (trusted_comment != expected_trusted_comment(m)) throw Invalid("trusted comment does not match the manifest");
    check_freshness(m, ctx);
    check_files(j.at("files"), m, ctx);
    return m;
}

}  // namespace

bool Manifest::has_runtime() const {
    return std::any_of(files.begin(), files.end(), [](const ManifestFile& f) { return f.area == "runtime"; });
}

uint64_t Manifest::total_bytes() const {
    uint64_t n = 0;
    for (const ManifestFile& f : files) n += f.size;
    return n;
}

std::string expected_trusted_comment(const Manifest& m) {
    return std::format("tona schema=1 seq={} version={}", m.seq, m.version);
}

namespace {

// Number made of exactly s[pos, pos+len) digits, or -1
int digits(std::string_view s, size_t pos, size_t len) {
    int v = 0;
    const char* first = s.data() + pos;
    const auto r = std::from_chars(first, first + len, v);
    return r.ec == std::errc() && r.ptr == first + len && std::isdigit(static_cast<unsigned char>(*first)) ? v : -1;
}

bool fixed_shape(std::string_view s) {
    return s.size() == 20 && s[4] == '-' && s[7] == '-' && s[10] == 'T' && s[13] == ':' && s[16] == ':' && s[19] == 'Z';
}

}  // namespace

bool parse_utc_time(std::string_view s, std::time_t& out) {
    // YYYY-MM-DDTHH:MM:SSZ: fixed positions, digits only
    if (!fixed_shape(s)) return false;
    const int fields[] = {digits(s, 0, 4),  digits(s, 5, 2),  digits(s, 8, 2),
                          digits(s, 11, 2), digits(s, 14, 2), digits(s, 17, 2)};
    if (std::any_of(std::begin(fields), std::end(fields), [](int v) { return v < 0; })) return false;
    std::tm tm{};
    tm.tm_year = fields[0] - 1900;
    tm.tm_mon = fields[1] - 1;
    tm.tm_mday = fields[2];
    tm.tm_hour = fields[3];
    tm.tm_min = fields[4];
    tm.tm_sec = fields[5];
    const std::time_t t = timegm(&tm);
    // Reformatting catches impossible dates (Feb 30, hour 24)
    std::tm back{};
    gmtime_r(&t, &back);
    char buf[24];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &back);
    if (s != buf) return false;
    out = t;
    return true;
}

ManifestResult parse_manifest(std::string_view text, std::string_view trusted_comment, const ManifestContext& ctx) {
    ManifestResult r;
    if (text.size() > kMaxManifestBytes) {
        r.error = "manifest is larger than 1 MiB";
        return r;
    }
    try {
        r.manifest = build(text, trusted_comment, ctx);
        r.ok = true;
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    return r;
}

}  // namespace tona::update
