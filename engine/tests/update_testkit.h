// Shared by the updater tests: a test signing key (the same seed as data/update/generate.py), a builder for signed
// fake releases, and a local HTTP server on 127.0.0.1 that serves them. Test keys live only here, never in a release.
#pragma once

#include <openssl/evp.h>

#include <atomic>
#include <charconv>
#include <chrono>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "tests/test_support.h"
#include "third_party/httplib.h"
#include "third_party/nlohmann/json.hpp"
#include "update/crypto.h"
#include "update/release.h"

namespace tona::test {

using nlohmann::json;
namespace up = tona::update;

inline std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline std::string base64_encode(const std::string& in) {
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < in.size(); i += 3) {
        const size_t n = std::min<size_t>(3, in.size() - i);
        uint32_t v = 0;
        for (size_t k = 0; k < 3; k++) v = (v << 8) | (k < n ? static_cast<uint8_t>(in[i + k]) : 0);
        for (size_t k = 0; k < 4; k++) out += k <= n ? kAlphabet[(v >> (18 - 6 * k)) & 63] : '=';
    }
    return out;
}

// Ed25519 key from a seed, with a minisign key id
struct TestKey {
    std::array<uint8_t, 32> seed{};
    std::array<uint8_t, 8> id{};

    static TestKey make(uint8_t first_seed_byte, const std::array<uint8_t, 8>& id) {
        TestKey k;
        for (size_t i = 0; i < 32; i++) k.seed[i] = static_cast<uint8_t>(first_seed_byte + i);
        k.id = id;
        return k;
    }

    std::string sign(const std::string& message) const {
        EVP_PKEY* key = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr, seed.data(), seed.size());
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        uint8_t sig[64];
        size_t len = sizeof sig;
        EVP_DigestSignInit(ctx, nullptr, nullptr, nullptr, key);
        EVP_DigestSign(ctx, sig, &len, reinterpret_cast<const uint8_t*>(message.data()), message.size());
        EVP_MD_CTX_free(ctx);
        EVP_PKEY_free(key);
        return std::string(reinterpret_cast<char*>(sig), len);
    }

    up::TrustedKey trusted() const {
        EVP_PKEY* key = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr, seed.data(), seed.size());
        up::TrustedKey t;
        size_t len = t.pub.size();
        EVP_PKEY_get_raw_public_key(key, t.pub.data(), &len);
        EVP_PKEY_free(key);
        t.id = id;
        return t;
    }

    // A minisign .minisig file for `data`
    std::string sign_file(const std::string& data, const std::string& comment) const {
        const auto digest = up::blake2b512(data);
        const std::string sig = sign(std::string(reinterpret_cast<const char*>(digest.data()), digest.size()));
        const std::string head = "ED" + std::string(reinterpret_cast<const char*>(id.data()), id.size()) + sig;
        return "untrusted comment: signature from minisign secret key\n" + base64_encode(head) +
               "\ntrusted comment: " + comment + "\n" + base64_encode(sign(sig + comment)) + "\n";
    }
};

inline TestKey main_key() { return TestKey::make(0, {1, 2, 3, 4, 5, 6, 7, 8}); }
inline TestKey other_key() { return TestKey::make(100, {0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8}); }

inline std::time_t utc(const char* text) {
    std::time_t t = 0;
    if (!up::parse_utc_time(text, t)) std::abort();
    return t;
}

inline std::string comment_for(uint64_t seq, const std::string& version) {
    return "tona schema=1 seq=" + std::to_string(seq) + " version=" + version;
}

// A signed release: manifest, signature and the asset bytes it names
struct FakeRelease {
    std::string version;
    uint64_t seq = 0;
    json manifest;
    std::string manifest_text;
    std::string signature;
    std::map<std::string, std::string> assets;  // asset name -> content

    void sign(const TestKey& key) {
        manifest_text = manifest.dump(2);
        signature = key.sign_file(manifest_text, comment_for(seq, version));
    }
};

inline FakeRelease make_release(const std::string& version, uint64_t seq,
                                const std::string& lock = std::string(64, '8')) {
    FakeRelease r;
    r.version = version;
    r.seq = seq;
    const std::string lock12 = lock.substr(0, 12);
    // In this order: the manifest tests edit the first file (the server) by position
    const std::vector<std::pair<std::string, std::pair<std::string, std::string>>> parts = {
        {"tona-server", {"bin/tona-server", "0755"}},
        {"tona", {"bin/tona", "0755"}},
        {"rocm-" + lock12 + "-libhip.so.7", {"runtime/libhip.so.7", "0644"}}};
    json files = json::array();
    for (const auto& [asset, info] : parts) {
        // The runtime depends on the ROCm lock only, so releases with the same lock share identical runtime files
        const bool runtime = info.first.starts_with("runtime/");
        std::string content = asset;
        content += runtime ? " of " + lock : " of " + version;
        content.append(100, 'x');
        r.assets[asset] = content;
        up::Sha256 h;
        h.update(content.data(), content.size());
        const auto d = h.finish();
        files.push_back({{"asset", asset},
                         {"path", info.first},
                         {"platform", std::string(up::kPlatform)},
                         {"size", content.size()},
                         {"sha256", up::to_hex(d.data(), d.size())},
                         {"mode", info.second}});
    }
    // A file of the other platform: a client must neither download nor count it
    files.push_back({{"asset", "other-platform-tona-server"},
                     {"path", "bin/tona-server"},
                     {"platform", up::kPlatform == up::kPlatformLinux ? "macos-arm64" : "linux-x86_64-gfx1200"},
                     {"size", 100},
                     {"sha256", std::string(64, 'e')},
                     {"mode", "0755"}});
    r.manifest = {{"schema", 1},
                  {"product", "tona"},
                  {"version", version},
                  {"seq", seq},
                  {"released", "2026-11-02T10:00:00Z"},
                  {"expires", "2027-05-02T00:00:00Z"},
                  {"commit", std::string(40, 'c')},
                  {"rocm_lock_sha256", lock},
                  {"severity", "normal"},
                  {"changelog", "Test release " + version},
                  {"files", files}};
    r.sign(main_key());
    return r;
}

// A release of a platform without a shared runtime (the Mac): only programs in bin/
inline FakeRelease make_release_without_runtime(const std::string& version, uint64_t seq) {
    FakeRelease r = make_release(version, seq);
    json kept = json::array();
    for (const json& f : r.manifest["files"]) {
        if (f["path"].get<std::string>().starts_with("runtime/")) {
            r.assets.erase(f["asset"].get<std::string>());
        } else {
            kept.push_back(f);
        }
    }
    r.manifest["files"] = kept;
    r.sign(main_key());
    return r;
}

// One path of the test server
struct Route {
    Route() = default;
    explicit Route(std::string content) : body(std::move(content)) {}
    std::string body;
    int status = 200;
    std::string location;         // with 3xx
    size_t cut_after = SIZE_MAX;  // send this many bytes, then drop the connection
    bool chunked = false;         // no Content-Length
    int delay_ms = 0;             // before answering
};

// httplib server on 127.0.0.1 and a free port, with counters per path
class TestServer {
public:
    // With a certificate and key (PEM files) the server speaks TLS
    explicit TestServer(const std::string& cert = "", const std::string& key = "") : tls_(!cert.empty()) {
        if (tls_) {
            svr_ = std::make_unique<httplib::SSLServer>(cert.c_str(), key.c_str());
        } else {
            svr_ = std::make_unique<httplib::Server>();
        }
        svr_->Get(R"(.*)", [this](const httplib::Request& req, httplib::Response& res) { handle(req, res); });
        port_ = svr_->bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { svr_->listen_after_bind(); });
        svr_->wait_until_ready();
    }
    ~TestServer() {
        svr_->stop();
        thread_.join();
    }
    TestServer(const TestServer&) = delete;
    TestServer& operator=(const TestServer&) = delete;

    int port() const { return port_; }
    std::string base() const {
        return std::string(tls_ ? "https" : "http") + "://127.0.0.1:" + std::to_string(port_) + "/releases";
    }
    void set(const std::string& path, Route r) {
        std::lock_guard lk(mu_);
        routes_[path] = std::move(r);
    }
    int hits(const std::string& path) {
        std::lock_guard lk(mu_);
        return hits_[path];
    }
    // Headers of the most recent request
    std::map<std::string, std::string> last_headers() {
        std::lock_guard lk(mu_);
        return header_names_;
    }
    void serve_latest(const FakeRelease& r) {
        set("/releases/latest/download/manifest.json", Route(r.manifest_text));
        set("/releases/latest/download/manifest.json.minisig", Route(r.signature));
        for (const auto& [asset, content] : r.assets)
            set("/releases/download/v" + r.version + "/" + asset, Route(content));
    }

private:
    void handle(const httplib::Request& req, httplib::Response& res) {
        Route r;
        {
            std::lock_guard lk(mu_);
            hits_[req.path]++;
            header_names_.clear();
            for (const auto& [name, value] : req.headers) header_names_[name] = value;
            const auto it = routes_.find(req.path);
            if (it == routes_.end()) {
                res.status = 404;
                return;
            }
            r = it->second;
        }
        if (r.delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(r.delay_ms));
        res.status = r.status;
        if (!r.location.empty()) res.set_header("Location", r.location);
        if (r.chunked) {
            res.set_chunked_content_provider("application/octet-stream", [r](size_t off, httplib::DataSink& sink) {
                if (off < r.body.size()) sink.write(r.body.data() + off, r.body.size() - off);
                sink.done();
                return true;
            });
        } else if (r.cut_after < r.body.size()) {
            res.set_content_provider(r.body.size(), "application/octet-stream",
                                     [r](size_t off, size_t, httplib::DataSink& sink) {
                                         if (off >= r.cut_after) return false;
                                         sink.write(r.body.data() + off, r.cut_after - off);
                                         return true;
                                     });
        } else if (size_t from = 0; r.status == 200 && range_start(req, from) && from < r.body.size()) {
            // A download that resumes sends "Range: bytes=N-"; httplib cuts the body and writes Content-Range when
            // the handler answers 206
            res.status = 206;
            res.set_content(r.body, "application/octet-stream");
        } else {
            res.set_content(r.body, "application/octet-stream");
        }
    }

    static bool range_start(const httplib::Request& req, size_t& from) {
        const std::string h = req.get_header_value("Range");
        if (!h.starts_with("bytes=") || !h.ends_with('-')) return false;
        const std::string n = h.substr(6, h.size() - 7);
        return !n.empty() && std::from_chars(n.data(), n.data() + n.size(), from).ec == std::errc();
    }

    bool tls_;
    std::unique_ptr<httplib::Server> svr_;
    std::thread thread_;
    int port_ = 0;
    std::mutex mu_;
    std::map<std::string, Route> routes_;
    std::map<std::string, int> hits_;
    std::map<std::string, std::string> header_names_;
};

// Environment of an update run against a test server; the install root lives in `dir`
inline up::UpdateEnv make_env(const TestServer& server, const std::filesystem::path& dir) {
    up::UpdateEnv env;
    env.layout.root = dir / "tona";
    const auto base = up::parse_base(server.base());
    if (!base) throw std::runtime_error("test server base URL does not parse");
    env.base = *base;
    env.fetch.allow_http_loopback = true;
    env.fetch.connect_seconds = 3;
    env.fetch.read_seconds = 5;
    env.keys = {main_key().trusted()};
    env.now = utc("2026-11-05T00:00:00Z");
    return env;
}

}  // namespace tona::test
