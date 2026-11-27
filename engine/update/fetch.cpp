#include "update/fetch.h"

#include <charconv>
#include <memory>
#include <optional>

#include "third_party/httplib.h"
#include "update/crypto.h"
#include "update/url.h"

namespace tona::update {

namespace {

struct Hop {
    int status = 0;
    std::string location;
    std::string error;  // set by us when we stop the transfer on purpose
};

struct Transfer {
    const FetchConfig& cfg;
    const FetchRequest& req;
    const Sink& sink;
    const Progress& progress;
    Sha256 hash;
    uint64_t bytes = 0;
    uint64_t limit = 0;
};

bool scheme_allowed(const FetchConfig& cfg, const Url& u) {
    if (u.scheme == "https") return true;
    return cfg.allow_http_loopback && u.scheme == "http" && u.host == "127.0.0.1";
}

std::unique_ptr<httplib::Client> make_client(const FetchConfig& cfg, const FetchRequest& req, const Url& u) {
    auto cli = std::make_unique<httplib::Client>(u.origin());
    cli->set_connection_timeout(cfg.connect_seconds, 0);
    cli->set_read_timeout(cfg.read_seconds, 0);
    cli->set_write_timeout(cfg.read_seconds, 0);
    cli->set_max_timeout(std::chrono::seconds(req.total_seconds));
    cli->set_follow_location(false);
    cli->set_keep_alive(false);
    cli->enable_server_certificate_verification(true);
    cli->enable_server_hostname_verification(true);
    if (!cfg.ca_file.empty()) cli->set_ca_cert_path(cfg.ca_file);
    return cli;
}

// "bytes <start>-<end>/<total>" with the expected start
bool range_starts_at(const std::string& header, uint64_t start) {
    const std::string prefix = "bytes ";
    if (!header.starts_with(prefix)) return false;
    uint64_t n = 0;
    const char* first = header.data() + prefix.size();
    const auto r = std::from_chars(first, header.data() + header.size(), n);
    return r.ec == std::errc() && r.ptr < header.data() + header.size() && *r.ptr == '-' && n == start;
}

// Status line and headers: accept 200 with a plausible length, remember a redirect, refuse the rest
bool on_response(const httplib::Response& res, Transfer& t, Hop& hop) {
    hop.status = res.status;
    if (res.status >= 300 && res.status < 400) {
        hop.location = res.get_header_value("Location");
        return false;
    }
    const bool resumed = t.req.offset != 0;
    if (res.status != (resumed ? 206 : 200)) {
        hop.error = "server answered HTTP " + std::to_string(res.status);
        return false;
    }
    if (resumed && !range_starts_at(res.get_header_value("Content-Range"), t.req.offset)) {
        hop.error = "server did not resume at the requested position";
        return false;
    }
    if (res.has_header("Content-Length")) {
        const std::string len = res.get_header_value("Content-Length");
        uint64_t n = 0;
        const auto r = std::from_chars(len.data(), len.data() + len.size(), n);
        if (r.ec != std::errc() || r.ptr != len.data() + len.size() || n > t.limit ||
            (t.req.exact_bytes != 0 && n != t.req.exact_bytes)) {
            hop.error = "announced size does not match the expected size";
            return false;
        }
    }
    return true;
}

bool on_data(const char* data, size_t n, Transfer& t, Hop& hop) {
    if (t.cfg.cancel && t.cfg.cancel->load()) {
        hop.error = "cancelled";
        return false;
    }
    if (n > t.limit - t.bytes) {
        hop.error = "more data than expected";
        return false;
    }
    t.bytes += n;
    t.hash.update(data, n);
    if (!t.sink(data, n)) {
        hop.error = "could not store the data";
        return false;
    }
    if (t.progress) t.progress(t.bytes);
    return true;
}

// One request without following redirects. Returns an error text, empty if the hop ended with 200 or a redirect.
std::string one_hop(Transfer& t, const Url& u, Hop& hop) {
    const auto cli = make_client(t.cfg, t.req, u);
    httplib::Headers headers = {{"User-Agent", "tona-update"}, {"Accept-Encoding", "identity"}};
    if (t.req.offset != 0) headers.emplace("Range", "bytes=" + std::to_string(t.req.offset) + "-");
    const auto res = cli->Get(
        u.target, headers, [&](const httplib::Response& r) { return on_response(r, t, hop); },
        [&](const char* d, size_t n) { return on_data(d, n, t, hop); });
    if (!hop.error.empty()) return hop.error;
    if (hop.status >= 300 && hop.status < 400) return "";
    if (!res) return "request failed: " + httplib::to_string(res.error());
    return "";
}

// Absolute URL, or a path on the same host
std::optional<Url> next_url(const Url& from, const std::string& location) {
    if (location.starts_with('/') && !location.starts_with("//")) {
        return parse_url(from.origin() + location);
    }
    return parse_url(location);
}

}  // namespace

FetchResult fetch(const FetchConfig& cfg, const FetchRequest& req, const Sink& sink, const Progress& progress) {
    FetchResult out;
    Transfer t{cfg, req, sink, progress, Sha256(), 0, req.exact_bytes ? req.exact_bytes : req.max_bytes};
    auto url = parse_url(req.url);
    for (int hop_no = 0; url; hop_no++) {
        if (!scheme_allowed(cfg, *url)) {
            out.error = "refusing a non-https address";
            return out;
        }
        Hop hop;
        if (std::string err = one_hop(t, *url, hop); !err.empty()) {
            out.error = std::move(err);
            return out;
        }
        if (hop.status == (req.offset != 0 ? 206 : 200)) {
            if (req.exact_bytes != 0 && t.bytes != req.exact_bytes) {
                out.error = "download ended early";
                return out;
            }
            const auto digest = t.hash.finish();
            out.ok = true;
            out.bytes = t.bytes;
            out.sha256 = to_hex(digest.data(), digest.size());
            return out;
        }
        if (hop_no >= cfg.max_redirects) {
            out.error = "too many redirects";
            return out;
        }
        url = next_url(*url, hop.location);
    }
    out.error = "invalid address or redirect target";
    return out;
}

FetchResult fetch_text(const FetchConfig& cfg, const FetchRequest& req, std::string& body) {
    body.clear();
    FetchResult r = fetch(cfg, req, [&](const char* d, size_t n) {
        body.append(d, n);
        return true;
    });
    if (!r.ok) body.clear();
    return r;
}

}  // namespace tona::update
