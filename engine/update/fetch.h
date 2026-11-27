// HTTPS download with hard limits: certificate and host name are verified, redirects are bounded and may only lead to
// https (plain http only to 127.0.0.1 in tests), the body is cut off at the size the signed manifest promised and is
// hashed while it arrives. Nothing is sent except the request itself.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace tona::update {

struct FetchConfig {
    bool allow_http_loopback = false;  // only from a validated TONA_UPDATE_BASE on 127.0.0.1
    std::string ca_file;  // tests only: replaces the system certificate store; never set from the environment
    int connect_seconds = 10;
    int read_seconds = 30;
    int max_redirects = 3;
    const std::atomic<bool>* cancel = nullptr;
};

struct FetchRequest {
    std::string url;
    uint64_t max_bytes = 0;    // hard limit; more than this is an error
    uint64_t exact_bytes = 0;  // if nonzero the body must have exactly this size (limit then is this size)
    int total_seconds = 60;    // whole transfer, all redirects included per hop
    // Resume: ask for the bytes from `offset` on ("Range: bytes=offset-"). The answer must be 206 with exactly that
    // start, otherwise it is an error (never a silent restart). `max_bytes`/`exact_bytes` then count the remaining
    // bytes; `FetchResult::sha256` covers only the bytes received in this call.
    uint64_t offset = 0;
};

struct FetchResult {
    bool ok = false;
    std::string error;
    uint64_t bytes = 0;
    std::string sha256;  // hex of the received bytes
};

// Receives each chunk; false aborts the transfer
using Sink = std::function<bool(const char*, size_t)>;
using Progress = std::function<void(uint64_t bytes_so_far)>;

FetchResult fetch(const FetchConfig& cfg, const FetchRequest& req, const Sink& sink, const Progress& progress = {});

// Whole body into memory, up to req.max_bytes
FetchResult fetch_text(const FetchConfig& cfg, const FetchRequest& req, std::string& body);

}  // namespace tona::update
