// Checks of the browser-controlled request headers for a local server: Host (DNS rebinding) and Origin (cross-site
// requests). A page the user visits can send requests to 127.0.0.1; with a rebound DNS name it can also read the
// answers, the class of Ollama CVE-2024-28224 (NCC Group advisory, 2024-04-08). Non-browser clients send no Origin.
#pragma once

#include <string>
#include <vector>

namespace tona::server {

// The server is bound to a loopback address (127.0.0.1, localhost, ::1)
bool is_loopback_bind(const std::string& bind_host);

// The Host header names this server on loopback: 127.0.0.1, localhost or [::1], with this port (no port only for 80)
bool host_allowed(const std::string& host_header, int port);

// No Origin, a loopback origin with this port, or one of extra (exact, case-insensitive): TONA_ALLOWED_ORIGINS
bool origin_allowed(const std::string& origin, int port, const std::vector<std::string>& extra);

// Comma-separated origins, blanks trimmed; null or empty gives none
std::vector<std::string> parse_origin_list(const char* list);

}  // namespace tona::server
