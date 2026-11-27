#include "core/port.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>

#include "third_party/httplib.h"

namespace tona::install {

namespace {

std::function<PortState(int)>& probe_override() {
    static std::function<PortState(int)> f;
    return f;
}

constexpr int kSearch = 50;

}  // namespace

void set_port_probe(std::function<PortState(int)> probe) { probe_override() = std::move(probe); }

PortState probe_port(int port) {
    if (probe_override()) return probe_override()(port);
    // The URL form: the (host, port) constructor of httplib trips a GCC array-bounds false positive with -Werror
    httplib::Client c("http://127.0.0.1:" + std::to_string(port));
    c.set_connection_timeout(0, 300000);
    c.set_read_timeout(1, 0);
    if (auto r = c.Get("/status"); r && r->status == 200) return PortState::Ours;
#ifdef SOCK_CLOEXEC
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
#else  // macOS has no SOCK_CLOEXEC; the flag follows at once (an agent launched in between keeps the socket)
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd >= 0) ::fcntl(fd, F_SETFD, FD_CLOEXEC);
#endif
    if (fd < 0) return PortState::Free;  // cannot tell: do not claim a conflict
    sockaddr_in addr{};
    addr.sin_family = AF_INET, addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    const bool bound = ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0;
    ::close(fd);
    return bound ? PortState::Free : PortState::Busy;
}

int next_free_port(int from) {
    for (int p = from; p < from + kSearch && p <= 65535; p++)
        if (probe_port(p) != PortState::Busy) return p;
    return from;
}

}  // namespace tona::install
