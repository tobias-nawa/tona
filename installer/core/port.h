// Is a TCP port of this machine free for the server? The same check `tona doctor` makes: ask /status first (a Tona
// server of ours counts as free: a new setup over an existing installation keeps its port), then try to bind.
#pragma once

#include <functional>

namespace tona::install {

enum class PortState { Free, Ours, Busy };

PortState probe_port(int port);

// The first free port from `from` on (a port of our own server counts as free); `from` itself if nothing up to 50
// further is free
int next_free_port(int from);

// For tests: replaces the probe (an empty function restores the real one)
void set_port_probe(std::function<PortState(int)> probe);

}  // namespace tona::install
