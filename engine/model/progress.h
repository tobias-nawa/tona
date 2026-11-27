// Progress of startup and running work for /status. Thread-safe; the engine reports, the server reads.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tona::progress {

struct Phase {
    std::string name;
    double done = 0, total = 0;  // units of the phase (bytes, layers, ...); total 0: unknown
    double t0 = 0, t1 = -1;      // seconds since program start; t1 < 0: running
};

// new phase (ends the running one)
void begin(const std::string& name, double total = 0);
void add(double n);
void set_total(double total);
// startup done: end the running phase
void finish();
std::vector<Phase> phases();
bool ready();
double now();  // seconds since program start

// running prefill: tokens done and total (0/0: none)
void prefill(int64_t done, int64_t total);
void prefill_done(int64_t done);
void prefill_state(int64_t& done, int64_t& total);

}  // namespace tona::progress
