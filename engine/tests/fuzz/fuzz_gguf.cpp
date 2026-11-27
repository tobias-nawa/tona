// libFuzzer: bytes of a GGUF file -> Gguf loader (header, metadata, tensor directory).
// The loader maps a file, so every input is written to one small file first. A std::runtime_error is the loader
// rejecting a bad file and is allowed; anything else (crash, ASan, UBSan, other exceptions) is a finding.
// Build and run: scripts/fuzz.sh
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "model/gguf.h"

namespace {

// A malformed value may throw runtime_error (the expected rejection); anything else is a finding
template <class F>
bool tolerate(F&& f) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

// One file per process, rewritten for every input; mkstemp keeps parallel fuzzer jobs apart.
const std::string& input_path() {
    static const std::string path = [] {
        char tmpl[] = "/tmp/tona-fuzz-gguf-XXXXXX";
        const int fd = mkstemp(tmpl);
        if (fd < 0) std::abort();
        close(fd);
        std::atexit([] { unlink(input_path().c_str()); });
        return std::string(tmpl);
    }();
    return path;
}

void write_input(const uint8_t* data, size_t size) {
    FILE* f = std::fopen(input_path().c_str(), "wb");
    if (!f) std::abort();
    if (size > 0 && std::fwrite(data, 1, size, f) != size) std::abort();
    std::fclose(f);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    write_input(data, size);
    try {
        const tona::Gguf g(input_path());
        for (const auto& t : g.tensors()) {
            (void)t.elements();
            (void)g.file_offset(t);
            (void)g.find(t.name);
        }
        for (const auto& [key, value] : g.kv()) {
            (void)tolerate([&] { (void)g.get_uint(key); });
            (void)tolerate([&] { (void)g.get_float(key); });
            (void)tolerate([&] { (void)g.get_string(key); });
            (void)tolerate([&] { (void)g.get_int_array(key); });
        }
    } catch (const std::runtime_error&) {
        return 0;  // allowed: the loader rejects the file
    }
    return 0;
}
