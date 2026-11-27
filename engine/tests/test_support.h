// Shared by the unit tests: expectations, a temporary directory and a hand-written GGUF writer that is independent of
// the loader under test.
#pragma once

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "tests/test_guard.h"

namespace tona::test {

inline int g_fail = 0;

// A function, not an if inside the macro: every EXPECT would add to the cognitive complexity of its test
inline void check(bool ok, const char* file, int line, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "%s:%d: expectation violated: %s\n", file, line, what);
    g_fail++;
}

// Exit code of a test program: 1 if any expectation was violated
inline int finish(const char* name) {
    if (g_fail) {
        std::fprintf(stderr, "%s: %d expectations violated\n", name, g_fail);
        return 1;
    }
    std::printf("%s: ok\n", name);
    return 0;
}

// Runs the tests of a program; an exception that escapes a test fails the program instead of terminating it
template <class F>
int run(const char* name, F&& tests) noexcept {
    try {
        tests();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: exception: %s\n", name, e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "%s: unknown exception\n", name);
        return 1;
    }
    return finish(name);
}

inline bool throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// Directory below the system temp dir, removed again at the end of the test. Canonical: on macOS the temp dir is
// under /var, a link to /private/var, and a test that compares with what a child's `pwd -P` sees would differ
class TempDir {
public:
    explicit TempDir(const std::string& name)
        : path_(std::filesystem::canonical(std::filesystem::temp_directory_path()) /
                ("tona-test-" + name + "-" + std::to_string(getpid()))) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// GGUF v3 bytes. Value type numbers are those of the GGUF specification.
struct GgufWriter {
    enum Type : uint32_t { U16 = 2, I32 = 5, U32 = 4, F32 = 6, BOOL = 7, STR = 8, ARR = 9 };
    std::vector<uint8_t> b;

    template <class T>
    void put(T v) {
        auto* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
    }
    void str(const std::string& s) {
        put<uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void header(uint64_t n_tensors, uint64_t n_kv) {
        b.insert(b.end(), {'G', 'G', 'U', 'F'});
        put<uint32_t>(3);
        put<uint64_t>(n_tensors);
        put<uint64_t>(n_kv);
    }
    void kv_u32(const std::string& k, uint32_t v) {
        str(k);
        put<uint32_t>(U32);
        put<uint32_t>(v);
    }
    void kv_u16(const std::string& k, uint16_t v) {
        str(k);
        put<uint32_t>(U16);
        put<uint16_t>(v);
    }
    void kv_f32(const std::string& k, float v) {
        str(k);
        put<uint32_t>(F32);
        put<float>(v);
    }
    void kv_bool(const std::string& k, bool v) {
        str(k);
        put<uint32_t>(BOOL);
        put<uint8_t>(v);
    }
    void kv_str(const std::string& k, const std::string& v) {
        str(k);
        put<uint32_t>(STR);
        str(v);
    }
    void kv_arr_i32(const std::string& k, const std::vector<int32_t>& v) {
        str(k);
        put<uint32_t>(ARR);
        put<uint32_t>(I32);
        put<uint64_t>(v.size());
        for (auto x : v) put<int32_t>(x);
    }
    void kv_arr_str(const std::string& k, const std::vector<std::string>& v) {
        str(k);
        put<uint32_t>(ARR);
        put<uint32_t>(STR);
        put<uint64_t>(v.size());
        for (const auto& x : v) str(x);
    }
    void tensor(const std::string& name, const std::vector<uint64_t>& dims, uint32_t type, uint64_t offset) {
        str(name);
        put<uint32_t>(dims.size());
        for (auto d : dims) put<uint64_t>(d);
        put<uint32_t>(type);
        put<uint64_t>(offset);
    }
    void align(size_t a = 32) { b.resize((b.size() + a - 1) / a * a, 0); }
    void save(const std::filesystem::path& p) const {
        std::ofstream f(p, std::ios::binary);
        f.write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    }
};

}  // namespace tona::test

#define EXPECT(c) tona::test::check((c), __FILE__, __LINE__, #c)
