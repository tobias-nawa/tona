// main() of a test program without an uncaught exception: bugprone-exception-escape checks main. Kept apart from
// test_support.h for the tests that bring their own EXPECT.
#pragma once

#include <cstdio>
#include <exception>

namespace tona::test {

// main() of a test whose body returns the exit code itself: an exception that escapes fails the program
template <class F>
int guard(const char* name, F&& body) noexcept {
    try {
        return body();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: exception: %s\n", name, e.what());
    } catch (...) {
        std::fprintf(stderr, "%s: unknown exception\n", name);
    }
    return 1;
}

}  // namespace tona::test
