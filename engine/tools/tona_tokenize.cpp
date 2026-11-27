// Tokenizes a file (or stdin) and prints the ids, one per line. Cross-check for llama-tokenize.
// Usage: tona-tokenize MODEL.gguf [FILE] [--no-special] [--decode]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include "model/gguf.h"
#include "model/tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: tona-tokenize MODEL.gguf [FILE] [--no-special] [--decode]\n");
        return 2;
    }
    bool special = true, decode = false;
    const char* file = nullptr;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--no-special"))
            special = false;
        else if (!std::strcmp(argv[i], "--decode"))
            decode = true;
        else
            file = argv[i];
    }
    try {
        tona::Gguf g(argv[1]);
        tona::Tokenizer tok(g);
        std::string text;
        if (file) {
            std::ifstream f(file, std::ios::binary);
            if (!f) throw std::runtime_error(std::string(file) + ": cannot read");
            text.assign(std::istreambuf_iterator<char>(f), {});
        } else {
            text.assign(std::istreambuf_iterator<char>(std::cin), {});
        }
        const auto t0 = std::chrono::steady_clock::now();
        const auto ids = tok.encode(text, special);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (decode) {
            fwrite(tok.decode(ids).data(), 1, tok.decode(ids).size(), stdout);
        } else {
            for (int32_t id : ids) printf("%d\n", id);
        }
        fprintf(stderr, "%zu bytes -> %zu tokens in %.3f s (%.1f MB/s)\n", text.size(), ids.size(), s,
                text.size() / s / 1e6);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
