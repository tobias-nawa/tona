// Prints the tensor directory of a GGUF file as JSON lines (or the metadata with --kv). Cross-check for
// tools/gguf_info.py.
// Usage: tona-gguf FILE.gguf [--kv]
#include <cstdio>
#include <cstring>
#include <exception>
#include <variant>

#include "model/gguf.h"

namespace {

// Scalar of a metadata value; false for strings and arrays
template <class Variant>
bool print_scalar(const Variant& v) {
    if (auto* u = std::get_if<uint64_t>(&v))
        std::printf("%llu", (unsigned long long)*u);
    else if (auto* i = std::get_if<int64_t>(&v))
        std::printf("%lld", (long long)*i);
    else if (auto* d = std::get_if<double>(&v))
        std::printf("%g", *d);
    else if (auto* b = std::get_if<bool>(&v))
        std::printf("%s", *b ? "true" : "false");
    else
        return false;
    return true;
}

// Short arrays of numbers in full, everything else (and the tokenizer tables) as a count
void print_array(const std::string& key, const tona::GgufArray& a) {
    if (a.size() > 16 || key.starts_with("tokenizer.")) {
        std::printf(" = [%zu entries]", a.size());
        return;
    }
    std::printf(" = [");
    for (size_t i = 0; i < a.size(); i++) {
        if (i) std::printf(", ");
        if (!print_scalar(a[i].v)) std::printf("?");
    }
    std::printf("]");
}

void print_kv(const tona::Gguf& g) {
    for (const auto& [k, v] : g.kv()) {
        std::printf("%s", k.c_str());
        if (auto* s = std::get_if<std::string>(&v.v)) {
            std::printf(" = \"%.60s\"", s->c_str());
        } else if (auto* a = std::get_if<tona::GgufArray>(&v.v)) {
            print_array(k, *a);
        } else {
            std::printf(" = ");
            print_scalar(v.v);
        }
        std::printf("\n");
    }
}

void print_tensors(const tona::Gguf& g) {
    for (const auto& t : g.tensors()) {
        std::printf(R"({"name":"%s","type":%u,"dims":[)", t.name.c_str(), t.type);
        for (size_t i = 0; i < t.dims.size(); i++) std::printf(i ? ",%llu" : "%llu", (unsigned long long)t.dims[i]);
        std::printf("],\"offset\":%llu,\"bytes\":%llu,\"file\":%d}\n", (unsigned long long)t.offset,
                    (unsigned long long)t.bytes, t.file);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: tona-gguf FILE.gguf [--kv]\n");
        return 2;
    }
    try {
        tona::Gguf g(argv[1]);
        if (argc > 2 && std::strcmp(argv[2], "--kv") == 0)
            print_kv(g);
        else
            print_tensors(g);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
