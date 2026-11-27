#include "backend/metal/device.h"

#include <mach-o/dyld.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "backend/env.h"

namespace tona::metal {

namespace {

std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("Metal: cannot read " + p.string());
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

// engine/model/quant_tables.h as MSL constants; the build does the same with tables.cmake
std::string tables_msl(const std::filesystem::path& header) {
    std::string t = slurp(header);
    t = std::regex_replace(
        t, std::regex(R"(#pragma once|#include <cstdint>|namespace tona::tables \{|\}  // namespace.*)"), "");
    for (const auto& [from, to] : {std::pair{"inline constexpr uint64_t", "constant ulong"},
                                   {"inline constexpr uint32_t", "constant uint"},
                                   {"inline constexpr uint8_t", "constant uchar"},
                                   {"inline constexpr int8_t", "constant char"}})
        t = std::regex_replace(t, std::regex(from), to);
    return t;
}

// all.metal with its local includes expanded in place (newLibraryWithSource has no include path)
std::string expand_sources(const std::filesystem::path& dir) {
    std::istringstream in(slurp(dir / "all.metal"));
    std::string out, line;
    const std::regex inc("#include \"([^\"]+)\"");
    std::smatch m;
    while (std::getline(in, line)) {
        if (!std::regex_match(line, m, inc)) {
            out += line + "\n";
            continue;
        }
        const std::string name = m[1];
        out += name == "quant_tables.metal" ? tables_msl(dir / "../../../model/quant_tables.h") : slurp(dir / name);
    }
    return out;
}

std::filesystem::path program_dir() {
    uint32_t n = 0;
    _NSGetExecutablePath(nullptr, &n);
    std::vector<char> p(n + 1);
    if (_NSGetExecutablePath(p.data(), &n) != 0) return ".";
    return std::filesystem::weakly_canonical(p.data()).parent_path();
}

// stringWithUTF8String returns nil for invalid UTF-8; the Metal calls need a string
NSString* ns(const char* s) {
    NSString* r = [NSString stringWithUTF8String:s];
    if (!r) throw std::runtime_error(std::string("Metal: not UTF-8: ") + s);
    return r;
}

}  // namespace

id<MTLLibrary> load_library(id<MTLDevice> dev) {
    NSError* err = nil;
    if (const char* dir = env::text("TONA_METAL_SOURCES")) {
        const std::string src = expand_sources(dir);
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;  // as the build (-fno-fast-math): see kernels/all.metal
        id<MTLLibrary> lib = [dev newLibraryWithSource:ns(src.c_str()) options:opt error:&err];
        if (!lib) throw std::runtime_error(std::string("Metal: kernels do not compile: ") + err.description.UTF8String);
        return lib;
    }
    const std::filesystem::path bin = program_dir();
    for (const auto& p : {bin / "tona.metallib", bin / "../lib/tona/tona.metallib"}) {
        if (!std::filesystem::exists(p)) continue;
        NSURL* url = [NSURL fileURLWithPath:ns(p.c_str())];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (!lib) throw std::runtime_error("Metal: cannot load " + p.string() + ": " + err.description.UTF8String);
        return lib;
    }
    throw std::runtime_error("Metal: tona.metallib not found next to the program or in ../lib/tona "
                             "(developers: TONA_METAL_SOURCES=engine/backend/metal/kernels)");
}

namespace {
// 500 ms: the wait after a pause starts between 0.5 and 1 s without a pass; prefill@8000 +2.05 %, an agent's tool
// step 1.04 instead of 1.35 to 1.43 s, no measurable idle cost (research/measurements/2026-11-15-mac-cache-hit.md)
constexpr int64_t kKeepaliveMs = 500;
}  // namespace

Device::Device() {
    dev_ = MTLCreateSystemDefaultDevice();
    if (!dev_) throw std::runtime_error("Metal: no device");
    queue_ = [dev_ newCommandQueue];
    lib_ = load_library(dev_);
    if (env::flag("TONA_METAL_PIN_OWN")) {
        if (@available(macOS 15.0, *)) {
            NSError* err = nil;
            id<MTLResidencySet> s = [dev_ newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&err];
            if (!s) throw std::runtime_error("Metal: no residency set");
            [queue_ addResidencySet:s];
            own_ = s;
        }
    }
    if (const int64_t ms = env::integer("TONA_METAL_KEEPALIVE_MS", kKeepaliveMs); ms > 0)
        keep_ = std::make_unique<Keepalive>(dev_, (double)ms, keepalive_window_s());
}

id<MTLComputePipelineState> Device::pipe(const std::string& name) {
    auto it = pipes_.find(name);
    if (it != pipes_.end()) return it->second;
    id<MTLFunction> f = [lib_ newFunctionWithName:ns(name.c_str())];
    if (!f) throw std::runtime_error("Metal: no kernel " + name);
    NSError* err = nil;
    id<MTLComputePipelineState> p = [dev_ newComputePipelineStateWithFunction:f error:&err];
    if (!p) throw std::runtime_error("Metal: pipeline " + name + ": " + err.description.UTF8String);
    return pipes_[name] = p;
}

Buf Device::floats(uint64_t n) {
    const uint64_t bytes = std::max<uint64_t>(16, n * 4);
    id<MTLBuffer> b = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("Metal: cannot allocate " + std::to_string(bytes) + " bytes");
    memset(b.contents, 0, bytes);
    allocated_ += bytes;
    if (own_) {
        if (@available(macOS 15.0, *)) {
            id<MTLResidencySet> s = own_;
            [s addAllocation:b];
            [s commit];
            [s requestResidency];
        }
    }
    return {b, 0};
}

void Dispatch::run_raw(const std::string& kernel, std::initializer_list<Buf> bufs, const void* params, size_t bytes,
                       MTLSize grid, MTLSize tg) {
    [enc setComputePipelineState:dev.pipe(kernel)];
    NSUInteger i = 0;
    for (const Buf& b : bufs) [enc setBuffer:b.b offset:b.off atIndex:i++];
    [enc setBytes:params length:bytes atIndex:i];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
}

}  // namespace tona::metal
