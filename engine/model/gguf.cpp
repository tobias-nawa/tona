#include "model/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <regex>
#include <stdexcept>

#include "model/quant.h"

namespace tona {

namespace {

// Values from ggml/src/ggml.c (type_traits), b11822
struct TypeEntry {
    uint32_t type;
    GgmlTypeInfo info;
};
constexpr TypeEntry kTypes[] = {
    {0, {"F32", 1, 4}},
    {1, {"F16", 1, 2}},
    {2, {"Q4_0", 32, 18}},
    {3, {"Q4_1", 32, 20}},
    {6, {"Q5_0", 32, 22}},
    {7, {"Q5_1", 32, 24}},
    {8, {"Q8_0", 32, 34}},
    {9, {"Q8_1", 32, 36}},
    {10, {"Q2_K", 256, 84}},
    {11, {"Q3_K", 256, 110}},
    {12, {"Q4_K", 256, 144}},
    {13, {"Q5_K", 256, 176}},
    {14, {"Q6_K", 256, 210}},
    {15, {"Q8_K", 256, 292}},
    {16, {"IQ2_XXS", 256, 66}},
    {17, {"IQ2_XS", 256, 74}},
    {18, {"IQ3_XXS", 256, 98}},
    {19, {"IQ1_S", 256, 50}},
    {20, {"IQ4_NL", 32, 18}},
    {21, {"IQ3_S", 256, 110}},
    {22, {"IQ2_S", 256, 82}},
    {23, {"IQ4_XS", 256, 136}},
    {24, {"I8", 1, 1}},
    {25, {"I16", 1, 2}},
    {26, {"I32", 1, 4}},
    {27, {"I64", 1, 8}},
    {28, {"F64", 1, 8}},
    {29, {"IQ1_M", 256, 56}},
    {30, {"BF16", 1, 2}},
    {34, {"TQ1_0", 256, 54}},
    {35, {"TQ2_0", 256, 66}},
    {39, {"MXFP4", 32, 17}},
    {40, {"NVFP4", 64, 36}},
    {41, {"Q1_0", 128, 18}},
    {42, {"Q2_0", 64, 18}},
    // Tona's own: IQ3_XXS repacked in 512-weight super-blocks with aligned fields (docs/use/tona-types.md)
    {1018, {"TONA_IQ3_XXS_R", 512, 196}},
    // Tona's own: Q6_K with 8 rows interleaved in MMA fragment order; rows are not contiguous (docs/use/tona-types.md)
    {1019, {"TONA_Q6_K_T", 256, 210}},
};

enum : uint32_t { U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 };

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    const std::string& path;

    void need(uint64_t n) const {
        if ((uint64_t)(end - p) < n)
            throw std::runtime_error(std::format("{}: file ends in the middle of the header", path));
    }
    template <class T>
    T get() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        need(n);
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
};

GgufValue read_value(Cursor& c, uint32_t t, int depth = 0);

GgufValue read_array(Cursor& c, int depth) {
    if (depth > 4) throw std::runtime_error(std::format("{}: arrays nested too deeply", c.path));
    uint32_t et = c.get<uint32_t>();
    uint64_t n = c.get<uint64_t>();
    if (n > (uint64_t)(c.end - c.p)) throw std::runtime_error(std::format("{}: array too long", c.path));
    GgufArray a;
    a.reserve(n);
    for (uint64_t i = 0; i < n; i++) a.push_back(read_value(c, et, depth + 1));
    return {std::move(a)};
}

GgufValue read_value(Cursor& c, uint32_t t, int depth) {
    switch (t) {
        case U8:
            return {(uint64_t)c.get<uint8_t>()};
        case I8:
            return {(int64_t)c.get<int8_t>()};
        case U16:
            return {(uint64_t)c.get<uint16_t>()};
        case I16:
            return {(int64_t)c.get<int16_t>()};
        case U32:
            return {(uint64_t)c.get<uint32_t>()};
        case I32:
            return {(int64_t)c.get<int32_t>()};
        case F32:
            return {(double)c.get<float>()};
        case BOOL:
            return {c.get<uint8_t>() != 0};
        case STR:
            return {c.str()};
        case U64:
            return {c.get<uint64_t>()};
        case I64:
            return {c.get<int64_t>()};
        case F64:
            return {c.get<double>()};
        case ARR:
            return read_array(c, depth);
        default:
            break;
    }
    throw std::runtime_error(std::format("{}: unknown GGUF value type {}", c.path, t));
}

uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

}  // namespace

const GgmlTypeInfo* ggml_type_info(uint32_t type) {
    for (const TypeEntry& e : kTypes)
        if (e.type == type) return &e.info;
    return nullptr;
}

uint64_t TensorInfo::elements() const {
    uint64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
}

MappedFile::MappedFile(const std::string& path) : path_(path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error(std::format("{}: {}", path, std::strerror(errno)));
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error(std::format("{}: {}", path, std::strerror(errno)));
    }
    size_ = (uint64_t)st.st_size;
    void* p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error(std::format("{}: mmap: {}", path, std::strerror(errno)));
    }
    fd_ = fd;  // stays open for read()
    data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
    if (data_) munmap(const_cast<uint8_t*>(data_), size_);
    if (fd_ >= 0) ::close(fd_);
}

void MappedFile::read(const uint8_t* ptr, void* dst, size_t n) const {
    if (!contains(ptr) || ptr + n > data_ + size_) throw std::runtime_error(path_ + ": read outside the file");
    off_t off = ptr - data_;
    auto* out = static_cast<uint8_t*>(dst);
    while (n > 0) {
        const ssize_t r = ::pread(fd_, out, n, off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error(std::format("{}: pread: {}", path_, std::strerror(errno)));
        out += r, off += r, n -= (size_t)r;
    }
}

Gguf::Gguf(const std::string& path) {
    files_.push_back(std::make_unique<MappedFile>(path));
    parse(0, true);
    if (has("split.no") && get_uint("split.no") != 0)
        throw std::runtime_error(
            std::format("{}: part {} of a split file, open part 1", path, get_uint("split.no") + 1));
    if (!has("split.count")) return;
    uint64_t count = get_uint("split.count");
    // Naming scheme of llama.cpp: <prefix>-00001-of-00002.gguf
    std::smatch m;
    static const std::regex re(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)");
    if (!std::regex_match(path, m, re) || std::stoull(m[2]) != 1 || std::stoull(m[3]) != count)
        throw std::runtime_error(std::format("{}: split file, but not part 1 of {}", path, count));
    for (uint64_t i = 2; i <= count; i++) {
        files_.push_back(std::make_unique<MappedFile>(std::format("{}-{:05}-of-{:05}.gguf", m[1].str(), i, count)));
        parse((int)files_.size() - 1, false);
    }
    if (has("split.tensors.count") && tensors_.size() != get_uint("split.tensors.count"))
        throw std::runtime_error(std::format("{}: found {} tensors, split.tensors.count says {}", path, tensors_.size(),
                                             get_uint("split.tensors.count")));
}

namespace {

std::map<std::string, GgufValue> read_kv(Cursor& c, uint64_t n_kv) {
    std::map<std::string, GgufValue> kv;
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = c.str();
        uint32_t t = c.get<uint32_t>();
        kv[key] = read_value(c, t);
    }
    return kv;
}

uint64_t read_alignment(const std::map<std::string, GgufValue>& kv, const std::string& path) {
    uint64_t alignment = 32;
    if (auto it = kv.find("general.alignment"); it != kv.end()) {
        // The value type comes from the file: only an integer is an alignment
        if (auto* u = std::get_if<uint64_t>(&it->second.v)) {
            alignment = *u;
        } else if (auto* i = std::get_if<int64_t>(&it->second.v); i && *i >= 0) {
            alignment = (uint64_t)*i;
        } else {
            throw std::runtime_error(std::format("{}: general.alignment is not a non-negative integer", path));
        }
    }
    if (alignment == 0 || (alignment & (alignment - 1)))
        throw std::runtime_error(std::format("{}: general.alignment {} is not a power of two", path, alignment));
    return alignment;
}

// A tile type interleaves 8 rows, so a tensor read by row (token_embd, output, per_layer_token_embd, the MTP head's
// tables) or by expert must never have one: only 2-D matrices of a layer (docs/use/tona-types.md)
void check_tile_type(const TensorInfo& t, const MappedFile& f) {
    if (row_contiguous(t.type)) return;
    if (!t.name.starts_with("blk.") || t.dims.size() != 2 || t.dims[0] % 256 || t.dims[1] % 8)
        throw std::runtime_error(
            std::format("{}: tensor {}: type {} only for 2-D layer matrices with rows a multiple of "
                        "8 and 256 weights (docs/use/tona-types.md)",
                        f.path(), t.name, ggml_type_info(t.type)->name));
}

TensorInfo read_tensor_info(Cursor& c, const MappedFile& f, int file_index) {
    TensorInfo t;
    t.name = c.str();
    uint32_t nd = c.get<uint32_t>();
    if (nd > 4) throw std::runtime_error(std::format("{}: tensor {} has {} dimensions", f.path(), t.name, nd));
    for (uint32_t d = 0; d < nd; d++) t.dims.push_back(c.get<uint64_t>());
    t.type = c.get<uint32_t>();
    t.offset = c.get<uint64_t>();
    t.file = file_index;
    const GgmlTypeInfo* ti = ggml_type_info(t.type);
    if (!ti) throw std::runtime_error(std::format("{}: tensor {}: unknown ggml type {}", f.path(), t.name, t.type));
    if (t.dims.empty() || t.dims[0] % ti->block_elems != 0)
        throw std::runtime_error(std::format("{}: tensor {}: row does not fit the block size", f.path(), t.name));
    uint64_t elems = 1;  // overflow-free, otherwise the bounds check in place_tensor would pass
    for (uint64_t dm : t.dims) {
        if (dm == 0 || elems > f.size() * 64 / dm)
            throw std::runtime_error(std::format("{}: tensor {}: shape too large", f.path(), t.name));
        elems *= dm;
    }
    t.bytes = t.elements() / ti->block_elems * ti->block_bytes;
    check_tile_type(t, f);
    return t;
}

}  // namespace

void Gguf::parse(int fi, bool first) {
    const MappedFile& f = *files_[fi];
    Cursor c{f.data(), f.data() + f.size(), f.path()};
    c.need(4);
    if (std::memcmp(c.p, "GGUF", 4) != 0) throw std::runtime_error(std::format("{}: not a GGUF file", f.path()));
    c.p += 4;
    uint32_t version = c.get<uint32_t>();
    if (version != 3) throw std::runtime_error(std::format("{}: GGUF version {} instead of 3", f.path(), version));
    if (first) version_ = version;
    uint64_t n_tensors = c.get<uint64_t>();
    uint64_t n_kv = c.get<uint64_t>();
    std::map<std::string, GgufValue> kv = read_kv(c, n_kv);
    const uint64_t alignment = read_alignment(kv, f.path());

    std::vector<TensorInfo> ts;
    ts.reserve(std::min<uint64_t>(n_tensors, f.size() / 24));  // a hostile count must not allocate
    for (uint64_t i = 0; i < n_tensors; i++) ts.push_back(read_tensor_info(c, f, fi));
    uint64_t data_start = align_up((uint64_t)(c.p - f.data()), alignment);
    for (auto& t : ts) {
        if (t.offset % alignment != 0 || data_start > f.size() || t.offset > f.size() - data_start ||
            t.bytes > f.size() - data_start - t.offset)
            throw std::runtime_error(std::format("{}: tensor {} lies outside the file", f.path(), t.name));
        t.data = f.data() + data_start + t.offset;
        if (by_name_.contains(t.name))
            throw std::runtime_error(std::format("{}: duplicate tensor {}", f.path(), t.name));
        by_name_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }
    if (first) kv_ = std::move(kv);
}

void MappedFile::drop(const uint8_t* ptr, size_t n) const {
    // Whole pages only, otherwise the neighbors would be hit
    const uintptr_t pg = 4096, a = (reinterpret_cast<uintptr_t>(ptr) + pg - 1) & ~(pg - 1),
                    e = (reinterpret_cast<uintptr_t>(ptr) + n) & ~(pg - 1);
    if (e <= a) return;
    madvise(reinterpret_cast<void*>(a), e - a, MADV_DONTNEED);
#if defined(__linux__)
    // macOS has no POSIX_FADV_DONTNEED; there the madvise above is all (docs/use/macos.md)
    posix_fadvise(fd_, (off_t)(a - reinterpret_cast<uintptr_t>(data_)), (off_t)(e - a), POSIX_FADV_DONTNEED);
#endif
}

void MappedFile::release(const uint8_t* ptr, size_t n) const {
    const uintptr_t pg = 4096, a = (reinterpret_cast<uintptr_t>(ptr) + pg - 1) & ~(pg - 1),
                    e = (reinterpret_cast<uintptr_t>(ptr) + n) & ~(pg - 1);
    if (e <= a) return;
    // PROT_NONE instead of munmap: the hole stays reserved, so a forgotten reader crashes instead of reading new memory
    void* r = mmap(reinterpret_cast<void*>(a), e - a, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
    if (r == MAP_FAILED) throw std::runtime_error(std::format("{}: release: {}", path_, std::strerror(errno)));
}

size_t Gguf::release(const TensorInfo& t) const {
    files_[t.file]->release(t.data, t.bytes);
    return t.bytes;
}

size_t Gguf::release_file(int i) const {
    files_[i]->release(files_[i]->data(), files_[i]->size());
    return files_[i]->size();
}

void Gguf::drop(const TensorInfo& t) const { files_[t.file]->drop(t.data, t.bytes); }
void Gguf::drop_range(const TensorInfo& t, uint64_t offset, size_t n) const {
    files_[t.file]->drop(t.data + offset, n);
}

void Gguf::read(const TensorInfo& t, uint64_t offset, void* dst, size_t n) const {
    if (offset + n > t.bytes) throw std::runtime_error(t.name + ": read past the end of the tensor");
    files_[t.file]->read(t.data + offset, dst, n);
}

const TensorInfo* Gguf::find(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

namespace {
const GgufValue& lookup(const std::map<std::string, GgufValue>& kv, const std::string& key) {
    auto it = kv.find(key);
    if (it == kv.end()) throw std::runtime_error(std::format("GGUF: key {} is missing", key));
    return it->second;
}
}  // namespace

uint64_t Gguf::get_uint(const std::string& key) const {
    const auto& v = lookup(kv_, key).v;
    if (auto* u = std::get_if<uint64_t>(&v)) return *u;
    if (auto* i = std::get_if<int64_t>(&v); i && *i >= 0) return (uint64_t)*i;
    throw std::runtime_error(std::format("GGUF: {} is not a non-negative integer", key));
}

double Gguf::get_float(const std::string& key) const {
    const auto& v = lookup(kv_, key).v;
    if (auto* d = std::get_if<double>(&v)) return *d;
    throw std::runtime_error(std::format("GGUF: {} is not a floating-point number", key));
}

std::string Gguf::get_string(const std::string& key) const {
    const auto& v = lookup(kv_, key).v;
    if (auto* s = std::get_if<std::string>(&v)) return *s;
    throw std::runtime_error(std::format("GGUF: {} is not a string", key));
}

std::vector<int64_t> Gguf::get_int_array(const std::string& key) const {
    const auto& v = lookup(kv_, key).v;
    auto* a = std::get_if<GgufArray>(&v);
    if (!a) throw std::runtime_error(std::format("GGUF: {} is not an array", key));
    std::vector<int64_t> out;
    for (const auto& e : *a) {
        if (auto* u = std::get_if<uint64_t>(&e.v))
            out.push_back((int64_t)*u);
        else if (auto* i = std::get_if<int64_t>(&e.v))
            out.push_back(*i);
        else
            throw std::runtime_error(std::format("GGUF: {} contains non-integers", key));
    }
    return out;
}

}  // namespace tona
