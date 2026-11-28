// GGUF reader: header, metadata, tensor directory, split files. Files are only mapped (mmap, read-only);
// tensors point straight into the mapping, nothing is copied.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace tona {

// ggml_type as of ggml in llama.cpp b11822 (ggml/include/ggml.h)
struct GgmlTypeInfo {
    const char* name;
    uint32_t block_elems;  // elements per block
    uint32_t block_bytes;  // bytes per block
};
const GgmlTypeInfo* ggml_type_info(uint32_t type);

struct GgufValue;
using GgufArray = std::vector<GgufValue>;
struct GgufValue {
    std::variant<uint64_t, int64_t, double, bool, std::string, GgufArray> v;
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;  // ggml order: dims[0] is the contiguous axis
    uint32_t type = 0;
    uint64_t offset = 0;  // relative to the file's data section
    uint64_t bytes = 0;
    int file = 0;  // index of the split file
    const uint8_t* data = nullptr;

    uint64_t elements() const;
};

// Bytes of n elements of a row of a ggml type (n a whole number of blocks)
inline int64_t type_row_bytes(uint32_t type, int64_t n) {
    const GgmlTypeInfo* ti = ggml_type_info(type);
    return n / ti->block_elems * ti->block_bytes;
}

// Bytes of one expert in a stacked tensor [.., n_expert]
inline size_t bytes_per_expert(const TensorInfo* t) { return t->bytes / t->dims[2]; }

class MappedFile {
public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const uint8_t* data() const { return data_; }
    uint64_t size() const { return size_; }
    const std::string& path() const { return path_; }
    bool contains(const uint8_t* p) const { return p >= data_ && p < data_ + size_; }
    // Reads n bytes from ptr (inside this mapping) with pread: no page faults, straight from the page cache
    void read(const uint8_t* ptr, void* dst, size_t n) const;
    // Drops a range from the mapping and the page cache (madvise + fadvise DONTNEED), e.g. after uploading it
    void drop(const uint8_t* ptr, size_t n) const;
    // Replaces the whole pages of a range with an inaccessible anonymous mapping: the file mapping is gone, a later
    // read faults at once. Only the whole pages inside the range, as in drop().
    void release(const uint8_t* ptr, size_t n) const;

private:
#ifdef _WIN32
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
    std::string path_;
    const uint8_t* data_ = nullptr;
    uint64_t size_ = 0;
};

class Gguf {
public:
    // Opens path; for part 1 of a split file (split.count > 1) the other parts are loaded too.
    explicit Gguf(const std::string& path);

    uint32_t version() const { return version_; }
    const std::map<std::string, GgufValue>& kv() const { return kv_; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* find(const std::string& name) const;
    int files() const { return (int)files_.size(); }
    const std::string& file_path(int i) const { return files_[i]->path(); }
    uint64_t file_offset(const TensorInfo& t) const { return (uint64_t)(t.data - files_[t.file]->data()); }
    // Reads bytes of a tensor (or a part of it) with pread instead of through the mapping
    void read(const TensorInfo& t, uint64_t offset, void* dst, size_t n) const;
    // Drops a tensor from the page cache so the RAM goes to the experts (they do not all fit with 32 GB)
    void drop(const TensorInfo& t) const;
    void drop_range(const TensorInfo& t, uint64_t offset, size_t n) const;
    // Drops the tensor and removes its pages from the address space (MappedFile::release). Returns the bytes released.
    size_t release(const TensorInfo& t) const;
    // The same for the whole file of the tensor with index i, regardless of tensors
    size_t release_file(int i) const;

    // Metadata with type checks; they throw if the key is missing or has the wrong type
    uint64_t get_uint(const std::string& key) const;
    double get_float(const std::string& key) const;
    std::string get_string(const std::string& key) const;
    std::vector<int64_t> get_int_array(const std::string& key) const;
    bool has(const std::string& key) const { return kv_.contains(key); }

private:
    void parse(int file_index, bool first);

    uint32_t version_ = 0;
    std::map<std::string, GgufValue> kv_;
    std::vector<TensorInfo> tensors_;
    std::map<std::string, size_t> by_name_;
    std::vector<std::unique_ptr<MappedFile>> files_;
};

}  // namespace tona
