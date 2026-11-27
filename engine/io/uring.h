#pragma once
// Small random reads through io_uring (without liburing): many requests to the NVMe at once. Used for the PLE n-gram
// table (28.8 GB, 16 rows of 90 bytes per token). direct: O_DIRECT (bypasses the page cache, 4 KiB sectors);
// otherwise buffered with POSIX_FADV_RANDOM (page-cache hits return at once, no readahead).
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct io_uring_sqe;
struct io_uring_cqe;

namespace tona {

class DirectRows {
public:
    // path: file, base: offset of the table in the file, row_bytes: bytes per row
    DirectRows(const std::string& path, uint64_t base, uint32_t row_bytes, bool direct = false, unsigned depth = 256);
    // Drops the file's page cache (for measurements)
    void evict() const;
    ~DirectRows();
    DirectRows(const DirectRows&) = delete;
    DirectRows& operator=(const DirectRows&) = delete;

    // Reads rows[0..n), row_bytes each, back to back into out
    void read(const int64_t* rows, size_t n, uint8_t* out);
    uint32_t row_bytes() const { return rb_; }

private:
    int fd_ = -1, ring_ = -1;
    uint64_t base_;
    uint32_t rb_, slot_;
    uint64_t align_;
    unsigned depth_;
    void *sq_ptr_ = nullptr, *cq_ptr_ = nullptr, *sqes_ptr_ = nullptr;
    size_t sq_sz_ = 0, cq_sz_ = 0, sqes_sz_ = 0;
    unsigned *sq_head_, *sq_tail_, *sq_mask_, *sq_array_, *cq_head_, *cq_tail_, *cq_mask_;
    ::io_uring_sqe* sqes_;
    ::io_uring_cqe* cqes_;
    uint8_t* buf_ = nullptr;  // one aligned slot per request in flight
    std::mutex mu_;
};

}  // namespace tona
