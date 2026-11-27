// macOS stand-in for io/uring.cpp behind the same interface (DirectRows): macOS has no io_uring, so every row is one
// pread. Synchronous, one request at a time; enough for the unit test and the shared tools. The Mac backend decides
// its own read path (research/21-backends.md).
// The ring fields of io/uring.h stay unused here; the header remains the Linux one unchanged
#pragma clang diagnostic ignored "-Wunused-private-field"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <format>
#include <stdexcept>

#include "io/uring.h"

namespace tona {

DirectRows::DirectRows(const std::string& path, uint64_t base, uint32_t row_bytes, bool direct, unsigned depth)
    : base_(base), rb_(row_bytes), align_(1), depth_(depth) {
    slot_ = row_bytes;
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error(std::format("{}: {}", path, std::strerror(errno)));
    // F_NOCACHE: the macOS counterpart of O_DIRECT (reads bypass the unified buffer cache), without alignment rules.
    // F_RDAHEAD off: the counterpart of POSIX_FADV_RANDOM (no readahead for scattered rows).
    if (direct && ::fcntl(fd_, F_NOCACHE, 1) != 0) {
        ::close(fd_);
        throw std::runtime_error(std::format("{}: F_NOCACHE: {}", path, std::strerror(errno)));
    }
    if (!direct) ::fcntl(fd_, F_RDAHEAD, 0);
}

DirectRows::~DirectRows() {
    if (fd_ >= 0) ::close(fd_);
}

// macOS offers no unprivileged way to drop one file's pages from the cache (purge needs root), so a measurement that
// needs a cold cache on the Mac opens the file with direct = true instead
void DirectRows::evict() const {}

void DirectRows::read(const int64_t* rows, size_t n, uint8_t* out) {
    std::lock_guard lk(mu_);
    for (size_t i = 0; i < n; i++) {
        const off_t off = (off_t)(base_ + (uint64_t)rows[i] * rb_);
        size_t got = 0;
        while (got < rb_) {
            const ssize_t r = ::pread(fd_, out + i * rb_ + got, rb_ - got, off + (off_t)got);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0)
                throw std::runtime_error(
                    std::format("DirectRows: read failed ({})", r < 0 ? std::strerror(errno) : "short read"));
            got += (size_t)r;
        }
    }
}

}  // namespace tona
