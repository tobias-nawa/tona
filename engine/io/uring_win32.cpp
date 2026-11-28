// Windows stand-in for io/uring.cpp behind the same interface (DirectRows): positional ReadFile with OVERLAPPED.
// Synchronous, one request at a time; enough for the unit test and the shared tools. The ring fields of io/uring.h
// stay unused here (sq_ptr_ holds the Win32 HANDLE), so the header remains the Linux one unchanged.
#include <windows.h>

#include <format>
#include <stdexcept>

#include "io/uring.h"

namespace tona {

DirectRows::DirectRows(const std::string& path, uint64_t base, uint32_t row_bytes, bool direct, unsigned depth)
    : base_(base), rb_(row_bytes), slot_(row_bytes), align_(1), depth_(depth) {
    (void)direct;  // FILE_FLAG_NO_BUFFERING requires sector-aligned offsets and row sizes
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS,
                           nullptr);
    if (h == INVALID_HANDLE_VALUE)
        throw std::runtime_error(std::format("{}: CreateFileA failed (error {})", path, GetLastError()));
    sq_ptr_ = h;
}

DirectRows::~DirectRows() {
    if (sq_ptr_) CloseHandle(static_cast<HANDLE>(sq_ptr_));
}

// Windows offers no unprivileged way to drop one file's pages from the standby cache
void DirectRows::evict() const {}

void DirectRows::read(const int64_t* rows, size_t n, uint8_t* out) {
    std::lock_guard lk(mu_);
    HANDLE h = static_cast<HANDLE>(sq_ptr_);
    for (size_t i = 0; i < n; i++) {
        const uint64_t off = base_ + static_cast<uint64_t>(rows[i]) * rb_;
        size_t got_total = 0;
        while (got_total < rb_) {
            const uint64_t pos = off + got_total;
            OVERLAPPED ov{};
            ov.Offset = static_cast<DWORD>(pos & 0xffffffffu);
            ov.OffsetHigh = static_cast<DWORD>(pos >> 32);
            DWORD got = 0;
            if (!ReadFile(h, out + i * rb_ + got_total, static_cast<DWORD>(rb_ - got_total), &got, &ov) || got == 0)
                throw std::runtime_error(std::format("DirectRows: ReadFile failed (error {})", GetLastError()));
            got_total += got;
        }
    }
}

}  // namespace tona
