#include "io/uring.h"

#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <format>
#include <stdexcept>

namespace tona {

namespace {
template <class T>
T load_acq(const unsigned* p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
void store_rel(unsigned* p, unsigned v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
}  // namespace

DirectRows::DirectRows(const std::string& path, uint64_t base, uint32_t row_bytes, bool direct, unsigned depth)
    : base_(base), rb_(row_bytes), align_(direct ? 4096 : 1), depth_(depth) {  // btrfs: direct I/O in 4 KiB sectors
    slot_ = (((row_bytes + 4095) / 4096 + 1) * 4096);
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
    if (fd_ < 0) throw std::runtime_error(std::format("{}: {}", path, std::strerror(errno)));
    if (!direct) posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
    io_uring_params p{};
    ring_ = (int)syscall(__NR_io_uring_setup, depth, &p);
    if (ring_ < 0) {
        ::close(fd_);
        throw std::runtime_error(std::format("io_uring_setup: {}", std::strerror(errno)));
    }
    depth_ = p.sq_entries;
    sq_sz_ = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    cq_sz_ = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    const bool single = p.features & IORING_FEAT_SINGLE_MMAP;
    if (single) sq_sz_ = cq_sz_ = std::max(sq_sz_, cq_sz_);
    sq_ptr_ = mmap(nullptr, sq_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_, IORING_OFF_SQ_RING);
    cq_ptr_ = single
                  ? sq_ptr_
                  : mmap(nullptr, cq_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_, IORING_OFF_CQ_RING);
    sqes_sz_ = p.sq_entries * sizeof(io_uring_sqe);
    sqes_ptr_ = mmap(nullptr, sqes_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_, IORING_OFF_SQES);
    if (sq_ptr_ == MAP_FAILED || cq_ptr_ == MAP_FAILED || sqes_ptr_ == MAP_FAILED)
        throw std::runtime_error("io_uring: mmap failed");
    auto* sq = static_cast<uint8_t*>(sq_ptr_);
    auto* cq = static_cast<uint8_t*>(cq_ptr_);
    sq_head_ = reinterpret_cast<unsigned*>(sq + p.sq_off.head);
    sq_tail_ = reinterpret_cast<unsigned*>(sq + p.sq_off.tail);
    sq_mask_ = reinterpret_cast<unsigned*>(sq + p.sq_off.ring_mask);
    sq_array_ = reinterpret_cast<unsigned*>(sq + p.sq_off.array);
    cq_head_ = reinterpret_cast<unsigned*>(cq + p.cq_off.head);
    cq_tail_ = reinterpret_cast<unsigned*>(cq + p.cq_off.tail);
    cq_mask_ = reinterpret_cast<unsigned*>(cq + p.cq_off.ring_mask);
    cqes_ = reinterpret_cast<io_uring_cqe*>(cq + p.cq_off.cqes);
    sqes_ = static_cast<io_uring_sqe*>(sqes_ptr_);
    buf_ = static_cast<uint8_t*>(std::aligned_alloc(4096, (size_t)depth_ * slot_));
    if (!buf_) throw std::runtime_error("DirectRows: out of memory");
}

DirectRows::~DirectRows() {
    std::free(buf_);
    if (sqes_ptr_ && sqes_ptr_ != MAP_FAILED) munmap(sqes_ptr_, sqes_sz_);
    if (cq_ptr_ && cq_ptr_ != MAP_FAILED && cq_ptr_ != sq_ptr_) munmap(cq_ptr_, cq_sz_);
    if (sq_ptr_ && sq_ptr_ != MAP_FAILED) munmap(sq_ptr_, sq_sz_);
    if (ring_ >= 0) ::close(ring_);
    if (fd_ >= 0) ::close(fd_);
}

void DirectRows::evict() const { posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED); }

void DirectRows::read(const int64_t* rows, size_t n, uint8_t* out) {
    std::lock_guard lk(mu_);
    // Slot per request: reuse freed slots through a list
    std::vector<unsigned> free_slots(depth_);
    for (unsigned i = 0; i < depth_; i++) free_slots[i] = depth_ - 1 - i;
    std::vector<uint32_t> skip(depth_);  // offset of the row from the aligned start, per slot
    std::vector<size_t> which(depth_);   // row per slot
    size_t next = 0, done = 0;
    unsigned inflight = 0;
    while (done < n) {
        unsigned tail = *sq_tail_, queued = 0;
        const unsigned mask = *sq_mask_;
        while (next < n && inflight < depth_) {
            const uint64_t off = base_ + (uint64_t)rows[next] * rb_;
            const uint64_t a = off / align_ * align_, e = (off + rb_ + align_ - 1) / align_ * align_;
            const unsigned s = free_slots.back();
            free_slots.pop_back();
            skip[s] = (uint32_t)(off - a), which[s] = next;
            io_uring_sqe* q = &sqes_[tail & mask];
            std::memset(q, 0, sizeof *q);
            q->opcode = IORING_OP_READ;
            q->fd = fd_;
            q->addr = reinterpret_cast<uint64_t>(buf_ + (size_t)s * slot_);
            q->len = (uint32_t)(e - a);
            q->off = a;
            q->user_data = s;
            sq_array_[tail & mask] = tail & mask;
            tail++, queued++, inflight++, next++;
        }
        store_rel(sq_tail_, tail);
        const int r = (int)syscall(__NR_io_uring_enter, ring_, queued, 1, IORING_ENTER_GETEVENTS, nullptr, 0);
        if (r < 0 && errno != EINTR) throw std::runtime_error(std::format("io_uring_enter: {}", std::strerror(errno)));
        unsigned head = *cq_head_;
        const unsigned ctail = load_acq<unsigned>(cq_tail_), cmask = *cq_mask_;
        for (; head != ctail; head++) {
            const io_uring_cqe& c = cqes_[head & cmask];
            const unsigned s = (unsigned)c.user_data;
            if (c.res < (int)(skip[s] + rb_))
                throw std::runtime_error(
                    std::format("DirectRows: read failed ({})", c.res < 0 ? std::strerror(-c.res) : "short read"));
            std::memcpy(out + which[s] * rb_, buf_ + (size_t)s * slot_ + skip[s], rb_);
            free_slots.push_back(s);
            inflight--, done++;
        }
        store_rel(cq_head_, head);
    }
}

}  // namespace tona
