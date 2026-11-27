// An exception must not leave an OpenMP region (the program would terminate): the body of each iteration runs inside
// run(), the first exception is kept, and rethrow() raises it after the region. A failing pread of an expert in the
// decode pass then returns an error to the request instead of ending the server. The iterations that follow a failure
// are skipped (failed()), the others have finished by the end of the region. Use: declare one OmpErrors before the
// region, wrap the body of every iteration in run(), call rethrow() after the region.
#pragma once

#include <atomic>
#include <exception>
#include <utility>

namespace tona::backend {

class OmpErrors {
public:
    template <class F>
    void run(F&& body) noexcept {
        if (failed_.load(std::memory_order_relaxed)) return;
        try {
            std::forward<F>(body)();
        } catch (...) {
            bool expected = false;
            // the thread that wins the flag stores its exception; later ones are dropped
            if (failed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
                error_ = std::current_exception();
        }
    }
    bool failed() const { return failed_.load(std::memory_order_acquire); }
    // After the region (every thread is done): raise the first exception, if any
    void rethrow() const {
        if (failed_.load(std::memory_order_acquire)) std::rethrow_exception(error_);
    }

private:
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
};

}  // namespace tona::backend
