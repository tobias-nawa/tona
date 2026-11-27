// Matvec on the GPU for the GGUF formats of the model: y = W x, W quantized row by row in VRAM, unpacked while
// reading. Decode path (width 1), memory-bound; the yardstick is 315 GB/s.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace tona::gpu {

bool matvec_supported(uint32_t type);

// The kernels read weights in pieces of 16 bytes and may read up to this many bytes behind the end of a weight
// buffer: every buffer used with matvec needs this overhang (content does not matter).
constexpr size_t kMatvecReadPad = 64;

// int8 path: x rounded to int8 per block of 32 (d = max|x|/127), plus the sums of the two halves of 16 of the
// rounded values (for the minimums of the K formats and the offset of Q6_K). The weights are unpacked to int8 and
// multiplied with v_dot4_i32_iu8: one instruction for 4 products instead of ~15 instructions per value in the float
// path, which is compute-bound for the low bit widths (research/measurements/2026-10-23-matvec-int8.md).
struct Q8Meta {
    float d;
    int16_t s0, s1;
};
struct Q8Vec {
    int8_t* q;     // [n], 16-byte aligned
    Q8Meta* meta;  // [n / 32]
};
// n a multiple of 32
void quantize_q8(const float* x, int64_t n, Q8Vec out, hipStream_t stream = nullptr);
bool matvec_q8_supported(uint32_t type);

// One matvec: rows rows of in elements in the ggml format type (w, rows packed one after the other), x: [in] float
// (or xq: x already quantized, for matvec_q8 and for formats that have the int8 path), y: [rows] float. All device
// pointers.
struct MatvecTask {
    uint32_t type;
    int64_t in, rows;
    const void* w;
    const float* x;
    float* y;
    Q8Vec xq{};  // set: int8 path for formats that have it (x must already be quantized)
};

void matvec(const MatvecTask& t, hipStream_t stream = nullptr);
void matvec_q8(const MatvecTask& t, hipStream_t stream = nullptr);

// A matvec chain, e.g. all matvecs of one token. Tasks of a phase do not depend on each other, phase p+1 may read
// the results of phase p. Three ways to compute it (all bit-identical):
//   run_single:     one kernel per matvec (like gpu::Forward)
//   run_grouped:    one kernel per phase, all rows of the phase spread evenly over the waves
//   run_persistent: a single kernel for the whole chain, own grid barrier between the phases, with a time limit
//                   (aborts instead of hanging; diag() reports it)
struct ChainDiag {
    int64_t aborts;        // runs that exceeded the time limit (result incomplete)
    double start_skew_us;  // largest gap between the latest and the earliest block start in a run
    double max_wait_us;    // longest wait of a block at a barrier
};

class MatvecChain {
public:
    MatvecChain(const std::vector<MatvecTask>& tasks, const std::vector<int>& phase_start);
    ~MatvecChain();
    MatvecChain(const MatvecChain&) = delete;
    MatvecChain& operator=(const MatvecChain&) = delete;

    void run_single(hipStream_t s = nullptr);
    void run_grouped(hipStream_t s = nullptr);
    void run_persistent(int blocks = 0, hipStream_t s = nullptr);  // blocks <= 0: as many as fit at the same time
    ChainDiag diag();  // over all run_persistent since reset_diag; synchronizes
    void reset_diag();
    int max_blocks() const { return max_blocks_; }
    int phases() const { return (int)phase_start_.size() - 1; }

private:
    std::vector<MatvecTask> tasks_;
    std::vector<int> phase_start_;
    std::vector<int64_t> phase_rows_;  // units per phase
    void *d_tasks_ = nullptr, *d_phase_start_ = nullptr, *d_bar_ = nullptr;
    int max_blocks_ = 0, wall_khz_ = 0;
};

}  // namespace tona::gpu
