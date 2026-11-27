// Generic kernel: one wave (32 lanes) per row, each lane unpacks sub-blocks of 32 values and forms the dot product
// with x. Unpacking as in model/quant.cpp (from ggml-quants.c), only per sub-block.
// x is read straight from memory (L1/L2). The first version copied x into LDS per block: with 8 rows per block that
// was more bytes than the weights themselves, 5-7x slower (research/measurements/2026-10-22-matvec.md).
// Separate kernels for BF16 and Q6_K (2/3 of the bytes per token) read wide and contiguous.
#include <algorithm>
#include <format>
#include <stdexcept>

#include "backend/env.h"
#include "gpu/matvec.h"
#include "gpu/matvec_dev.h"

namespace tona::gpu {

namespace {
template <uint32_t T>
__global__ void __launch_bounds__(WAVE* ROWS_PER_BLOCK)
    k_matvec(const uint8_t* __restrict__ w, const float* __restrict__ xs, float* __restrict__ y, int in, int64_t rows,
             int64_t row_bytes) {
    const int lane = threadIdx.x % WAVE;
    const int64_t r = (int64_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    STAGE_DECL;
    if (r >= rows) return;
    const float s = row_generic<T>(w + r * row_bytes, xs, in, lane, stage);
    if (lane == 0) y[r] = s;
}

__global__ void __launch_bounds__(WAVE* ROWS_PER_BLOCK)
    k_matvec_bf16(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int in,
                  int64_t rows) {
    const int lane = threadIdx.x % WAVE;
    const int64_t r = (int64_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= rows) return;
    const float s = row_bf16(w + r * (int64_t)in * 2, x, in, lane);
    if (lane == 0) y[r] = s;
}

__global__ void __launch_bounds__(WAVE* ROWS_PER_BLOCK)
    k_matvec_q6k(const uint8_t* __restrict__ w, const float* __restrict__ xs, float* __restrict__ y, int in,
                 int64_t rows) {
    const int lane = threadIdx.x % WAVE;
    const int64_t r = (int64_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= rows) return;
    const float s = row_q6k(w + r * (int64_t)(in / 256) * 210, xs, in, lane);
    if (lane == 0) y[r] = s;
}

dim3 row_grid(int64_t rows) { return dim3((unsigned)((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK)); }

template <uint32_t T>
void launch(const void* w, const float* x, float* y, int64_t in, int64_t rows, hipStream_t st) {
    hipLaunchKernelGGL(k_matvec<T>, row_grid(rows), dim3(WAVE * ROWS_PER_BLOCK), 0, st, static_cast<const uint8_t*>(w),
                       x, y, (int)in, rows, row_bytes_of(T, in));
}

void launch_bf16(const void* w, const float* x, float* y, int64_t in, int64_t rows, hipStream_t st) {
    if (in % 8) return launch<30>(w, x, y, in, rows, st);
    hipLaunchKernelGGL(k_matvec_bf16, row_grid(rows), dim3(WAVE * ROWS_PER_BLOCK), 0, st,
                       static_cast<const uint8_t*>(w), x, y, (int)in, rows);
}

void launch_q6k(const void* w, const float* x, float* y, int64_t in, int64_t rows, hipStream_t st) {
    hipLaunchKernelGGL(k_matvec_q6k, row_grid(rows), dim3(WAVE * ROWS_PER_BLOCK), 0, st, static_cast<const uint8_t*>(w),
                       x, y, (int)in, rows);
}

// Trailing parameters of k_matvec_q8 as one struct: the offsets in the kernel argument buffer stay those of the
// separate parameters. The pointers and in stay separate parameters so that they keep __restrict__.
struct Q8Rows {
    int64_t rows, row_bytes;
    int L;
};

// Grid formats: tables into LDS per block (accesses with arbitrary index; from device memory each one is a separate
// load instruction over the vector memory path).
template <typename D, typename S>
__device__ __forceinline__ void copy_to_lds(D* dst, const S* src, int n) {
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (int i = threadIdx.x; i < n * (int)sizeof(S) / 16; i += blockDim.x) d4[i] = s4[i];
}

template <uint32_t T>
__device__ __forceinline__ void tables_to_lds(Tabs& tb) {
#if TONA_Q8_LDS_TABLES
    if constexpr (T == 22) {
        __shared__ __attribute__((aligned(16))) uint64_t g[1024], mk[256];
        copy_to_lds(g, d_iq2s_grid, 1024), copy_to_lds(mk, d_kmask8, 256);
        tb.iq2s = g, tb.m8 = mk;
        __syncthreads();
    } else if constexpr (T == 18) {
        __shared__ __attribute__((aligned(16))) uint32_t g[256];
        __shared__ __attribute__((aligned(16))) uint64_t mk[128];
        copy_to_lds(g, d_iq3xxs_grid, 256), copy_to_lds(mk, d_kmask3, 128);
        tb.iq3xxs = g, tb.m3 = mk;
        __syncthreads();
    } else if constexpr (T == 21) {
        __shared__ __attribute__((aligned(16))) uint32_t g[512];
        __shared__ __attribute__((aligned(16))) uint64_t mk[256];
        copy_to_lds(g, d_iq3s_grid, 512), copy_to_lds(mk, d_kmask8, 256);
        tb.iq3s = g, tb.m8 = mk;
        __syncthreads();
    }
#endif
}

template <uint32_t T>
__global__ void __launch_bounds__(WAVE* ROWS_PER_BLOCK)
    k_matvec_q8(const uint8_t* __restrict__ w, const int8_t* __restrict__ xq, const Q8Meta* __restrict__ meta,
                float* __restrict__ y, int in, Q8Rows d) {
    Tabs tb{d_iq2s_grid, d_iq3xxs_grid, d_iq3s_grid, d_kmask8, d_kmask3};
    tables_to_lds<T>(tb);
    const int lane = threadIdx.x % WAVE, per_wave = WAVE / d.L;
    const int64_t r = ((int64_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE) * per_wave + lane / d.L;
    if (r >= d.rows) return;  // whole row groups drop out together, the sum stays within the group
    float s[1];
    row_q8<T, 1>(Q8RowArgs{row_at<T>(w, r, d.row_bytes), xq, meta, in, lane % d.L, d.L, 1, 0, 1u}, tb, s);
    if (lane % d.L == 0) y[r] = s[0];
}

// One wave per block of 32: absolute maximum, rounding, sums of the two halves
__global__ void k_quantize_q8(const float* __restrict__ x, int64_t n, int8_t* __restrict__ q,
                              Q8Meta* __restrict__ meta) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % WAVE;
    const float v = i < n ? x[i] : 0.0f;
    float amax = fabsf(v);
    for (int off = WAVE / 2; off > 0; off /= 2) amax = fmaxf(amax, __shfl_xor(amax, off, WAVE));
    const float d = amax / 127.0f;
    const int qi = d > 0 ? (int)rintf(v / d) : 0;
    int s = qi;
    for (int off = 8; off > 0; off /= 2) s += __shfl_xor(s, off, WAVE);  // sum per half of 16
    if (i < n) q[i] = (int8_t)qi;
    if (i < n && lane == 0) meta[i / 32].d = d, meta[i / 32].s0 = (int16_t)s;
    if (i < n && lane == 16) meta[i / 32].s1 = (int16_t)s;
}

template <uint32_t T>
void launch_q8(const void* w, Q8Vec x, float* y, int64_t in, int64_t rows, hipStream_t st) {
    // Forced lanes per row only for experiments; read once per process.
    static const int force_L = (int)env::integer("TONA_Q8_LANES", 0);
    const int L = force_L > 0 ? force_L : lanes_per_row(in);
    const int64_t per_block = (int64_t)ROWS_PER_BLOCK * (WAVE / L);
    const dim3 grid((unsigned)((rows + per_block - 1) / per_block)), block(WAVE * ROWS_PER_BLOCK);
    hipLaunchKernelGGL(k_matvec_q8<T>, grid, block, 0, st, static_cast<const uint8_t*>(w), x.q, x.meta, y, (int)in,
                       Q8Rows{rows, row_bytes_of(T, in), L});
}

__global__ void __launch_bounds__(WAVE* ROWS_PER_BLOCK) __attribute__((amdgpu_waves_per_eu(CHAIN_WAVES)))
k_chain_phase(const DevTask* __restrict__ tasks, int a, int b) {
    STAGE_DECL;
    TABS_DECL;
    run_phase(PhaseRange{tasks, a, b},
              WaveCtx{(int64_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE, (int64_t)gridDim.x * ROWS_PER_BLOCK,
                      (int)(threadIdx.x % WAVE), stage, tabs});
}

// Trailing parameters of k_chain_persistent as one struct (offsets in the kernel argument buffer as for separate
// parameters): block start times, counters over all runs and the barrier time limit. They point into the buffer
// that also holds the barrier (see MatvecChain::run_persistent).
struct ChainSync {
    unsigned long long* start;
    unsigned long long* acc;
    uint64_t timeout_ticks;
};

__global__ void __launch_bounds__(WAVE* ROWS_PER_BLOCK) __attribute__((amdgpu_waves_per_eu(CHAIN_WAVES)))
k_chain_persistent(const DevTask* __restrict__ tasks, const int* __restrict__ phase_start, int n_phases, unsigned* bar,
                   ChainSync sync) {
    if (threadIdx.x == 0) {
        const unsigned long long t = wall_clock64();
        atomicMax(&sync.start[0], t);
        atomicMax(&sync.start[1], ~t);
    }
    STAGE_DECL;
    TABS_DECL;
    const WaveCtx wc{(int64_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE, (int64_t)gridDim.x * ROWS_PER_BLOCK,
                     (int)(threadIdx.x % WAVE), stage, tabs};
    for (int p = 0; p < n_phases; p++) {
        run_phase(PhaseRange{tasks, phase_start[p], phase_start[p + 1]}, wc);
        if (!grid_barrier(bar, sync.acc, gridDim.x, plain_limits(sync.timeout_ticks))) return;
        // after the first barrier all blocks have started: record the start skew of this run
        if (p == 0 && blockIdx.x == 0 && threadIdx.x == 0)
            atomicMax(&sync.acc[1], __hip_atomic_load(&sync.start[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) -
                                        ~__hip_atomic_load(&sync.start[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    }
}

void check_task(const MatvecTask& t) {
    // The chain kernels have no loops for NVFP4 and IQ2_XXS (ExtraQuants, matvec_phase.h): a task would compute nothing
    if (!matvec_supported(t.type) || t.in % 32 || t.type == 16 || t.type == 40)
        throw std::runtime_error(std::format("MatvecChain: type {} with row length {}", t.type, t.in));
}

// base: position of the task in the flat unit sequence of its phase
DevTask make_dev_task(const MatvecTask& t, int64_t base) {
    const bool q8 = t.xq.q && matvec_q8_supported(t.type);
    const int L = q8 ? lanes_per_row(t.in) : WAVE;
    const int64_t units = q8 ? (t.rows + WAVE / L - 1) / (WAVE / L) : t.rows;
    return {static_cast<const uint8_t*>(t.w),
            t.x,
            t.y,
            q8 ? t.xq.q : nullptr,
            q8 ? t.xq.meta : nullptr,
            t.rows,
            units,
            row_bytes_of(t.type, t.in),
            base,
            t.type,
            (int32_t)t.in,
            L};
}

// Tasks as the kernels see them; phase_units gets the number of units per phase.
std::vector<DevTask> make_dev_tasks(const std::vector<MatvecTask>& tasks, const std::vector<int>& phase_start,
                                    std::vector<int64_t>& phase_units) {
    std::vector<DevTask> dev(tasks.size());
    for (size_t p = 0; p + 1 < phase_start.size(); p++) {
        int64_t base = 0;
        for (int i = phase_start[p]; i < phase_start[p + 1]; i++) {
            check_task(tasks[i]);
            dev[i] = make_dev_task(tasks[i], base);
            base += dev[i].units;
        }
        phase_units.push_back(base);
    }
    return dev;
}

void upload(void** dst, const void* src, size_t n) {
    if (hipMalloc(dst, n) != hipSuccess || hipMemcpy(*dst, src, n, hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("MatvecChain: upload failed");
}

// Blocks of the persistent kernel that fit on the GPU at the same time
int resident_blocks() {
    int per_cu = 0, cus = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, k_chain_persistent, WAVE * ROWS_PER_BLOCK, 0) !=
            hipSuccess ||
        hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0) != hipSuccess)
        throw std::runtime_error("MatvecChain: cannot determine occupancy");
    return per_cu * cus;
}

}  // namespace

MatvecChain::MatvecChain(const std::vector<MatvecTask>& tasks, const std::vector<int>& phase_start)
    : tasks_(tasks), phase_start_(phase_start) {
    upload_tables();
    if (phase_start.empty() || phase_start.front() != 0 || phase_start.back() != (int)tasks.size())
        throw std::runtime_error("MatvecChain: phase boundaries do not match the tasks");
    const std::vector<DevTask> dev = make_dev_tasks(tasks, phase_start, phase_rows_);
    upload(&d_tasks_, dev.data(), dev.size() * sizeof(DevTask));
    upload(&d_phase_start_, phase_start.data(), phase_start.size() * sizeof(int));
    max_blocks_ = resident_blocks();
    if (hipDeviceGetAttribute(&wall_khz_, hipDeviceAttributeWallClockRate, 0) != hipSuccess || wall_khz_ <= 0)
        throw std::runtime_error("MatvecChain: clock rate of wall_clock64 unknown");
    if (hipMalloc(&d_bar_, 64) != hipSuccess || hipMemset(d_bar_, 0, 64) != hipSuccess)
        throw std::runtime_error("MatvecChain: cannot create the barrier");
}

MatvecChain::~MatvecChain() {
    (void)hipFree(d_tasks_);
    (void)hipFree(d_phase_start_);
    (void)hipFree(d_bar_);
}

void MatvecChain::run_single(hipStream_t s) {
    for (const MatvecTask& t : tasks_)
        if (t.xq.q && matvec_q8_supported(t.type))
            matvec_q8(t, s);
        else
            matvec(t, s);
}

void MatvecChain::run_grouped(hipStream_t s) {
    const auto* tasks = static_cast<const DevTask*>(d_tasks_);
    for (size_t p = 0; p + 1 < phase_start_.size(); p++) {
        const unsigned blocks = (unsigned)((phase_rows_[p] + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        hipLaunchKernelGGL(k_chain_phase, dim3(blocks), dim3(WAVE * ROWS_PER_BLOCK), 0, s, tasks, phase_start_[p],
                           phase_start_[p + 1]);
    }
}

void MatvecChain::run_persistent(int blocks, hipStream_t s) {
    if (blocks <= 0 || blocks > max_blocks_) blocks = max_blocks_;
    // Reset barrier and start times per run (one small memset); acc stays
    if (hipMemsetAsync(d_bar_, 0, 32, s) != hipSuccess) throw std::runtime_error("MatvecChain: memset failed");
    auto* base = static_cast<uint8_t*>(d_bar_);
    const ChainSync sync{reinterpret_cast<unsigned long long*>(base + 16),
                         reinterpret_cast<unsigned long long*>(base + 32), (uint64_t)wall_khz_ * CHAIN_TIMEOUT_MS};
    hipLaunchKernelGGL(k_chain_persistent, dim3(blocks), dim3(WAVE * ROWS_PER_BLOCK), 0, s,
                       static_cast<const DevTask*>(d_tasks_), static_cast<const int*>(d_phase_start_),
                       (int)phase_start_.size() - 1, static_cast<unsigned*>(d_bar_), sync);
}

ChainDiag MatvecChain::diag() {
    unsigned long long acc[3];
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(acc, static_cast<uint8_t*>(d_bar_) + 32, sizeof acc, hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("MatvecChain: diagnostics unreadable");
    const double us = 1e3 / wall_khz_;
    return {(int64_t)acc[0], (double)acc[1] * us, (double)acc[2] * us};
}

void MatvecChain::reset_diag() {
    if (hipMemset(d_bar_, 0, 64) != hipSuccess) throw std::runtime_error("MatvecChain: memset failed");
}

bool matvec_supported(uint32_t type) { return row_bytes_of(type, 256) > 0; }

void quantize_q8(const float* x, int64_t n, Q8Vec out, hipStream_t st) {
    if (n % 32) throw std::runtime_error(std::format("gpu::quantize_q8: length {}", n));
    hipLaunchKernelGGL(k_quantize_q8, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, st, x, n, out.q, out.meta);
}

// NVFP4 and IQ2_XXS (16, 40) count as supported for the program builder of the megakernel, whose kernel for a model
// that has them runs the loops (ExtraQuants); matvec_q8 and the chain kernels of this unit have none
bool matvec_q8_supported(uint32_t type) {
    switch (type) {
        case 8:
        case 12:
        case 13:
        case 14:
        case 20:
        case 23:
        case 22:
        case 18:
        case 21:
        case 42:
        case 1019:
        case 16:
        case 40:
            return true;
        default:
            break;
    }
    return false;
}

void matvec_q8(const MatvecTask& t, hipStream_t st) {
    upload_tables();
    if (t.in % 32 != 0) throw std::runtime_error(std::format("gpu::matvec_q8: row length {}", t.in));
    switch (t.type) {
        case 8:
            return launch_q8<8>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 12:
            return launch_q8<12>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 13:
            return launch_q8<13>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 14:
            return launch_q8<14>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 1019:
            return launch_q8<1019>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 20:
            return launch_q8<20>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 42:
            return launch_q8<42>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 23:
            return launch_q8<23>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 22:
            return launch_q8<22>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 18:
            return launch_q8<18>(t.w, t.xq, t.y, t.in, t.rows, st);
        case 21:
            return launch_q8<21>(t.w, t.xq, t.y, t.in, t.rows, st);
        default:
            break;
    }
    throw std::runtime_error(std::format("gpu::matvec_q8: ggml type {} not supported", t.type));
}

namespace {

// TONA_MATVEC_V1: BF16 and Q6_K through the generic kernel, to compare before/after in one program. True if it ran.
bool matvec_v1(const MatvecTask& t, hipStream_t st) {
    static const bool v1 = env::flag("TONA_MATVEC_V1");
    if (!v1) return false;
    if (t.type == 14)
        launch<14>(t.w, t.x, t.y, t.in, t.rows, st);
    else if (t.type == 30)
        launch<30>(t.w, t.x, t.y, t.in, t.rows, st);
    else
        return false;
    return true;
}

// Format dispatch of the float path; throws for a format without kernel
void launch_float(const MatvecTask& t, hipStream_t st) {
    switch (t.type) {
        case 0:
            return launch<0>(t.w, t.x, t.y, t.in, t.rows, st);
        case 1:
            return launch<1>(t.w, t.x, t.y, t.in, t.rows, st);
        case 30:
            return launch_bf16(t.w, t.x, t.y, t.in, t.rows, st);
        case 8:
            return launch<8>(t.w, t.x, t.y, t.in, t.rows, st);
        case 20:
            return launch<20>(t.w, t.x, t.y, t.in, t.rows, st);
        case 42:
            return launch<42>(t.w, t.x, t.y, t.in, t.rows, st);
        case 12:
            return launch<12>(t.w, t.x, t.y, t.in, t.rows, st);
        case 13:
            return launch<13>(t.w, t.x, t.y, t.in, t.rows, st);
        case 14:
            return launch_q6k(t.w, t.x, t.y, t.in, t.rows, st);
        case 22:
            return launch<22>(t.w, t.x, t.y, t.in, t.rows, st);
        case 18:
            return launch<18>(t.w, t.x, t.y, t.in, t.rows, st);
        case 21:
            return launch<21>(t.w, t.x, t.y, t.in, t.rows, st);
        case 23:
            return launch<23>(t.w, t.x, t.y, t.in, t.rows, st);
        default:
            break;
    }
    throw std::runtime_error(std::format("gpu::matvec: ggml type {} not supported", t.type));
}

}  // namespace

void matvec(const MatvecTask& t, hipStream_t st) {
    upload_tables();
    if (t.in % 32 != 0) throw std::runtime_error(std::format("gpu::matvec: row length {}", t.in));
    if (!matvec_v1(t, st)) launch_float(t, st);
}

}  // namespace tona::gpu
