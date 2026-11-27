// Device and pinned allocations of the Mega backend and the upload of weights.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>

#include "backend/budget.h"
#include "backend/env.h"
#include "gpu/mega.h"
#include "gpu/mega_util.h"
#include "model/progress.h"

namespace tona::gpu {

int64_t vram_free_sysfs() {
    auto read = [](int card, const char* name) -> int64_t {
        std::ifstream in(std::format("/sys/class/drm/card{}/device/{}", card, name));
        long long v = -1;
        if (!(in >> v)) v = -1;
        return v;
    };
    for (int card = 0; card < 8; card++) {
        const int64_t total = read(card, "mem_info_vram_total"), used = read(card, "mem_info_vram_used");
        if (total > (4LL << 30) && used >= 0) return total - used;
    }
    throw std::runtime_error("gpu::Mega: cannot read the VRAM usage from sysfs");
}

KvSplit plan_kv_split(size_t n_main, size_t n_mtp, int64_t max_ctx, int64_t row) {
    const int64_t reserve = env::integer("TONA_VRAM_RESERVE_MB", 1024) << 20;
    int64_t free_b = vram_free_sysfs();
    if (backend::vram_limit_gb() >= 0) free_b = std::min(free_b, (int64_t)(backend::vram_limit_gb() * 1e9));
    const double gb = env::number("TONA_KV_VRAM_GB", -1.0);
    const size_t budget =
        gb >= 0 ? (size_t)(gb * 1e9) : KvSplit::default_budget(n_main, n_mtp, max_ctx, (size_t)row, free_b, reserve);
    const KvSplit p = KvSplit::plan(n_main, n_mtp, max_ctx, (size_t)row, budget);
    std::fprintf(stderr,
                 "KV cache: budget %.2f GB (VRAM free %.2f GB, %.2f GB per layer at %lld cells): %lld of %lld cells of "
                 "every attention layer in VRAM, MTP layer in %s\n",
                 budget / 1e9, free_b / 1e9, (double)max_ctx * (double)row / 1e9, (long long)max_ctx,
                 (long long)std::min(p.hot, max_ctx), (long long)max_ctx, p.mtp_host ? "RAM" : "VRAM");
    return p;
}

template <class T>
T* PinnedAlloc<T>::allocate(size_t n) {
    void* p = nullptr;
    HIPCHECK(trace::host_malloc(&p, n * sizeof(T) + 64, hipHostMallocDefault));
    return static_cast<T*>(p);
}
template <class T>
void PinnedAlloc<T>::deallocate(T* p, size_t) noexcept {
    (void)trace::host_free(p);
}
template struct PinnedAlloc<float>;
template struct PinnedAlloc<uint8_t>;

float* Mega::alloc(size_t floats) {
    void* p;
    HIPCHECK(trace::dev_malloc(&p, floats * sizeof(float) + 64));
    HIPCHECK(hipMemset(p, 0, floats * sizeof(float) + 64));
    allocs_.push_back(p);
    vram_ += floats * sizeof(float);
    return static_cast<float*>(p);
}

float* Mega::alloc_b(size_t n) {
    float* p = alloc(n * MEGA_BMAX);
    strides_[reinterpret_cast<uintptr_t>(p)] = {n * MEGA_BMAX * sizeof(float), (int64_t)n};
    rows_[reinterpret_cast<uintptr_t>(p)] = {n * MEGA_BMAX * sizeof(float), (int64_t)(n * sizeof(float))};
    return p;
}

Q8Vec Mega::alloc_q8_b(size_t n) {
    constexpr size_t B = MEGA_BMAX;
    Q8Vec v;
    v.q = reinterpret_cast<int8_t*>(alloc(n * B / 4 + 4));
    v.meta = reinterpret_cast<Q8Meta*>(alloc(n * B / 32 * sizeof(Q8Meta) / 4 + 2));
    strides_[reinterpret_cast<uintptr_t>(v.q)] = {n * B, (int64_t)n};
    rows_[reinterpret_cast<uintptr_t>(v.q)] = {n * B, (int64_t)n};
    rows_[reinterpret_cast<uintptr_t>(v.meta)] = {n * B / 32 * sizeof(Q8Meta), (int64_t)(n / 32 * sizeof(Q8Meta))};
    return v;
}

// Entry of the map whose buffer holds p, or 0 if p is in no per-token buffer
static int64_t lookup_per_token(const std::map<uintptr_t, std::pair<size_t, int64_t>>& m, const void* p) {
    const auto a = reinterpret_cast<uintptr_t>(p);
    auto it = m.upper_bound(a);
    if (it == m.begin()) return 0;
    --it;
    return a < it->first + it->second.first ? it->second.second : 0;
}

int64_t Mega::stride_of(const void* p) const { return lookup_per_token(strides_, p); }

int64_t Mega::row_bytes_at(const void* p) const { return lookup_per_token(rows_, p); }

float* Mega::alloc_state(size_t floats) {
    // Measuring mode TONA_STATE_HOST: state in pinned RAM instead of VRAM, to cost the memory of further sequences;
    // the GPU reaches it over PCIe
    static const bool host = env::flag("TONA_STATE_HOST");
    float* p;
    if (host) {
        HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&p), floats * sizeof(float) + 64, hipHostMallocDefault));
        std::memset(p, 0, floats * sizeof(float) + 64);
        pin_allocs_.push_back(p);
    } else {
        p = alloc(floats);
    }
    state_bufs_.emplace_back(p, floats * sizeof(float));
    return p;
}

// The 128 layer matrices of the target that a tile file would hold as 1019 (tools/coder_requant/repack_tile.py): the
// same name rule, so both routes give the same bytes and the same logits
static bool tile_on_upload(const TensorInfo& t, const std::vector<TensorInfo>& all) {
    // off by default: the kernel that has the tile loops is 2.0 ms slower per pass at 1 token and 1.3 to 1.7 ms at 5 to
    // 8 than the one without them, more than the tile layout gains (2026-11-15-pc-q6k-tile-spills.md)
    static const bool on = env::integer("TONA_Q6_K_TILE", 0) != 0;
    if (!on || t.type != 14 || t.dims.size() != 2 || t.dims[0] % 256 || t.dims[1] % 8) return false;
    if (&t < all.data() || &t >= all.data() + all.size())
        return false;  // the MTP head and the drafter stay as they are
    static const std::regex layer_matrix(R"(^blk\.\d+\.[a-z_]+\.weight$)");
    return std::regex_match(t.name, layer_matrix);
}

const void* Mega::upload(const TensorInfo& t) {
    progress::add((double)t.bytes);
    void* p;
    HIPCHECK(trace::dev_malloc(&p, t.bytes + kMatvecReadPad));
    if (tile_on_upload(t, m_.gguf().tensors())) {
        // One buffer for all tensors; the page-cache read of the source happens here instead of in hipMemcpy
        static thread_local std::vector<uint8_t> tiles;
        tiles.resize(t.bytes);
        const int64_t k = (int64_t)t.dims[0], n_tiles = (int64_t)t.dims[1] / 8;
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n_tiles; i += 16)
            tile_q6_k(t.data, k, i, std::min<int64_t>(i + 16, n_tiles), tiles.data());
        HIPCHECK(hipMemcpy(p, tiles.data(), t.bytes, hipMemcpyHostToDevice));
        tiled_.insert(p);
    } else {
        HIPCHECK(hipMemcpy(p, t.data, t.bytes, hipMemcpyHostToDevice));
    }
    allocs_.push_back(p);
    vram_ += t.bytes;
    return p;
}

const float* Mega::upload_f32(const TensorInfo& t) {
    progress::add((double)t.bytes);
    const auto v = dequant_host(t);
    float* p = alloc(v.size());
    HIPCHECK(hipMemcpy(p, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice));
    return p;
}

// BF16 -> Q8_0 (blocks of 32, f16 scale = max|w| / 127, like ggml quantize_row_q8_0)
const void* Mega::upload_q8_0(const TensorInfo& t) {
    progress::add((double)t.bytes);
    const auto v = dequant_host(t);
    const size_t nb = v.size() / 32;
    std::vector<uint8_t> out(nb * 34);
#pragma omp parallel for schedule(static)
    for (size_t b = 0; b < nb; b++) {
        float amax = 0;
        for (int k = 0; k < 32; k++) amax = std::max(amax, std::fabs(v[b * 32 + k]));
        const float dd = amax / 127.0f, id = dd > 0 ? 1.0f / dd : 0.0f;
        const _Float16 h = (_Float16)dd;
        std::memcpy(&out[b * 34], &h, 2);
        for (int k = 0; k < 32; k++) out[b * 34 + 2 + k] = (uint8_t)(int8_t)std::lround(v[b * 32 + k] * id);
    }
    void* p;
    HIPCHECK(trace::dev_malloc(&p, out.size() + kMatvecReadPad));
    HIPCHECK(hipMemcpy(p, out.data(), out.size(), hipMemcpyHostToDevice));
    allocs_.push_back(p);
    vram_ += out.size();
    return p;
}

}  // namespace tona::gpu
