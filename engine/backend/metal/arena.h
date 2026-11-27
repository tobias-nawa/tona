// The expert arena: for a model whose routed experts do not fit the GPU's working set (the Coder, 25.1 GB of experts
// on the M1 Max 32 GB). Without it the experts stay in the file mapping and the page cache re-reads 23 to 43 GB per
// pass (research/measurements/2026-11-15-mac-coder-baseline.md, run 2b). The arena is memory of the process's own in
// blocks: the most used experts by the profile (backend/expert_tiers.h select_pinned, one ranking over all layers),
// read once at start, a cache of recently read experts and staging slots. After the router of a layer the pass stops,
// the experts it routes to that are not in the arena are read from the file (pread with F_NOCACHE, no read-ahead, no
// page cache), and the layer's address table points at them; the expert kernels read every expert through that table
// (kernels/moe_table.metal). Under memory pressure blocks are freed and later read again (arena_govern.mm).
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "backend/metal/device.h"
#include "backend/ram_governor.h"
#include "model/qwen4exp.h"

namespace tona::metal {

// One matrix of one expert in the file and its slot: the whole pages around it are read, the kernel reads from `skew`
struct ExpertPiece {
    int fd = -1;
    uint64_t file_at = 0;  // page-aligned start in the file
    uint64_t bytes = 0;    // whole pages
    uint64_t skew = 0;     // the matrix starts this far into the pages
};

struct ArenaStats {
    // Lookahead at the stops it predicted: experts predicted (not resident), routed (not resident), both
    uint64_t predicted = 0, routed = 0, useful = 0;
    uint64_t stops = 0;                           // layer stops of the passes (with and without cold experts)
    uint64_t layers = 0, experts = 0, bytes = 0;  // cold reads: stops that read, experts read, bytes read
    double seconds = 0;                           // wall time of the cold reads
    double waited = 0;                            // wall time from the commit before a stop to its completion
    double gpu = 0;                               // GPU time of those command buffers
    // Purgeable blocks: made non-volatile, found purged, made volatile; experts read again into purged blocks; the
    // wall time of the stops' wakes
    uint64_t woken = 0, purged = 0, settled = 0, refilled = 0;
    double wake_s = 0;
    uint64_t demands = 0;  // samples that saw other programs' demand (arena_purge.mm demand())
};

// A block of the arena: offsets [at, at + bytes) of the arena's address space in one buffer, nil while freed
struct ArenaBlock {
    id<MTLBuffer> buf = nil;
    uint64_t at = 0, bytes = 0;
    // TONA_METAL_ARENA_PURGEABLE (arena_purge.mm): volatile while no pass reads it
    bool vol = false;
    uint64_t used = 0;  // the last pass that routed to it, 0: none
};

int64_t own_compressed();  // bytes of this process in the compressor (arena_purge.mm), -1 if unknown

class Arena {
public:
    // nullptr if TONA_METAL_ARENA_GB is 0, or unset and the model fits the working set; "auto" or unset: the budget
    // from the free memory at load(); without a profile (TONA_EXPERT_PROFILE) all experts rank equal
    static std::unique_ptr<Arena> create(Device& dev, const Qwen4ExpModel& m);
    ~Arena();
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Plans, allocates and reads the resident experts. Call once the rest of the engine is loaded: with "auto" the
    // budget is what is left (arena_auto_budget); mapped: our weight mappings, gpu: all our buffers but the arena
    void load(Device& dev, uint64_t mapped, uint64_t gpu);

    // The routed expert tensors held here: Weights leaves them out of its buffers
    std::vector<std::string> tensor_names() const;
    bool holds(T t) const { return where_.count(t) != 0; }
    // Layer of an expert tensor held here
    uint32_t layer_of(T t) const { return where_.at(t).first; }
    // The address table of an expert tensor: [experts] GPU addresses
    Buf table(T t) const;
    // Reads the experts the routing ids[0..n) of layer l uses that are not resident and points the table at them. Call
    // while the GPU waits before the layer's routed experts. keep: put them into the layer's cache of recently read
    // experts (decode and verify passes); a prefill chunk reads into the staging slots and leaves the cache as it is.
    void read_cold(uint32_t l, const int32_t* ids, int n, bool keep);
    // Lookahead: starts reading the cold experts that ids[0..n) predict for layer l into its cache, in the background;
    // the next read_cold waits for them first
    void prefetch(uint32_t l, const int32_t* ids, int n);
    // A stop's wait for the GPU (commit to completion); TONA_METAL_ARENA_LOG=n: a line every n stops
    void note_stop(double waited, double gpu);
    uint64_t bytes() const { return held_.load(); }
    const ArenaStats& stats() const { return stats_; }

    // Held by the runner for each forward and prefill chunk, until its command buffers completed: blocks are freed and
    // read again only outside, so no command buffer reads a freed block
    class PassScope {
    public:
        explicit PassScope(Arena* a);
        ~PassScope();
        PassScope(const PassScope&) = delete;
        PassScope& operator=(const PassScope&) = delete;

    private:
        Arena* a_;
    };

private:
    Arena(Device& dev, const Qwen4ExpModel& m, int64_t budget, std::vector<uint32_t> profile);
    static double reserve_gb();
    int64_t auto_budget(Device& dev, uint64_t mapped, uint64_t gpu) const;
    void plan(int64_t budget);
    uint64_t lay_out(size_t pins);
    void allocate(Device& dev, bool all);
    void fill();
    void fill_in_steps();
    int64_t drop_cache() const;
    void trace(uint32_t l, const int32_t* ids, int n, const std::vector<uint8_t>& cold);
    // Where cold expert e of layer l goes: its cache slot (read = false on a hit), a free or the least recently used
    // cache slot, or the next staging slot at `staging`
    uint64_t place(uint32_t l, int e, bool keep, uint64_t& staging, bool& read);
    using Jobs = std::vector<std::pair<const ExpertPiece*, uint8_t*>>;
    static void read_pieces(const Jobs& jobs);
    void point(uint32_t l, int e, uint64_t at, Jobs* jobs);  // the table at offset `at`, reads into it if jobs
    void finish_ahead();
    const ExpertPiece& piece(uint32_t l, int m, int e) const { return pieces_[((size_t)l * 3 + m) * experts_ + e]; }
    uint64_t* tab(uint32_t l, int m) const { return (uint64_t*)tab_.contents + ((size_t)l * 3 + m) * experts_; }
    ArenaBlock& block(uint64_t at);
    uint8_t* host(uint64_t at);
    uint64_t gpu(uint64_t at);

    // arena_govern.mm: the pressure monitor and the release and re-read of blocks
    void start_governor();
    void stop_governor();
    void monitor();
    void govern(bool pass_running);
    void begin_pass();
    void end_pass();
    int64_t shrink(int64_t bytes);
    int64_t grow(int64_t bytes);
    void free_block(size_t k);
    void read_block(size_t k);
    void residency(id<MTLBuffer> b, bool add);
    void read_pins(size_t k);

    // arena_purge.mm: blocks the system may take at once instead of a reserve (research/53-mac-purgeable-arena.md)
    bool purge_ = false;
    std::vector<int32_t> pin_of_;  // [layer][expert]: its index in pins_, -1: not pinned
    std::vector<uint8_t> pin_ok_;  // [pin]: its bytes are in its block (a purged block reads zero)
    bool in_pressure_ = false;     // a pressure episode was logged; its end logs the experts read again meanwhile
    uint64_t refilled_at_ = 0;
    void wake(uint32_t l, const int32_t* ids, int n);  // the blocks a stop routes to non-volatile, purged experts read
    void settle();                                     // every pin block volatile, the longest unused first
    void purged(size_t k);                             // block k came back empty: its pins invalid, the gap counted
    void read_level();                                 // the system's memory pressure level into pressure_
    // Other programs' demand: the machine compressed pages in the last sample and we did not (idle, the machine
    // compresses nothing: 0 of 600 samples of 50 ms, research/53-mac-purgeable-arena.md). Then every block the
    // running pass has not routed to goes volatile at once, from the monitor thread
    void demand();
    std::mutex state_mu_;  // the blocks' purgeable state, between wake() on the engine and demand() on the monitor
    uint64_t compressions_ = 0;
    int64_t own_at_ = 0;
    std::atomic<int64_t> refill_bytes_{0};  // read again into purged blocks: our own demand, not other programs'
    int64_t refill_at_ = 0;
    // On demand the arena gives way: a stop that finds a block purged frees it (its experts come through the staging
    // slots) instead of reading it again; regrow() reads freed blocks back a growth step at a time, before a pass, and
    // a step after which other programs compressed doubles the clean samples the next one waits for
    void regrow();
    bool wake_block(size_t k);
    std::atomic<uint64_t> samples_{0}, last_demand_{0};
    uint64_t step_at_ = 0, wait_ = 1, given_ = 0;
    // Samples between two demand signals, the last 64: other programs compress in bursts (about 3 s apart while
    // Tona held the full arena), so a step is judged over their median, not over one sample
    std::array<uint64_t, 64> dgaps_{};
    size_t dgaps_n_ = 0;
    uint64_t demand_gap() const;
    bool step_pending_ = false;
    void mark_volatile(size_t k);
    void log_pressure(bool on, size_t marked);
    void purge_order();  // after the fill: all volatile, the least used marked first, so macOS takes them first
    int64_t free_ram(int64_t* compressed = nullptr) const;  // compressed: bytes the OS compressed since boot

    int64_t budget_ = -1;  // bytes, -1: auto
    std::vector<uint32_t> profile_;
    std::vector<uint32_t> loaded_, use_;  // the profile read at start, the uses counted (TONA_EXPERT_PROFILE_OUT)
    void write_profile() const;
    void write_on_signal();
    id sig_src_[2] = {nil, nil};  // dispatch_source_t of SIGINT and SIGTERM
    Device* dev_ = nullptr;
    uint32_t layers_ = 0, experts_ = 0;
    std::vector<int> fds_;             // per file of the model, F_NOCACHE
    std::vector<ExpertPiece> pieces_;  // [layer][matrix][expert]
    std::vector<int64_t> slot_;        // [layer][expert]: offset of the expert's first piece in the arena, -1: cold
    std::vector<uint64_t> span_;       // [layer]: bytes of one expert's three slots
    std::vector<uint64_t> mslot_;      // [layer][matrix]: bytes of one slot (whole pages plus one for the skew)
    std::vector<std::pair<uint32_t, int>> pins_;  // the pinned experts, most used first
    std::vector<uint64_t> pin_at_;                // their offsets
    uint64_t block_ = 0;                          // most bytes of a block (TONA_METAL_ARENA_BLOCK_MB)
    std::vector<uint64_t> starts_;                // offsets where the blocks before the staging slots begin
    uint64_t cache_at_ = 0, staging_ = 0, staging_bytes_ = 0, size_ = 0;
    std::vector<ArenaBlock> blocks_;  // [0, staging_) in blocks of at most block_ bytes, then the staging slots
    std::atomic<int64_t> held_{0};    // bytes of the live blocks
    // Cache of recently read experts (TONA_METAL_ARENA_LRU slots per layer): an answer's passes route to many of the
    // same experts; at equal memory it saved more reads than more pins (2026-11-15-mac-coder-arena.md). Slot-major,
    // so freeing the last blocks takes the same slots from every layer
    int lru_ = 0;
    std::vector<uint64_t> coff_;   // [layer][slot]: offset of the slot
    std::vector<int> cexp_;        // [layer][slot]: expert in the slot, -1: free, -2: its block is freed
    std::vector<uint64_t> ctick_;  // [layer][slot]: stop of the last use
    std::vector<int16_t> cof_;     // [layer][expert]: its cache slot, -1: none
    uint64_t tick_ = 0;
    bool ticked_ = false;             // prefetch() advanced tick_ for the next stop
    Jobs ahead_;                      // the reads in flight
    std::vector<uint8_t> ahead_set_;  // [expert] of the layer they are for
    uint32_t ahead_layer_ = UINT32_MAX;
    id ahead_group_ = nil;  // dispatch_group_t of the reads
    std::atomic<int> ahead_failed_{0};
    std::map<T, std::pair<uint32_t, int>> where_;  // expert tensor -> layer, matrix (0 gate, 1 up, 2 down)
    id<MTLBuffer> tab_ = nil;
    id rset_ = nil;  // id<MTLResidencySet> (macOS 15) of the blocks and the table
    ArenaStats stats_;

    // The governor (TONA_METAL_ARENA_GOVERN): backend/ram_governor.h on free RAM sampled every 250 ms and the
    // dispatch memory-pressure source; TONA_METAL_ARENA_CHURN=n frees and re-reads blocks every n passes (test)
    std::mutex mu_;  // held during passes and while blocks change
    backend::RamGovernor gov_;
    std::thread mon_;
    std::atomic<bool> stop_{false}, pressure_{false};
    std::atomic<int64_t> free_{0};
    std::atomic<uint64_t> sample_{0};
    std::atomic<double> last_pass_{0};
    uint64_t sample_seen_ = 0;
    std::atomic<uint64_t> passes_{0};  // read by demand() on the monitor thread
    void sample_free();
    static constexpr double WINDOW_S = 10;
    std::array<int64_t, 40> window_{};  // raw free RAM of the last WINDOW_S seconds, every 250 ms
    size_t window_at_ = 0;
    int64_t churn_ = 0, bias_ = 0, calib_ = 0, reserve_ = 0;
    double bias_until_ = 0, bias_s_ = 0, first_pass_ = -1;
    std::atomic<bool> calibrated_{false}, window_reset_{false};
    id pressure_src_ = nil;  // dispatch_source_t
};

}  // namespace tona::metal
