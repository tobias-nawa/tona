// Upload of an op program: the matvec tasks in the form the device wants (units per phase, lanes per row), x of a
// phase staged in LDS, and the dependency tables that replace the barrier after dn.in.
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>

#include "backend/env.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega.h"
#include "gpu/mega_dump.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

// Dependencies instead of barriers: the consumer waits only for the blocks that computed the rows it reads
struct DepTables {
    std::vector<uint64_t> masks;  // [BMAX][block]: blocks whose data the consumer's block reads
    std::vector<int> flat;        // [BMAX][stride]: per width the units of dn.conv per wave
    int stride = 0;
};

// Which blocks compute a row range of the producer, for one width of the pass. Wave w runs the units g = w (mod W)
// of the flat sequence, so a span of a few units lies on a few blocks.
struct DepPlan {
    int blocks;
    int64_t off, waves, base;
    int per;  // rows per unit
    uint64_t all;

    uint64_t producers(int64_t r0, int64_t r1) const {
        const int64_t u0 = base + r0 / per, u1 = base + (r1 - 1) / per + 1;
        if (u1 - u0 >= waves) return all;
        uint64_t m = 0;
        for (int64_t g = u0; g < u1; g++) m |= 1ull << ((g % waves) / MWAVES + off);
        return m;
    }
};

DepPlan dep_plan(const DevTask& D, const MegaOp& prod, int blocks, bool wide) {
    DepPlan P{};
    P.blocks = blocks;
    P.off = prod.i[7] == 1 ? 1 : 0;
    P.waves = (blocks - P.off) * MWAVES;
    // wide passes: wide_loop runs 16 rows per unit and counts the units of the phase from 0 (qkv is the first task)
    P.per = wide ? 16 : WAVE / D.L;
    P.base = wide ? 0 : D.base;
    P.all = blocks == 64 ? ~0ull : (1ull << blocks) - 1;
    return P;
}

// Units of dn.conv like op_dn_conv: (q/k head, token), then (32 v channels, token). Each goes to the least loaded
// block among those that compute its rows, there round robin to the waves.
void assign_conv_units(const DepPlan& P, const MegaOp& conv, int nb, uint64_t* masks,
                       std::vector<std::vector<int>>& per_blk) {
    const int64_t kd = conv.i[0], vd = conv.i[1], nk = conv.i[4];
    const int64_t nqk = 2 * nk * nb, n = nqk + vd / 32 * nb;
    for (int64_t g = 0; g < n; g++) {
        const bool qk = g < nqk;
        const int64_t u = qk ? g / nb : 2 * nk + (g - nqk) / nb;
        const int64_t r0 = qk ? u * DS : 2 * kd + (u - 2 * nk) * 32, r1 = r0 + (qk ? DS : 32);
        const uint64_t m = P.producers(r0, r1);
        int best = -1;
        for (int b = 0; b < P.blocks; b++)
            if (m >> b & 1 && (best < 0 || per_blk[b].size() < per_blk[best].size())) best = b;
        per_blk[best].push_back((int)g);
        masks[best] |= m;
    }
}

// [waves + 1 offsets][units of wave 0][units of wave 1]...
std::vector<int> wave_table(const std::vector<std::vector<int>>& per_blk, int blocks) {
    const int64_t waves = (int64_t)blocks * MWAVES;
    std::vector<std::vector<int>> per_w(waves);
    for (int b = 0; b < blocks; b++)
        for (size_t k = 0; k < per_blk[b].size(); k++) per_w[b * MWAVES + k % MWAVES].push_back(per_blk[b][k]);
    std::vector<int> table(waves + 1, 0);
    for (int64_t w = 0; w < waves; w++) table[w + 1] = table[w] + (int)per_w[w].size();
    for (int64_t w = 0; w < waves; w++) table.insert(table.end(), per_w[w].begin(), per_w[w].end());
    return table;
}

// Lanes of a wave that work on one row
int lanes_of(const MegaTask& mt, bool q8) {
    // Expert groups with 8 lanes per row (4 rows per wave): gate_up 3.55 -> 3.31 ms, down 2.32 -> 2.21 ms against the
    // rule (16 resp. 4). Dense matrices: 32/16 equal, 8 slower. TONA_MEGA_LANES_EXP to measure.
    static const int lanes_exp = (int)env::integer("TONA_MEGA_LANES_EXP", 8);
    static const int lanes_max = (int)env::integer("TONA_MEGA_LANES_MAX", 32);
    if (!q8) return WAVE;
    return mt.n_exp > 1 && lanes_exp > 0 ? lanes_exp : std::min(lanes_max, lanes_per_row(mt.t.in));
}

}  // namespace

class ProgramUploader {
public:
    ProgramUploader(Mega& g, std::vector<MegaOp>& ops, const std::vector<MegaTask>& tasks)
        : g_(g), ops_(ops), tasks_(tasks), dev_(tasks.size()), target_(&ops == &g.ops_) {}
    void run(void*& d_ops, void*& d_tasks);

private:
    void lay_out_phase(MegaOp& o);
    struct SharedX {
        const int8_t* q = nullptr;
        const Q8Meta* meta = nullptr;
        int64_t stride = 0;
        bool uniform = true;
    };
    SharedX shared_x(const MegaOp& o) const;
    void stage_x(MegaOp& o) const;
    DevTask dev_task(const MegaTask& mt, int64_t b0) const;
    void install_deps();
    void install_dn_deps(size_t o);
    void* upload_table(const void* data, size_t bytes);
    bool dn_conv_deps(const DevTask& D, const MegaOp& prod, const MegaOp& conv, DepTables& out) const;
    void dump() const;

    Mega& g_;
    std::vector<MegaOp>& ops_;
    const std::vector<MegaTask>& tasks_;
    std::vector<DevTask> dev_;
    const bool target_;
};

void Mega::upload_prog(std::vector<MegaOp>& ops, const std::vector<MegaTask>& tasks, void*& d_ops, void*& d_tasks) {
    ProgramUploader(*this, ops, tasks).run(d_ops, d_tasks);
}

void ProgramUploader::run(void*& d_ops, void*& d_tasks) {
    for (MegaOp& o : ops_)
        if (o.kind == OP_MATVEC) lay_out_phase(o);
    // TONA_DEPS=0: off (research/measurements/2026-10-25-barriers)
    static const bool deps_on = env::integer("TONA_DEPS", 1) != 0;
    if (deps_on && target_) install_deps();
    HIPCHECK(trace::dev_malloc(&d_tasks, dev_.size() * sizeof(DevTask) + 64));
    HIPCHECK(hipMemcpy(d_tasks, dev_.data(), dev_.size() * sizeof(DevTask), hipMemcpyHostToDevice));
    // Only a program with tile tasks launches the kernel that has the tile loops (Q6Layout)
    if (std::any_of(dev_.begin(), dev_.end(), [](const DevTask& t) { return t.type == kTaskTile; }))
        g_.tile_tasks_.insert(d_tasks);
    HIPCHECK(trace::dev_malloc(&d_ops, ops_.size() * sizeof(MegaOp)));
    HIPCHECK(hipMemcpy(d_ops, ops_.data(), ops_.size() * sizeof(MegaOp), hipMemcpyHostToDevice));
    dump();
}

void ProgramUploader::dump() const {
    if (!g_.dump_) return;
    const bool mtp = &ops_ == &g_.mops_, mk = &ops_ == &g_.mkops_;
    const char* name = target_ ? "ops" : mtp ? "mops" : mk ? "mkops" : "dops";
    g_.dump_->ops(name, ops_, target_ ? g_.labels_ : mtp ? g_.mlabels_ : mk ? g_.mklabels_ : g_.dlabels_);
    g_.dump_->dev_tasks(name, dev_);
}

// x of a phase in LDS if all its int8 tasks read the same x and it fits for PNB tokens
ProgramUploader::SharedX ProgramUploader::shared_x(const MegaOp& o) const {
    SharedX x;
    for (int i = o.a; i < o.b; i++) {
        const MegaTask& mt = tasks_[i];
        if (!mt.t.xq.q) continue;
        if (!x.q) x.q = mt.t.xq.q, x.meta = mt.t.xq.meta, x.stride = mt.xb;
        x.uniform = x.uniform && mt.t.xq.q == x.q && mt.t.xq.meta == x.meta && mt.xb == x.stride && mt.x_e == 0;
    }
    return x;
}

void ProgramUploader::stage_x(MegaOp& o) const {
    const SharedX x = shared_x(o);
    const int64_t sxs = x.stride;
    // At 4 tokens the x vectors did not fit next to the weight stream in L0; the same address for all tokens was 6.6 ms
    // per pass faster (2026-10-23-verify-pass)
    if (x.q && x.uniform && sxs > 0 && sxs % 32 == 0 &&
        PNB * (sxs + sxs / 32 * (int64_t)sizeof(Q8Meta)) <= XSTAGE_BYTES && !env::flag("TONA_MEGA_NO_STAGE"))
        o.i[0] = (int32_t)sxs, o.p[0] = (void*)x.q, o.p[1] = (void*)x.meta;
    if (o.i[6] > 0 && (o.i[0] == 0 || o.b - o.a != 1 || tasks_[o.a].t.type != 8 ||
                       MEGA_BMAX * (o.i[0] + o.i[0] / 32 * (int64_t)sizeof(Q8Meta)) > XSTAGE_BYTES))
        throw std::runtime_error("gpu::Mega: quantizing on load needs x in LDS and a Q8_0 task (TONA_FUSE_QUANT=0)");
}

DevTask ProgramUploader::dev_task(const MegaTask& mt, int64_t b0) const {
    const MatvecTask& t = mt.t;
    if (!matvec_supported(t.type) || t.in % 32)
        throw std::runtime_error(std::format("gpu::Mega: type {} with row length {}", t.type, t.in));
    // The row pointers of the tile type carry the row within the tile in the low 4 bits (matvec_q8_tile.h)
    if (t.type == 1019 && (reinterpret_cast<uintptr_t>(t.w) & 15))
        throw std::runtime_error("gpu::Mega: TONA_Q6_K_T matrix is not 16-byte aligned");
    const bool q8 = t.xq.q != nullptr;
    const int L = lanes_of(mt, q8);
    // float path with few rows: one unit per token (TONA_MEGA_TSPLIT = row limit, 0 = off)
    static const int tsplit_max = (int)env::integer("TONA_MEGA_TSPLIT", 64);
    const bool tsplit = !q8 && mt.n_exp <= 1 && t.rows <= tsplit_max;
    const int64_t upm = q8 ? (t.rows + WAVE / L - 1) / (WAVE / L) : t.rows * (tsplit ? MEGA_BMAX : 1);
    const int64_t units = upm * mt.n_exp;
    DevTask D = {static_cast<const uint8_t*>(t.w),
                 t.x,
                 t.y,
                 q8 ? t.xq.q : nullptr,
                 q8 ? t.xq.meta : nullptr,
                 t.rows,
                 units,
                 row_bytes_of(t.type, t.in),
                 b0,
                 t.type,
                 (int32_t)t.in,
                 L};
    D.slot_e = mt.slot_e, D.slot_stride = mt.slot_stride, D.y_pos_stride = mt.y_pos_stride;
    D.act_scale = mt.act_scale;
    D.n_exp = mt.n_exp, D.upe = (int32_t)upm, D.x_e = mt.x_e, D.y_e = mt.y_e;
    D.xb = mt.xb, D.yb = mt.yb;
    D.tsplit = tsplit ? MEGA_BMAX : 0;
    D.wide = mt.wide && q8 && mt.n_exp <= 1 && mt.slot_e < 0;
    return D;
}

void ProgramUploader::lay_out_phase(MegaOp& o) {
    stage_x(o);
    static const bool log_tasks = env::text("TONA_MEGA_TASKS") != nullptr;
    int64_t b0 = 0;
    for (int i = o.a; i < o.b; i++) {
        const DevTask D = dev_task(tasks_[i], b0);
        dev_[i] = D;
        if (log_tasks && target_)
            std::fprintf(stderr, "Task %d (step %d): type %u, %lld x %lld, L %d, %lld units, q8 %d\n", i,
                         (int)(&o - ops_.data()), D.type, (long long)D.rows, (long long)D.in, D.L, (long long)D.units,
                         (int)(D.xq != nullptr));
        b0 += D.units;
    }
}

// Tables for one consumer of the qkv matvec: dn.conv per width of the pass
bool ProgramUploader::dn_conv_deps(const DevTask& D, const MegaOp& prod, const MegaOp& conv, DepTables& out) const {
    out.masks.assign(MEGA_BMAX * 64, 0);
    std::vector<int> tab[MEGA_BMAX];
    for (int nb = 1; nb <= MEGA_BMAX; nb++) {
        const int blocks = g_.kblocks_[Mega::kix(nb)];
        if (blocks > 64) return false;
        const DepPlan P = dep_plan(D, prod, blocks, D.wide && g_.wide_min_ > 0 && nb >= g_.wide_min_);
        std::vector<std::vector<int>> per_blk(blocks);
        assign_conv_units(P, conv, nb, out.masks.data() + (nb - 1) * 64, per_blk);
        tab[nb - 1] = wave_table(per_blk, blocks);
    }
    for (const auto& t : tab) out.stride = std::max(out.stride, (int)t.size());
    out.flat.assign((size_t)out.stride * MEGA_BMAX, 0);
    for (int k = 0; k < MEGA_BMAX; k++) std::copy(tab[k].begin(), tab[k].end(), out.flat.begin() + k * out.stride);
    return true;
}

// Step o gets nobar = 2, the next one (dn.conv) in p[13] the blocks it reads from per width and own block. So far:
// dn.in -> dn.conv.
void ProgramUploader::install_deps() {
    for (size_t o = 0; o + 1 < ops_.size(); o++) install_dn_deps(o);
}

// Upload of a dependency table (copied to the device, freed with the other allocations)
void* ProgramUploader::upload_table(const void* data, size_t bytes) {
    void* d = nullptr;
    HIPCHECK(trace::dev_malloc(&d, bytes));
    HIPCHECK(hipMemcpy(d, data, bytes, hipMemcpyHostToDevice));
    g_.allocs_.push_back(d);
    return d;
}

void ProgramUploader::install_dn_deps(size_t o) {
    MegaOp& prod = ops_[o];
    MegaOp& conv = ops_[o + 1];
    if (prod.kind != OP_MATVEC || conv.kind != OP_DN_CONV || prod.nobar || conv.p[12] || conv.p[13] || conv.i[5])
        return;
    const DevTask& D = dev_[prod.a];  // qkv
    DepTables t;
    if (!D.xq || D.n_exp > 1 || D.tsplit || !dn_conv_deps(D, prod, conv, t)) return;
    conv.p[13] = upload_table(t.masks.data(), t.masks.size() * sizeof(uint64_t));
    conv.p[12] = upload_table(t.flat.data(), t.flat.size() * sizeof(int)), conv.i[5] = t.stride;
    prod.nobar = 2;
    if (g_.dump_) {
        g_.dump_->table("deps.masks", t.masks.data(), t.masks.size() * sizeof(uint64_t));
        g_.dump_->table("deps.tab", t.flat.data(), t.flat.size() * sizeof(int));
    }
    static const bool log = env::text("TONA_DEPS_DUMP") != nullptr;
    if (!log) return;
    std::string line;
    for (int b = 0; b < g_.kblocks_[1]; b++) line += std::format(" {}", std::popcount(t.masks[3 * 64 + b]));
    std::fprintf(stderr, "Dependency after step %zu (%s), B = 4, blocks per block:%s\n", o, g_.labels_[o].c_str(),
                 line.c_str());
}

}  // namespace tona::gpu
