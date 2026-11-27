// Builds the op programs of the megakernel and allocates their buffers. One ProgramBuilder lives for one
// Mega::build(). It is the toolbox shared by all architectures; what is specific to a model (buffers, layer pattern,
// MTP head, drafter) is an ArchProgram (arch_program.h, one directory per architecture under arch/). The programs are
// what the kernel runs: the order of the ops and of the allocations is part of the result (TONA_MEGA_DUMP_PROGRAM
// compares it).
#pragma once

#include <initializer_list>
#include <string>
#include <vector>

#include "gpu/arch_program.h"
#include "gpu/mega.h"
#include "model/qwen35moe.h"

namespace tona::gpu {

// A layer's cache as the steps see it: kv is the VRAM part (or the device pointer of a cache in RAM), cold the pinned
// part from the cell `hot` on (null: not split)
struct KvBuf {
    uint8_t *kv = nullptr, *cold = nullptr;
    int64_t hot = INT64_MAX;
};

class ProgramBuilder {
public:
    // The program being appended to: target, MTP head or drafter
    struct Target {
        std::vector<MegaOp>* ops;
        std::vector<MegaTask>* tasks;
        std::vector<std::string>* labels;
    };
    // One matvec as the builder is asked for it
    struct Mv {
        uint32_t type;
        int64_t in, rows;
        const void* w;
        const float* x;
        const Q8Vec* xq;
        float* y;
    };
    struct MvExtra {
        int slot_e = -1;
        int64_t slot_stride = 0;
        float act_scale = 0;
    };

    explicit ProgramBuilder(Mega& g);
    void run(ArchProgram& arch);

    // --- what an architecture works on
    Mega& mega() { return g_; }
    const MoeShape& shape() const { return h_; }
    const ModelBase& base() const { return m_; }
    // The concrete model of the architecture being built; throws for another architecture
    const Qwen4ExpModel& model() const { return model_as<Qwen4ExpModel>(m_); }
    const Hparams& hparams() const { return model().hp(); }
    const Qwen35MoeModel& ornith() const { return model_as<Qwen35MoeModel>(m_); }
    // Selects the program that op(), add() and the phases append to; wide_mark: whether its dense int8 tasks may use
    // the wide path (the target's do, the drafter's by TONA_WIDE_DRAFT, the MTP head's not)
    void select_program(const Target& t, bool wide_mark);
    const Target& program() const { return tg_; }

    // --- ops, buffers and primitives (mega_build_ops.cpp)
    void mv_begin();
    void mv_end(const char* label) const;  // the label names the step for the timing (TONA_MEGA_PROF)
    void add(const Mv& mv);
    void add(const Mv& mv, const MvExtra& ex);
    void addw(const TensorInfo& t, const void* w, const float* x, const Q8Vec* xq, float* y, float act_scale = 0);
    void op(int kind, std::initializer_list<void*> p, std::initializer_list<int> i = {},
            std::initializer_list<float> f = {}) const;
    void group(const Mv& mv, int64_t slot_bytes, int64_t x_e, int64_t y_e);
    PfW pw(const TensorInfo& t, const void* w) const;

    // --- buffers (mega_build_bufs.cpp)
    uint8_t* alloc_kv(size_t bytes);  // in the place of Mega::kv_host_
    uint8_t* alloc_kv(size_t bytes, bool host);
    // The KV cache of one attention layer with its first `hot` cells in VRAM (kv_place.h): whole in VRAM (hot >=
    // max_ctx), whole in pinned RAM (hot <= 0) or split; registered for the sequence slots and the stash
    KvBuf kv_buffer(int64_t hot);
    void* dev_buf(const void* src, size_t bytes);

private:
    // --- allocation and set-up around the architecture's steps (mega_build_bufs.cpp, mega_build_slots.cpp)
    void alloc_io();
    void init_cache_layout();
    void setup_host_io();
    void setup_prefetch();
    void setup_cpu_mode();
    void alloc_slots();
    void alloc_prefill_arena();
    void alloc_restore_point();
    void size_slots();
    void plan_elastic();
    void grow_elastic();
    void alloc_slots_and_arena();
    size_t slots_chunk_bytes() const;
    void install_slots();
    void install_extra_slots();
    void upload_programs();
    void setup_diagnostics();
    void setup_arrival_stamps();

    Mega& g_;
    const MoeShape& h_;
    const ModelBase& m_;
    const int d_, ff_,
        ne_;  // model width, expert width and experts per token: what the expert caches and host buffers size from
    Target tg_;
    bool prefill_ = true;    // the architecture has prompt kernels (ArchProgram::has_prefill)
    bool wide_mark_ = true;  // tasks of the target model; drafter and experts not (see add)
    int phase0_ = 0;
    size_t max_slot_ = 0;
    size_t vram_before_snap_ = 0;
    size_t snap_bytes_ = 0;
    size_t per_layer_bytes_ = 0;
};

}  // namespace tona::gpu
