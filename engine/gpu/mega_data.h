// Data members of the Mega class that belong to one topic, split off to keep gpu/mega.h short. Mega inherits them
// privately; the code that uses them is in mega_mtp.cpp, arch/qwen4exp/mtp.cpp and gpu/prefill.hip.
#pragma once

#include <hip/hip_runtime.h>

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/expert_layout.h"
#include "backend/mtp_track.h"
#include "backend/mtp_vocab.h"
#include "gpu/kernel_set.h"
#include "gpu/mega_elastic.h"
#include "gpu/mega_mtp_dyn.h"
#include "gpu/mega_prefill_types.h"
#include "gpu/mega_types.h"
#include "model/gguf.h"
#include "model/qwen4exp.h"

namespace tona::gpu {

// Experts per layer and the profile: use per (layer, expert), counted per token; TONA_EXPERT_PROFILE loads one (initial
// contents of the VRAM cache), TONA_EXPERT_PROFILE_OUT writes the sum of the loaded and the newly counted one at exit.
// Both tables are flat, layer l starts at experts_.offset(l): the layers need not have the same number of experts.
struct MegaExpertData {
    explicit MegaExpertData(backend::ExpertLayout layout) : experts_(std::move(layout)) {}
    backend::ExpertLayout experts_;
    std::vector<uint32_t> use_;
    std::vector<uint32_t> profile_;
    // TONA_VRAM_GOVERNOR=1: slots in blocks of virtual memory that go back to the driver under pressure
    // (mega_elastic.h)
    ElasticPool elastic_;
    // Selection biases of the routers that the CPU reads every pass, copied out of the file mapping (released after the
    // upload, Mega::drop_page_cache); the routing policies point into these
    std::vector<std::unique_ptr<std::vector<float>>> route_bias_;
    int elastic_steps_ = 0, elastic_k_ = 4;  // generations planned, slots per layer in one
};

// MTP head: the third program, its buffers and the truncated vocabulary head
struct MegaMtpData {
    const MtpHead* mtp_ = nullptr;  // the Coder's separate MTP file
    // An MTP program exists: the Coder's file, or a head inside the model file (ModelBase::mtp_source);
    // TONA_NO_MTP=1: not built
    bool mtp_on_ = false;
    std::vector<MegaOp> mops_;
    std::vector<MegaTask> mtasks_;
    std::vector<std::string> mlabels_;
    std::vector<MegaOp> mkops_;  // KV only: start of mops_ up to qsa.prep
    std::vector<std::string> mklabels_;
    void *d_mkops_ = nullptr, *d_mktasks_ = nullptr;
    void *d_mops_ = nullptr, *d_mtasks_ = nullptr;
    MegaParams* mparams_host_ = nullptr;
    MegaParams* d_mparams_ = nullptr;
    float *me_ = nullptr, *mlog_ = nullptr, *mtp_in_host_ = nullptr;
    const float *h_trunk_ = nullptr, *mtp_hout_ = nullptr;
    float* mpick_ = nullptr;
    float* mpick_host_ = nullptr;
    float* mlog_host_ = nullptr;  // mtp_keep_logits
    bool keep_mlog_ = false;
    int32_t* mtp_map_dev_ = nullptr;
    uint8_t* mtp_ex_ = nullptr;
    uint32_t mtp_ex_type_ = 42;
    int64_t mtp_ex_up_ = 0, mtp_ex_down_ = 0, mtp_ex_bytes_ = 0;
    int64_t fw_pos_ = 0;
    int fw_m_ = 0;
    uint8_t* mtp_kv_ = nullptr;
    backend::MtpVocabMap mtp_vocab_;  // row -> token of the truncated head; empty: the full head
    void* d_mprof_ = nullptr;
    std::vector<double> mprof_us_;
    int64_t mprof_n_ = 0;
    uint32_t mtp_head_type_ = 0;
    uint8_t *mtp_head_ = nullptr, *mtp_rows_host_ = nullptr;
    int64_t mtp_rb_ = 0;
    bool mtp_head_q4_ = false;
    MtpDyn mtp_dyn_;  // dynamic rows of the head (TONA_MTP_SCORER)
    float *mtp_pool_ = nullptr, *mtp_hin_ = nullptr, *mtp_hlast_ = nullptr, *mtp_bad_ = nullptr;
    backend::MtpTrack mtrk_{.nb = 1};  // which MTP entries are known, where h is (of the active slot)
    bool trunk_ok_ = false;
};

// Prefill (gpu/prefill.hip): weights and states noted while building the program
struct MegaPrefillData {
    std::vector<std::pair<std::string, float*>> taps_;  // diagnostics: named device buffers (TONA_BOLT_TAP)
    float* mtp_pf_e_host_ = nullptr;                    // pinned: embeddings of a partial chunk
    size_t mtp_pf_e_rows_ = 0;
    int64_t pf_prog_base_ = 0;  // progress: first token of the running group
    std::vector<PfLayer> pf_;
    std::any pf_arch_;  // the architecture's own records of the prefill (residual weights, norm gains)
    PfW pf_out_, pf_ple_key_, pf_ple_val_;
    const float *pf_ple_nk_ = nullptr, *pf_ple_nq_ = nullptr, *pf_ple_nc_ = nullptr, *pf_ple_conv_ = nullptr;
    float* pf_ple_hist_ = nullptr;
    int64_t kv_row_ = 0, sstride_ = 0;
    int dswa_ = 0;
    int pf_chunk_ = 0;
    uint8_t* pf_arena_ = nullptr;
    size_t pf_arena_bytes_ = 0;
    int pf_pass_ = 1 << 30;   // target passes since the last prefill (stats early_*)
    bool extras_on_ = false;  // extra slots in the workspace (mega_extra.h)
    float* pf_host_ =
        nullptr;  // pinned, twice: embedding and PLE vector per token of the chunk [2][max(chunk, lchunk)][2*d]
    int32_t* pf_hostc_ = nullptr;  // pinned: counts and offsets of the experts per layer
    hipStream_t pf_cs_ = nullptr;  // copy stream: pinned experts into VRAM
    hipEvent_t pf_ev_[4] = {};     // per buffer: free (0, 1), filled (2, 3)
    int pf_last_chunk_ = 0, pf_last_group_ = 0;
    int pf_lchunk_ = 0;              // chunk size of the layered prefill (TONA_PF_LCHUNK)
    uint8_t* slots_base_ = nullptr;  // all slots in one allocation, layer by layer
    size_t slots_total_ = 0;
    std::vector<size_t> slot_off_;  // start of the slots per layer in slots_base_
    std::vector<PfLoan> pf_loans_;
    hipEvent_t pf_lev_[4] = {};  // layer buffers: free (0, 1), filled (2, 3)
};

// Layer Q6_K matrices tiled on the way to VRAM (TONA_Q6_K_TILE, mega_alloc.cpp): the engine's view of their type is
// 1019
constexpr uint32_t kTaskTile = 1019;

struct MegaTileData {
    const KernelSet* kernels_ = nullptr;  // the megakernel of the model's architecture (kernels_of, set first thing)
    MailboxRows mrows_;                   // router and prediction rows of the mailbox in the model's size class
    std::unordered_set<const void*> tiled_;
    // Device task arrays with a tile task: only their launches use the kernel that has the tile loops
    std::unordered_set<const void*> tile_tasks_;
    uint32_t weight_type(const TensorInfo& t, const void* dev_w) const {
        return tiled_.contains(dev_w) ? kTaskTile : t.type;
    }
};

}  // namespace tona::gpu
