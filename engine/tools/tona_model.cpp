// Checks a GGUF against the description of its architecture (every tensor bound and shape-checked) and shows the
// weight plan; for the Coder also the VRAM budget of this machine.
// Usage: tona-model FILE.gguf [--ctx TOKENS]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "model/model_registry.h"
#include "model/qwen4exp.h"

namespace {

double gb(uint64_t b) { return b / 1e9; }

// hipMemGetInfo does not see the desktop's VRAM (research/measurements/2026-10-22-micro.md); sysfs does
uint64_t sysfs_u64(const char* path) {
    std::ifstream f(path);
    uint64_t v = 0;
    f >> v;
    return v;
}

// The dGPU is the card with the most VRAM (the iGPU has 512 MiB); the number cardN can change
std::string dgpu_sysfs() {
    std::string best;
    uint64_t best_total = 0;
    for (const auto& e : std::filesystem::directory_iterator("/sys/class/drm")) {
        const std::string p = e.path().string() + "/device/mem_info_vram_total";
        if (e.path().filename().string().contains('-') || !std::filesystem::exists(p)) continue;
        if (uint64_t t = sysfs_u64(p.c_str()); t > best_total) best_total = t, best = e.path().string() + "/device/";
    }
    return best;
}

}  // namespace

// Weight plan as every architecture reports it
void print_plan(const tona::WeightPlan& w) {
    printf("\nWeights\n");
    printf("  dense            %6.2f GB  (of which hyper-connection %.2f, LM head %.2f)\n", gb(w.dense), gb(w.dense_hc),
           gb(w.dense_lm_head));
    printf("  experts          %6.2f GB  (%.3f MB per expert, %.2f GB per token)\n", gb(w.experts),
           w.expert_bytes / 1e6, gb(w.experts_per_token));
    printf("  embedding        %6.2f GB  (one row per token)\n", gb(w.embedding));
    if (w.ple_table)
        printf("  PLE table        %6.2f GB  (rows of %llu B, several per token)\n", gb(w.ple_table),
               (unsigned long long)w.ple_row_bytes);
    if (w.mtp) printf("  MTP layer        %6.2f GB  (inside the model file)\n", gb(w.mtp));
    printf("  read per token   %6.2f GB  -> ceiling at 315 GB/s: %.1f passes/s if everything is in VRAM\n",
           gb(w.per_token()), 315e9 / w.per_token());
}

// Coder only: the VRAM split of this machine (KV and DeltaNet state of the Coder's layers)
void print_coder_budget(const tona::Qwen4ExpModel& m, uint64_t ctx) {
    const tona::Hparams& h = m.hp();
    const tona::WeightPlan w = m.plan();
    // State: KV int8 (K+V, KV heads x head dim) and raw indexer key int8 per QSA layer, pooled keys per block
    const uint64_t n_qsa = h.n_layer / h.full_attn_interval;
    const uint64_t kv_tok = n_qsa * (2ull * h.n_head_kv * h.head_dim + h.idx_dim + h.idx_dim / h.idx_block);
    const uint64_t dn_state = (uint64_t)(h.n_layer - n_qsa) * h.dn_v_heads * h.dn_head_dim * h.dn_head_dim * 4;
    const std::string dev = dgpu_sysfs();
    if (dev.empty()) throw std::runtime_error("no GPU with mem_info_vram_total in /sys/class/drm");
    const uint64_t total = sysfs_u64((dev + "mem_info_vram_total").c_str());
    const uint64_t used = sysfs_u64((dev + "mem_info_vram_used").c_str());
    const uint64_t reserve = 512ull << 20;  // work buffers, kernel code, headroom for the desktop
    printf("\nVRAM at %llu tokens of context (KV int8)\n", (unsigned long long)ctx);
    printf("  free per sysfs   %6.2f GB  (total %.2f, used %.2f)\n", gb(total - used), gb(total), gb(used));
    printf("  dense            %6.2f GB\n", gb(w.dense));
    printf("  KV + indexer     %6.2f GB  (%llu B per token)\n", gb(kv_tok * ctx), (unsigned long long)kv_tok);
    printf("  DeltaNet state   %6.2f GB\n", gb(dn_state));
    printf("  reserve          %6.2f GB\n", gb(reserve));
    const int64_t left = (int64_t)(total - used) - (int64_t)(w.dense + kv_tok * ctx + dn_state + reserve);
    if (left > 0)
        printf("  -> expert cache  %6.2f GB  = %.0f %% of all experts in VRAM\n", gb(left), 100.0 * left / w.experts);
    else
        printf("  -> does not fit, %.2f GB short\n", gb(-left));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: tona-model FILE.gguf [--ctx TOKENS]\n");
        return 2;
    }
    uint64_t ctx = 65536;
    for (int i = 2; i + 1 < argc; i++)
        if (std::strcmp(argv[i], "--ctx") == 0) ctx = std::strtoull(argv[i + 1], nullptr, 10);
    try {
        const auto m = tona::load_model(argv[1]);
        printf("%s: %s, all shapes checked\n", m->gguf().get_string("general.name").c_str(), m->describe().c_str());
        print_plan(m->plan());
        if (m->arch() == tona::ArchId::Qwen4Exp) print_coder_budget(tona::model_as<tona::Qwen4ExpModel>(*m), ctx);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
