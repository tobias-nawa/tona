// The llama.cpp counterpart of tona-logits-dump: the logits of a fixed token sequence from the pinned llama.cpp,
// same output format (engine/tools/tona_logits_dump.cpp), so tools/logits_compare.py can set the two engines side
// by side. A separate program against libllama, outside engine/ because it is not part of the engine build
// (scripts/build-llama-logits-dump.sh).
// Usage: llama-logits-dump MODEL.gguf IDS.txt OUT.bin [--last K] [--mode hip-cpu-experts|cpu|hip] [--threads N]
//   hip-cpu-experts: attention, DeltaNet and the rest on the GPU, the routed experts in RAM (the way a 16 GB card runs
//   a 35B MoE); cpu: everything on the CPU (the baseline whose distance to the GPU modes is llama.cpp's own noise).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

namespace {

std::vector<llama_token> read_ids(const char* path) {
    std::ifstream f(path);
    std::vector<llama_token> ids;
    for (long v; f >> v;) ids.push_back((llama_token)v);
    return ids;
}

struct Options {
    const char *model = nullptr, *ids = nullptr, *out = nullptr, *mode = "hip-cpu-experts";
    int last = 64, threads = 6;
};

bool parse(int argc, char** argv, Options& o) {
    if (argc < 4) return false;
    o.model = argv[1], o.ids = argv[2], o.out = argv[3];
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--last"))
            o.last = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--mode"))
            o.mode = argv[i + 1];
        else if (!std::strcmp(argv[i], "--threads"))
            o.threads = std::atoi(argv[i + 1]);
        else
            return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse(argc, argv, o)) {
        std::fprintf(stderr,
                     "usage: llama-logits-dump MODEL.gguf IDS.txt OUT.bin [--last K] [--mode M] [--threads N]\n");
        return 2;
    }
    const std::vector<llama_token> ids = read_ids(o.ids);
    if (ids.empty()) return 2;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    const llama_model_tensor_buft_override cpu_experts[] = {
        {"\\.ffn_(up|down|gate)_exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    const bool cpu = !std::strcmp(o.mode, "cpu");
    mp.n_gpu_layers = cpu ? 0 : 999;
    if (!std::strcmp(o.mode, "hip-cpu-experts")) mp.tensor_buft_overrides = cpu_experts;
    llama_model* model = llama_model_load_from_file(o.model, mp);
    if (!model) return 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)ids.size() + 16, cp.n_batch = 512, cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = o.threads;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const size_t n = ids.size(), first = n > (size_t)o.last ? n - (size_t)o.last : 0;
    std::ofstream out(o.out, std::ios::binary);
    const uint32_t head[2] = {(uint32_t)vocab, (uint32_t)(n - first)};
    out.write(reinterpret_cast<const char*>(head), sizeof head);
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (size_t i = 0; i < n; i += 512) {
        const int m = (int)std::min<size_t>(512, n - i);
        batch.n_tokens = m;
        for (int j = 0; j < m; j++) {
            batch.token[j] = ids[i + (size_t)j], batch.pos[j] = (llama_pos)(i + (size_t)j);
            batch.n_seq_id[j] = 1, batch.seq_id[j][0] = 0;
            batch.logits[j] = i + (size_t)j >= first;
        }
        if (llama_decode(ctx, batch) != 0) return 1;
        for (int j = 0; j < m; j++) {
            if (i + (size_t)j < first) continue;
            const int32_t idx = (int32_t)(i + (size_t)j);
            out.write(reinterpret_cast<const char*>(&idx), sizeof idx);
            out.write(reinterpret_cast<const char*>(llama_get_logits_ith(ctx, j)), (std::streamsize)vocab * 4);
        }
        std::fprintf(stderr, "%zu/%zu\n", i + (size_t)m, n);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
