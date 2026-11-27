// Dumps the logits of a model for a fixed token sequence, teacher-forced, so another engine's logits on the same ids
// can be compared with them (tools/logits_compare.py). Works for every architecture with a backend or an oracle. Usage:
// tona-logits-dump MODEL.gguf IDS.txt OUT.bin [--last K] [--width W] [--ctx N] [--oracle] [--max N] [--prefill]
//   IDS.txt: token ids separated by white space (tona-tokenize prints them one per line). The ids are fed in passes of
//   W tokens (default 1, at most 8); the logits of the last K positions (default 64) are written. --oracle runs the CPU
//   oracle instead of the backend; --max N cuts the sequence to its first N ids; --prefill runs the ids before the last
//   K through the backend's prefill (one call) and only the last K in passes, so those rows are the decode after a
//   kernel-prefilled prompt.
// OUT.bin: uint32 vocab, uint32 count, then per position int32 index and vocab float32 (the logits that predict the id
//   at index + 1).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/builtin.h"
#include "backend/registry.h"
#include "cpu/ref.h"
#include "model/model_registry.h"
#include "tona_args.h"

namespace {

struct Options {
    std::string model, ids, out;
    int last = 64, width = 1, max_ids = 1 << 30;
    int64_t ctx = 0;
    bool oracle = false, prefill = false;
};

Options parse(int argc, char** argv) {
    if (argc < 4) throw std::invalid_argument("usage: tona-logits-dump MODEL.gguf IDS.txt OUT.bin [options]");
    Options o{argv[1], argv[2], argv[3]};
    for (int i = 4; i < argc; i++) {
        const bool more = i + 1 < argc;
        if (!std::strcmp(argv[i], "--oracle"))
            o.oracle = true;
        else if (!std::strcmp(argv[i], "--prefill"))
            o.prefill = true;
        else if (!std::strcmp(argv[i], "--last") && more)
            o.last = tona::tools::parse_int(argv[++i]);
        else if (!std::strcmp(argv[i], "--width") && more)
            o.width = tona::tools::parse_int(argv[++i]);
        else if (!std::strcmp(argv[i], "--max") && more)
            o.max_ids = tona::tools::parse_int(argv[++i]);
        else if (!std::strcmp(argv[i], "--ctx") && more)
            o.ctx = tona::tools::parse_int64(argv[++i]);
        else
            throw std::invalid_argument(std::string("unknown option: ") + argv[i]);
    }
    if (o.width < 1 || o.width > 8) throw std::invalid_argument("--width: 1 to 8");
    return o;
}

std::vector<int32_t> read_ids(const std::string& path, int max_ids) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(path + ": cannot read");
    std::vector<int32_t> ids;
    for (long v; f >> v && (int)ids.size() < max_ids;) ids.push_back((int32_t)v);
    return ids;
}

// The rows of the last K positions, as they come
class Dump {
public:
    Dump(const std::string& path, size_t vocab, size_t n, int last)
        : vocab_(vocab), first_(n > (size_t)last ? n - last : 0) {
        out_.open(path, std::ios::binary);
        if (!out_) throw std::runtime_error(path + ": cannot write");
        const uint32_t head[2] = {(uint32_t)vocab, (uint32_t)(n - first_)};
        out_.write(reinterpret_cast<const char*>(head), sizeof head);
    }
    void row(size_t index, const float* logits) {
        if (index < first_) return;
        const int32_t i = (int32_t)index;
        out_.write(reinterpret_cast<const char*>(&i), sizeof i);
        out_.write(reinterpret_cast<const char*>(logits), (std::streamsize)(vocab_ * sizeof(float)));
    }

private:
    std::ofstream out_;
    size_t vocab_, first_;
};

void run_oracle(const tona::ModelBase& model, const std::vector<int32_t>& ids, Dump& dump) {
    const auto fw = tona::ref::make_oracle(model);
    for (size_t i = 0; i < ids.size(); i++) {
        const std::vector<float> lg = fw->step(ids[i]);
        dump.row(i, lg.data());
        if ((i + 1) % 16 == 0) std::fprintf(stderr, "%zu/%zu\n", i + 1, ids.size());
    }
}

void run_backend(const tona::ModelBase& model, const Options& o, const std::vector<int32_t>& ids, Dump& dump) {
    tona::backend::register_builtin();
    tona::backend::CreateOptions co;
    co.max_ctx = o.ctx > 0 ? o.ctx : (int64_t)ids.size() + 16;
    const auto be = tona::backend::registry::create("auto", model, co);
    std::fprintf(stderr, "backend %s, %zu ids, passes of %d\n", be->info().name.c_str(), ids.size(), o.width);
    const size_t vocab = model.moe_shape().vocab;
    size_t start = 0;
    if (o.prefill) {
        start = ids.size() > (size_t)o.last ? ids.size() - (size_t)o.last : 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (start > 0) be->prefill(ids.data(), (int)start, nullptr);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "prefill: %zu tokens in %.2f s (%.0f tokens/s)\n", start, sec, (double)start / sec);
    }
    for (size_t i = start; i < ids.size(); i += (size_t)o.width) {
        const int nb = (int)std::min<size_t>((size_t)o.width, ids.size() - i);
        const float* lg = be->forward(&ids[i], nb);
        for (int j = 0; j < nb; j++) dump.row(i + (size_t)j, lg + (size_t)j * vocab);
        be->commit(nb);
        if ((i / o.width + 1) % 256 == 0) std::fprintf(stderr, "%zu/%zu\n", i + (size_t)nb, ids.size());
    }
}

int run(int argc, char** argv) {
    const Options o = parse(argc, argv);
    const std::unique_ptr<tona::ModelBase> model = tona::load_model(o.model);
    const std::vector<int32_t> ids = read_ids(o.ids, o.max_ids);
    if (ids.empty()) throw std::runtime_error("no ids");
    Dump dump(o.out, model->moe_shape().vocab, ids.size(), o.last);
    if (o.oracle)
        run_oracle(*model, ids, dump);
    else
        run_backend(*model, o, ids, dump);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
