// N2c for Bolt: every block of the GPU program against the CPU oracle. The GPU program copies the residual stream after
// each of the 88 blocks into a buffer of its own (TONA_BOLT_TAP, set by this tool); the oracle's trace holds the same
// residual per block. Usage: tona-bolt-layers MODEL.gguf IDS.txt [--n N] [--width W]
//   The first N ids (default 8) go through the backend in passes of W tokens (1 to 8, default 1) and one by one through
//   the oracle; per position and block the deviation max|gpu - oracle| / max|oracle| is printed, per block kind the
//   worst of them, and the exit status is 1 if a block is above --tol (default 0.05).
// What the number means: the backend quantizes activations to int8 and the oracle does not, so the deviation grows with
// depth by that rounding alone; a misreading shows as a jump at one block (the first block of a kind that is wrong), a
// tensor scale applied twice as a deviation of |1 - scale| in the Mamba-2 blocks.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/builtin.h"
#include "backend/registry.h"
#include "cpu/ref.h"
#include "gpu/mega.h"
#include "model/model_registry.h"
#include "model/nemotron_h_moe.h"
#include "tona_args.h"

namespace {

struct Options {
    std::string model, ids;
    int n = 8, width = 1;
    double tol = 0.05;
};

Options parse(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("usage: tona-bolt-layers MODEL.gguf IDS.txt [--n N] [--width W]");
    Options o{argv[1], argv[2]};
    for (int i = 3; i < argc; i++) {
        const bool more = i + 1 < argc;
        if (!std::strcmp(argv[i], "--n") && more)
            o.n = tona::tools::parse_int(argv[++i]);
        else if (!std::strcmp(argv[i], "--width") && more)
            o.width = tona::tools::parse_int(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && more)
            o.tol = tona::tools::parse_double(argv[++i]);
        else
            throw std::invalid_argument(std::string("unknown option: ") + argv[i]);
    }
    if (o.width < 1 || o.width > 8) throw std::invalid_argument("--width: 1 to 8");
    return o;
}

std::vector<int32_t> read_ids(const std::string& path, int n) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(path + ": cannot read");
    std::vector<int32_t> ids;
    for (long v; f >> v && (int)ids.size() < n;) ids.push_back((int32_t)v);
    return ids;
}

const char* kind_name(tona::BoltBlockKind k) {
    return k == tona::BoltBlockKind::Mamba2 ? "mamba2" : k == tona::BoltBlockKind::Attention ? "attention" : "moe";
}

// The residual of each block after the oracle's step of each token: [token][block][d]
std::vector<std::vector<std::vector<float>>> oracle_trace(const tona::ModelBase& model,
                                                          const std::vector<int32_t>& ids) {
    const auto fw = tona::ref::make_oracle(model);
    std::vector<std::vector<std::vector<float>>> all;
    for (int32_t id : ids) {
        tona::ref::Trace t;
        fw->step(id, &t);
        all.push_back(std::move(t.residual));
    }
    return all;
}

double deviation(const std::vector<float>& got, const std::vector<float>& want) {
    double top = 0, worst = 0;
    for (float w : want) top = std::max(top, (double)std::fabs(w));
    for (size_t i = 0; i < want.size(); i++) worst = std::max(worst, (double)std::fabs(got[i] - want[i]));
    return worst / (top + 1e-6);
}

int run(int argc, char** argv) {
    const Options o = parse(argc, argv);
    setenv("TONA_BOLT_TAP", "1", 1);
    const std::unique_ptr<tona::ModelBase> model = tona::load_model(o.model);
    const tona::NemotronHMoeModel& bolt = tona::model_as<tona::NemotronHMoeModel>(*model);
    const std::vector<int32_t> ids = read_ids(o.ids, o.n);
    if (ids.empty()) throw std::runtime_error("no ids");
    const auto want = oracle_trace(*model, ids);
    std::fprintf(stderr, "oracle done: %zu tokens\n", ids.size());

    tona::backend::register_builtin();
    tona::backend::CreateOptions co;
    co.max_ctx = (int64_t)ids.size() + 16;
    const auto be = tona::backend::registry::create("auto", *model, co);
    const auto* mega = dynamic_cast<tona::gpu::Mega*>(be.get());
    if (!mega || mega->taps().size() != bolt.hp().n_layer) throw std::runtime_error("the backend has no block taps");
    const size_t d = bolt.hp().d;
    std::map<std::string, double> worst_kind;
    std::vector<double> worst_block(bolt.hp().n_layer, 0.0);
    for (size_t i = 0; i < ids.size(); i += (size_t)o.width) {
        const int nb = (int)std::min<size_t>((size_t)o.width, ids.size() - i);
        be->forward(&ids[i], nb);
        for (size_t b = 0; b < mega->taps().size(); b++) {
            std::vector<float> host((size_t)nb * d);
            if (hipMemcpy(host.data(), mega->taps()[b].second, host.size() * sizeof(float), hipMemcpyDeviceToHost) !=
                hipSuccess)
                throw std::runtime_error("tap copy failed");
            for (int j = 0; j < nb; j++) {
                const std::vector<float> got(host.begin() + (size_t)j * d, host.begin() + (size_t)(j + 1) * d);
                const double dev = deviation(got, want[i + (size_t)j][b]);
                worst_block[b] = std::max(worst_block[b], dev);
                auto& w = worst_kind[kind_name(bolt.layers()[b].kind)];
                w = std::max(w, dev);
            }
        }
        be->commit(nb);
    }
    int bad = 0;
    for (size_t b = 0; b < worst_block.size(); b++) {
        const bool over = worst_block[b] > o.tol;
        bad += over;
        std::printf("block %2zu %-9s worst deviation %.5f%s\n", b, kind_name(bolt.layers()[b].kind), worst_block[b],
                    over ? "  ABOVE TOLERANCE" : "");
    }
    for (const auto& [k, v] : worst_kind) std::printf("kind %-9s worst deviation %.5f\n", k.c_str(), v);
    const tona::Mamba2Weights& m0 = bolt.layers()[0].ssm;
    const auto scale = [](const tona::TensorInfo* t) { return t ? *reinterpret_cast<const float*>(t->data) : 1.0f; };
    std::printf("block 0 tensor scales: in %.6f out %.6f (a test of the scale handling needs them different from 1)\n",
                scale(m0.in_scale), scale(m0.out_scale));
    std::printf("%d of %zu blocks above tolerance %.3f\n", bad, worst_block.size(), o.tol);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
