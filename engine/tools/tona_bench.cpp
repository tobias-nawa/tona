// Speed of a backend through the Backend interface alone, without server, sampling or tokenizer: per round a reset,
// a prefill of P tokens (the backend's own prefill), then D decoded tokens in passes of width B, each pass committing
// all B (the time of a pass whose B tokens are all accepted). Tokens are the argmax of the previous pass, so the
// inputs follow the model as in greedy decoding. One warmup round first (kernel pipelines, first touch of the weights)
// is printed but not counted. Prints every round and the median and range over the rounds. With --mtp the MTP head
// runs after every pass as in the server (a fill with the pass's last argmax, then K chained steps), timed apart from
// the pass: the costs the width choice of speculative decoding needs.
// Usage: tona-bench MODEL.gguf [--backend NAME|auto] [-p PREFILL] [-d DECODE] [-b WIDTH] [-r ROUNDS] [--ctx N]
//        [--mtp MTP.gguf] [-k STEPS]
#ifndef _WIN32
#include <sys/resource.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/builtin.h"
#include "backend/registry.h"
#include "model/model_registry.h"
#include "model/qwen4exp.h"

namespace {

struct Options {
    std::string model, backend = "auto", mtp;
    int prefill = 512, decode = 64, width = 1, rounds = 3, steps = 3;
    int64_t ctx = 4096;
};

Options parse(int argc, char** argv) {
    if (argc < 2) throw std::invalid_argument("usage: tona-bench MODEL.gguf [-p N] [-d N] [-b N] [-r N] [--ctx N]");
    Options o;
    o.model = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--backend")
            o.backend = v;
        else if (a == "-p")
            o.prefill = std::stoi(v);
        else if (a == "-d")
            o.decode = std::stoi(v);
        else if (a == "-b")
            o.width = std::stoi(v);
        else if (a == "-r")
            o.rounds = std::stoi(v);
        else if (a == "--ctx")
            o.ctx = std::stoll(v);
        else if (a == "--mtp")
            o.mtp = v;
        else if (a == "-k")
            o.steps = std::stoi(v);
        else
            throw std::invalid_argument("unknown option " + a);
    }
    return o;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int32_t argmax(const float* v, size_t n) { return (int32_t)(std::max_element(v, v + n) - v); }

struct Round {
    double prefill_tps = 0, decode_tps = 0, pass_ms = 0, fill_ms = 0, step_ms = 0;
};

// The head after a pass: a fill with the next token, then o.steps chained steps on its own picks; adds the times
void run_head(tona::backend::Backend& be, const Options& o, int32_t next, Round& r) {
    double t0 = now_s();
    tona::backend::MtpPick pk = be.mtp_fill(next);
    r.fill_ms += (now_s() - t0) * 1e3;
    t0 = now_s();
    for (int k = 0; k < o.steps && pk.token >= 0; k++) pk = be.mtp_step(pk.token);
    r.step_ms += (now_s() - t0) * 1e3;
}

// Prompt tokens: a fixed spread over the vocabulary, the same in every round
Round one_round(tona::backend::Backend& be, const Options& o, size_t vocab) {
    be.reset();
    std::vector<int32_t> prompt(o.prefill);
    for (int i = 0; i < o.prefill; i++) prompt[i] = (int32_t)((uint64_t)(i + 1) * 2654435761u % vocab);
    Round r;
    double t0 = now_s();
    be.prefill(prompt.data(), o.prefill, nullptr);
    r.prefill_tps = o.prefill / (now_s() - t0);
    std::vector<int32_t> tok(o.width, prompt.back());
    const int passes = o.decode / o.width;
    double t_pass = 0;
    t0 = now_s();
    for (int i = 0; i < passes; i++) {
        const double tp = now_s();
        const float* lg = be.forward(tok.data(), o.width);
        for (int b = 0; b < o.width; b++) tok[b] = argmax(lg + (size_t)b * vocab, vocab);
        be.commit(o.width);
        t_pass += now_s() - tp;
        if (!o.mtp.empty()) run_head(be, o, tok[o.width - 1], r);
    }
    r.decode_tps = passes * o.width / (now_s() - t0);
    r.pass_ms = t_pass * 1e3 / passes, r.fill_ms /= passes, r.step_ms /= passes * std::max(1, o.steps);
    return r;
}

void summary(const char* what, std::vector<double> v, const char* unit = "tok/s") {
    std::sort(v.begin(), v.end());
    std::printf("%s: median %.2f %s, range %.2f ... %.2f (%zu rounds)\n", what, v[v.size() / 2], unit, v.front(),
                v.back(), v.size());
}

int run(int argc, char** argv) {
    const Options o = parse(argc, argv);
    if (o.width < 1 || o.decode < o.width || o.prefill < 1) throw std::invalid_argument("need 1 <= -b <= -d, -p >= 1");
    tona::backend::register_builtin();
    const auto model_ptr = tona::load_model_as<tona::Qwen4ExpModel>(o.model);
    const tona::Qwen4ExpModel& model = *model_ptr;
    tona::backend::CreateOptions co;
    co.max_ctx = o.ctx;
    std::unique_ptr<tona::MtpHead> head;
    if (!o.mtp.empty()) head = std::make_unique<tona::MtpHead>(o.mtp, model.hp()), co.mtp = head.get();
    auto be = tona::backend::registry::create(o.backend, model, co);
    const size_t vocab = model.hp().vocab;
    std::printf("Backend %s, prefill %d, decode %d in passes of %d, %d rounds\n", be->info().name.c_str(), o.prefill,
                o.decode, o.width, o.rounds);
    std::vector<double> pre, dec, pms, fms, sms;
    const Round w = one_round(*be, o, vocab);
    std::printf("  warmup: prefill %.2f tok/s, decode %.2f tok/s (not counted)\n", w.prefill_tps, w.decode_tps);
    for (int i = 0; i < o.rounds; i++) {
        const Round r = one_round(*be, o, vocab);
        std::printf("  round %d: prefill %.2f tok/s, decode %.2f tok/s, pass %.2f ms, MTP fill %.2f ms, step %.2f ms\n",
                    i + 1, r.prefill_tps, r.decode_tps, r.pass_ms, r.fill_ms, r.step_ms);
        pre.push_back(r.prefill_tps), dec.push_back(r.decode_tps), pms.push_back(r.pass_ms), fms.push_back(r.fill_ms),
            sms.push_back(r.step_ms);
    }
    summary("prefill", pre);
    summary("decode", dec);
    summary("pass", pms, "ms");
    if (!o.mtp.empty()) summary("MTP fill", fms, "ms"), summary("MTP step", sms, "ms");
#ifndef _WIN32
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);  // major faults: pages read back from disk (dropped weights, n-gram rows)
    std::printf("page-ins: %ld major faults over the whole run\n", ru.ru_majflt);
#endif
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 2;
    }
}
