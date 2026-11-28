// Offline acceptance of a layer-skip draft, on the CPU reference. The full model generates greedily; every
// STRIDE tokens the draft pass of each mask in MASKS drafts DEPTH tokens from there, with the true tokens as prefix,
// and the tool records whether each drafted token equals the full model's: the accepted length of a self-speculative
// chain without running a kernel. Usage: tona-layer-skip MODEL.gguf --prompt FILE [--prompt FILE ...] --out OUT.jsonl
//          [--masks FILE] [--gen N] [--stride S] [--depth K] [--cos]
//   FILE of --prompt: the text of the prompt as the model sees it (chat markers included), one sequence per file
//   MASKS: one draft pass per line: "name [skip=3,5,7] [topk=4] [routed=off]"; skip = layers left out (0-based), topk =
//   routed experts per kept layer (default: the model's), routed=off: the shared expert only; # starts a comment
//   helper=FILE: the routed experts are replaced by the trained FFNs of FILE (cpu/ref_helper.h)
//   --moe-dump DIR: write the MoE input and its routed and shared output of every layer for every token the full model
//   runs, as F16 [token][layer][x, routed, shared][d], to DIR/moe-<sequence>.f16 (the data a helper is distilled from)
//   --cos: also write, per layer, the mean cosine and relative change of the residual the layer makes
// Output, one JSON object per line: {"kind":"seq"} (prompt and generated ids), {"kind":"cos"}, {"kind":"draft"}
// (match, top-1 probability and -log p of the true token per depth).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "cpu/ref.h"
#include "cpu/ref_helper.h"
#include "model/model_registry.h"
#include "model/quant.h"
#include "model/tokenizer.h"
#include "server/sampling.h"
#include "tona_args.h"

namespace {

struct Options {
    std::vector<const char*> prompts;
    const char* masks = nullptr;
    const char* out = nullptr;
    int gen = 128;
    int stride = 8;
    int depth = 6;
    bool cos = false;
    const char* moe_dir = nullptr;
};

struct Mask {
    std::string name;
    tona::ref::DraftPass pass;
};

struct DepthStat {
    int match = 0, top = 0;
    float ptop = 0, nll = 0;
};

struct LayerSum {
    double cos = 0, rel = 0;
    int n = 0;
};

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 2; i < argc; i++) {
        const bool more = i + 1 < argc;
        if (!std::strcmp(argv[i], "--cos")) {
            o.cos = true;
        } else if (!more) {
            throw std::invalid_argument(std::string("option without value: ") + argv[i]);
        } else if (!std::strcmp(argv[i], "--moe-dump")) {
            o.moe_dir = argv[++i];
        } else if (!std::strcmp(argv[i], "--prompt")) {
            o.prompts.push_back(argv[++i]);
        } else if (!std::strcmp(argv[i], "--masks")) {
            o.masks = argv[++i];
        } else if (!std::strcmp(argv[i], "--out")) {
            o.out = argv[++i];
        } else if (!std::strcmp(argv[i], "--gen")) {
            o.gen = tona::tools::parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--stride")) {
            o.stride = tona::tools::parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--depth")) {
            o.depth = tona::tools::parse_int(argv[++i]);
        } else {
            throw std::invalid_argument(std::string("unknown option: ") + argv[i]);
        }
    }
    if (o.prompts.empty() || !o.out) throw std::invalid_argument("--prompt and --out are required");
    if (o.stride <= o.depth) throw std::invalid_argument("--stride must exceed --depth");
    return o;
}

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) v.push_back(tona::tools::parse_int(item.c_str()));
    return v;
}

// "name [skip=a,b] [topk=n]"
Mask parse_mask(const std::string& line, size_t n_layer, size_t d) {
    std::stringstream ss(line);
    Mask m;
    ss >> m.name;
    m.pass.keep.assign(n_layer, 1);
    bool skipped = false;
    for (std::string tok; ss >> tok;) {
        if (tok.starts_with("skip=")) {
            for (int l : parse_list(tok.substr(5))) {
                if (l < 0 || (size_t)l >= n_layer) throw std::invalid_argument("layer outside the model: " + tok);
                m.pass.keep[l] = 0, skipped = true;
            }
        } else if (tok.starts_with("helper=")) {
            m.pass.helper = tona::ref::load_helper(tok.substr(7), n_layer, d);
        } else if (tok == "routed=off") {
            m.pass.n_used = -1;
        } else if (tok.starts_with("topk=")) {
            m.pass.n_used = tona::tools::parse_int(tok.c_str() + 5);
        } else {
            throw std::invalid_argument("mask term not understood: " + tok);
        }
    }
    if (!skipped) m.pass.keep.clear();
    return m;
}

std::vector<Mask> read_masks(const char* path, size_t n_layer, size_t d) {
    std::vector<Mask> masks;
    if (!path) return masks;
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("cannot read ") + path);
    for (std::string line; std::getline(f, line);)
        if (!line.empty() && line[0] != '#') masks.push_back(parse_mask(line, n_layer, d));
    return masks;
}

// argmax, top-1 probability and -log p of `truth` of one row of logits
DepthStat row_stat(const std::vector<float>& lg, int32_t truth) {
    const int top = tona::server::argmax(lg.data(), lg.size());
    double sum = 0;
    for (float v : lg) sum += std::exp((double)v - lg[top]);
    DepthStat s;
    s.match = top == truth;
    s.top = top;
    s.ptop = (float)(1.0 / sum);
    s.nll = (float)(std::log(sum) + lg[top] - lg[truth]);
    return s;
}

// Draft pass over seq[t], seq[t+1], ... with the true tokens as prefix: entry i judges the guess for seq[t+i+1]
std::vector<DepthStat> run_draft(tona::ref::Oracle& fw, const Mask& m, const std::vector<int32_t>& seq, size_t t,
                                 int depth) {
    fw.set_draft(m.pass);
    std::vector<DepthStat> r;
    r.reserve(depth);
    for (int i = 0; i < depth; i++) r.push_back(row_stat(fw.step(seq[t + i]), seq[t + i + 1]));
    fw.set_draft({});
    return r;
}

void write_draft(std::FILE* out, size_t seq_no, size_t t, const Mask& m, const std::vector<DepthStat>& r) {
    std::fprintf(out, R"({"kind":"draft","seq":%zu,"t":%zu,"mask":"%s","match":[)", seq_no, t, m.name.c_str());
    for (size_t i = 0; i < r.size(); i++) std::fprintf(out, "%s%d", i ? "," : "", r[i].match);
    std::fprintf(out, "],\"top\":[");
    for (size_t i = 0; i < r.size(); i++) std::fprintf(out, "%s%d", i ? "," : "", r[i].top);
    std::fprintf(out, "],\"ptop\":[");
    for (size_t i = 0; i < r.size(); i++) std::fprintf(out, "%s%.4f", i ? "," : "", r[i].ptop);
    std::fprintf(out, "],\"nll\":[");
    for (size_t i = 0; i < r.size(); i++) std::fprintf(out, "%s%.4f", i ? "," : "", r[i].nll);
    std::fprintf(out, "]}\n");
}

// Cosine and relative size of what layer l added to the residual (layer 0: the embedding is not in the trace)
void add_layer_change(std::vector<LayerSum>& sum, const tona::ref::Trace& tr) {
    sum.resize(tr.residual.size());
    for (size_t l = 1; l < tr.residual.size(); l++) {
        double dot = 0, na = 0, nb = 0, nd = 0;
        const auto &a = tr.residual[l - 1], &b = tr.residual[l];
        for (size_t i = 0; i < a.size(); i++) {
            dot += (double)a[i] * b[i], na += (double)a[i] * a[i], nb += (double)b[i] * b[i];
            nd += ((double)b[i] - a[i]) * ((double)b[i] - a[i]);
        }
        sum[l].cos += dot / std::sqrt(na * nb), sum[l].rel += std::sqrt(nd / na), sum[l].n++;
    }
}

// One token's MoE data of every layer as F16
void dump_moe(std::FILE* f, const tona::ref::Trace& tr) {
    std::vector<uint16_t> buf;
    for (size_t l = 0; l < tr.moe_x.size(); l++)
        for (const auto* v : {&tr.moe_x[l], &tr.moe_routed[l], &tr.moe_shared[l]})
            for (float x : *v) buf.push_back(tona::fp32_to_fp16(x));
    std::fwrite(buf.data(), sizeof(uint16_t), buf.size(), f);
}

struct Run {
    tona::ref::Oracle& fw;
    const std::vector<Mask>& masks;
    const Options& o;
    std::FILE* out;
    std::vector<LayerSum>& layers;
    size_t seq_no;
};

// Drafts from the state saved at position t, when seq reaches t + depth + 1; the current state comes back afterwards
void draft_all(Run& r, const tona::ref::Snapshot& at_t, size_t t, const std::vector<int32_t>& seq) {
    const auto now = r.fw.snapshot();
    for (const Mask& m : r.masks) {
        r.fw.restore(at_t);  // every mask starts from the same state
        write_draft(r.out, r.seq_no, t, m, run_draft(r.fw, m, seq, t, r.o.depth));
    }
    r.fw.restore(*now);
}

std::FILE* open_moe_file(const Run& r) {
    if (!r.o.moe_dir) return nullptr;
    const std::string name = std::string(r.o.moe_dir) + "/moe-" + std::to_string(r.seq_no) + ".f16";
    std::FILE* f = std::fopen(name.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + name);
    return f;
}

void write_seq(const Run& r, const std::vector<int32_t>& seq, size_t prompt) {
    std::fprintf(r.out, R"({"kind":"seq","seq":%zu,"prompt":%zu,"ids":[)", r.seq_no, prompt);
    for (size_t i = 0; i < seq.size(); i++) std::fprintf(r.out, "%s%d", i ? "," : "", seq[i]);
    std::fprintf(r.out, "]}\n");
    std::fflush(r.out);
}

// A draft start every stride tokens after the prompt: its state is saved and drafted from once the tokens are known
struct Pending {
    std::unique_ptr<tona::ref::Snapshot> state;
    size_t t = 0;
};

// Full step on seq[fed]; the argmax of the logits extends seq once the prompt is through
void full_step(Run& r, std::vector<int32_t>& seq, size_t fed, size_t prompt, std::FILE* moe_file) {
    tona::ref::Trace trace;
    trace.want_moe = moe_file != nullptr;
    const auto logits = r.fw.step(seq[fed], r.o.cos || moe_file ? &trace : nullptr);
    if (moe_file) dump_moe(moe_file, trace);
    if (r.o.cos && fed >= prompt) add_layer_change(r.layers, trace);
    if (fed + 1 >= prompt) seq.push_back(tona::server::argmax(logits.data(), logits.size()));
}

void run_sequence(Run& r, const tona::Tokenizer& tok, const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(f)), {});
    std::vector<int32_t> seq = tok.encode(text);
    const size_t P = seq.size();
    std::FILE* moe_file = open_moe_file(r);
    Pending pend;
    for (size_t fed = 0; fed + 1 < P + r.o.gen + 1; fed++) {
        if (fed >= P && (fed - P) % r.o.stride == 0 && !r.masks.empty()) pend.state = r.fw.snapshot(), pend.t = fed;
        full_step(r, seq, fed, P, moe_file);
        if (pend.state && seq.size() > pend.t + r.o.depth) draft_all(r, *pend.state, pend.t, seq), pend.state.reset();
        if ((fed + 1) % 16 == 0) std::fprintf(stderr, "%s: %zu/%zu\n", path, fed + 1, P + r.o.gen);
    }
    write_seq(r, seq, P);
    if (moe_file) std::fclose(moe_file);
    r.fw.reset();
}

void write_layers(std::FILE* out, const std::vector<LayerSum>& layers) {
    for (size_t l = 1; l < layers.size(); l++)
        if (layers[l].n)
            std::fprintf(out, "{\"kind\":\"cos\",\"layer\":%zu,\"cos\":%.6f,\"rel\":%.6f,\"n\":%d}\n", l,
                         layers[l].cos / layers[l].n, layers[l].rel / layers[l].n, layers[l].n);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: tona-layer-skip MODEL.gguf --prompt FILE --out OUT.jsonl [--masks FILE]\n");
        return 2;
    }
    try {
        const Options o = parse_options(argc, argv);
        const auto m = tona::load_model(argv[1]);
        const tona::Tokenizer tok(m->gguf());
        const auto fw = tona::ref::make_oracle(*m);
        const auto masks = read_masks(o.masks, m->dims().n_layer, m->dims().d_model);
        std::FILE* out = std::fopen(o.out, "w");
        if (!out) throw std::runtime_error(std::string("cannot write ") + o.out);
        std::vector<LayerSum> layers;
        size_t n = 0;
        for (const char* p : o.prompts) {
            Run r{*fw, masks, o, out, layers, n++};
            run_sequence(r, tok, p);
        }
        write_layers(out, layers);
        std::fclose(out);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
