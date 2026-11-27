// Mixture of experts in gpu::Forward: routing, the VRAM expert cache and the expert matvecs.
#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

#include "backend/nan_guard.h"
#include "backend/omp_guard.h"
#include "gpu/forward.h"
#include "gpu/forward_util.h"
#include "gpu/ops.h"

namespace tona::gpu {

Forward::Expert Forward::expert(uint32_t l, int e) {
    LayerDev& D = L_[l];
    int slot;
    if (fake_experts()) {
        const uint8_t* base = D.slots + (size_t)(e % slots_) * D.slot_bytes;
        return {base, base + D.off_up, base + D.off_down};
    }
    auto it = D.where.find(e);
    if (it != D.where.end()) {
        stats_.expert_hits++;
        slot = it->second.first;
        D.lru.splice(D.lru.begin(), D.lru, it->second.second);
    } else {
        stats_.expert_misses++;
        if (D.free_slots.empty()) {
            const int victim = D.lru.back();
            D.lru.pop_back();
            slot = D.where[victim].first;
            D.where.erase(victim);
        } else {
            slot = D.free_slots.back();
            D.free_slots.pop_back();
        }
        const Layer& S = m_.layers()[l];
        uint8_t* dst = D.slots + slot * D.slot_bytes;
        auto copy = [&](const TensorInfo& t, size_t off) {
            const size_t n = t.bytes / t.dims[2];
            HIPCHECK(hipMemcpy(dst + off, t.data + (size_t)e * n, n, hipMemcpyHostToDevice));
            stats_.upload_bytes += (double)n;
        };
        copy(*S.gate_exps, 0);
        copy(*S.up_exps, D.off_up);
        copy(*S.down_exps, D.off_down);
        D.lru.push_front(e);
        D.where[e] = {slot, D.lru.begin()};
    }
    const uint8_t* base = D.slots + slot * D.slot_bytes;
    return {base, base + D.off_up, base + D.off_down};
}

bool Forward::reserve(uint32_t l, int e, Expert& out) {
    LayerDev& D = L_[l];
    int slot;
    auto it = D.where.find(e);
    const bool present = it != D.where.end();
    if (present) {
        stats_.expert_hits++;
        slot = it->second.first;
        D.lru.splice(D.lru.begin(), D.lru, it->second.second);
    } else {
        stats_.expert_misses++;
        if (D.free_slots.empty()) {
            const int victim = D.lru.back();  // the current selection is at the front, so it is not evicted
            D.lru.pop_back();
            slot = D.where[victim].first;
            D.where.erase(victim);
        } else {
            slot = D.free_slots.back();
            D.free_slots.pop_back();
        }
        D.lru.push_front(e);
        D.where[e] = {slot, D.lru.begin()};
    }
    const uint8_t* base = D.slots + slot * D.slot_bytes;
    out = {base, base + D.off_up, base + D.off_down};
    return !present;
}

// Router matvec, softmax, top k, renormalize: as cpu/ref.cpp
Forward::Routing Forward::route(uint32_t l, const Q8Vec* xq) {
    const Layer& S = m_.layers()[l];
    const int E = (int)h_.n_expert;
    mv(*S.router, L_[l].router, mixed_, t1_, xq);
    HIPCHECK(hipMemcpy(router_host_.data(), t1_, E * sizeof(float), hipMemcpyDeviceToHost));
    tona::nan::report("router logits (Forward)", tona::nan::clean(router_host_.data(), (size_t)E),
                      std::format("layer {}, position {}", l, pos_));
    const float mx = *std::max_element(router_host_.begin(), router_host_.end());
    double sum = 0;
    std::vector<float> p(E);
    for (int e = 0; e < E; e++) sum += (p[e] = std::exp(router_host_[e] - mx));
    for (auto& v : p) v = (float)(v / sum);
    Routing r;
    r.idx.resize(E);
    std::iota(r.idx.begin(), r.idx.end(), 0);
    std::partial_sort(r.idx.begin(), r.idx.begin() + h_.n_expert_used, r.idx.end(),
                      [&](int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
    r.idx.resize(h_.n_expert_used);
    double wsum = 0;
    for (int e : r.idx) wsum += p[e];
    for (int e : r.idx) r.weight.push_back((float)(p[e] / wsum));
    return r;
}

// One expert (pointers into its cache slot): gate and up, SiLU, down, added to y with its routing weight
void Forward::run_expert(uint32_t l, const Pending& p, const Q8Vec* xq, float* y) {
    const Layer& S = m_.layers()[l];
    const int d = (int)h_.d, ff = (int)h_.expert_ff;
    run_matvec(MatvecTask{S.gate_exps->type, d, ff, p.ex.gate, mixed_, t2_}, xq);
    run_matvec(MatvecTask{S.up_exps->type, d, ff, p.ex.up, mixed_, t3_}, xq);
    ops::silu_mul(t2_, t3_, ff);
    run_matvec(MatvecTask{S.down_exps->type, ff, d, p.ex.down, t2_, t4_}, quant(t2_, ff, qtmp_));
    ops::axpy(y, t4_, p.weight, d);
}

// Hits first; meanwhile the CPU copies the missing experts into the pinned buffer and the copy stream moves them to
// VRAM. Only then does the main stream compute the misses.
void Forward::run_experts_cached(uint32_t l, const Routing& r, const Q8Vec* xq, float* y) {
    std::vector<Pending> hit, miss;
    for (size_t i = 0; i < r.idx.size(); i++) {
        Pending p{r.idx[i], r.weight[i], {}};
        (reserve(l, p.e, p.ex) ? miss : hit).push_back(p);
    }
    for (const Pending& p : hit) run_expert(l, p, xq, y);
    if (miss.empty()) return;
    load_misses(l, miss);
    for (const Pending& p : miss) run_expert(l, p, xq, y);
}

void Forward::load_misses(uint32_t l, const std::vector<Pending>& miss) {
    const Layer& S = m_.layers()[l];
    const LayerDev& D = L_[l];
    HIPCHECK(hipStreamSynchronize(copy_stream_));  // buffer of the previous layer is free
    const TensorInfo* parts[3] = {S.gate_exps, S.up_exps, S.down_exps};
    const size_t offs[3] = {0, D.off_up, D.off_down};
    backend::OmpErrors err;  // a failing pread must reach the request, not end the process
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < (int)miss.size(); i++)
        for (int k = 0; k < 3; k++)
            err.run([&] {
                // pread instead of memcpy from the mapping: no page faults (measured 1.3 million per 200 tokens)
                const size_t n = parts[k]->bytes / parts[k]->dims[2];
                m_.gguf().read(*parts[k], (size_t)miss[i].e * n, staging_ + i * D.slot_bytes + offs[k], n);
            });
    err.rethrow();
    for (int i = 0; i < (int)miss.size(); i++) {
        HIPCHECK(hipMemcpyAsync(const_cast<void*>(miss[i].ex.gate), staging_ + i * D.slot_bytes, D.slot_bytes,
                                hipMemcpyHostToDevice, copy_stream_));
        stats_.upload_bytes += (double)D.slot_bytes;
    }
    HIPCHECK(hipEventRecord(copied_, copy_stream_));
    HIPCHECK(hipStreamWaitEvent(nullptr, copied_, 0));
}

void Forward::shared_expert(uint32_t l, const Q8Vec* xq, float* y) {
    const Layer& S = m_.layers()[l];
    const LayerDev& D = L_[l];
    mv(*S.shexp_gate, D.sh_gate, mixed_, t2_, xq);
    mv(*S.shexp_up, D.sh_up, mixed_, t3_, xq);
    ops::silu_mul(t2_, t3_, (int)h_.shexp_ff);
    mv(*S.shexp_down, D.sh_down, t2_, t4_, quant(t2_, (int64_t)h_.shexp_ff, qtmp_));
    mv(*S.shexp_gate_inp, D.sh_inp, mixed_, t5_, xq);
    ops::axpy_sigmoid(y, t4_, t5_, (int)h_.d);
}

void Forward::moe(uint32_t l, float* y) {
    const Q8Vec* xq = quant(mixed_, h_.d, qmix_);
    const Routing r = route(l, xq);
    HIPCHECK(hipMemsetAsync(y, 0, h_.d * sizeof(float)));
    if (fake_experts()) {
        for (size_t i = 0; i < r.idx.size(); i++) run_expert(l, {r.idx[i], r.weight[i], expert(l, r.idx[i])}, xq, y);
    } else {
        run_experts_cached(l, r, xq, y);
    }
    shared_expert(l, xq, y);
}

}  // namespace tona::gpu
