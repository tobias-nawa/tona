#include "backend/metal/weights.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <stdexcept>

#include "backend/env.h"
#include "backend/metal/host_math.h"
#include "model/quant.h"

namespace tona::metal {

const std::map<uint32_t, std::string>& kernel_types() {
    static const std::map<uint32_t, std::string> names = {
        {0, "F32"},     {1, "F16"},     {8, "Q80"},        {12, "Q4K"},   {13, "Q5K"},  {14, "Q6K"},
        {16, "IQ2XXS"}, {18, "IQ3XXS"}, {20, "IQ4NL"},     {21, "IQ3S"},  {22, "IQ2S"}, {23, "IQ4XS"},
        {30, "BF16"},   {42, "Q20"},    {1018, "IQ3XXSR"}, {1019, "Q6KT"}};
    return names;
}

std::string kernel_type(uint32_t t) {
    auto it = kernel_types().find(t);
    return it == kernel_types().end() ? "" : it->second;
}

Weights::Weights(Device& dev, const Qwen4ExpModel& m) : dev_(dev) {
    // The n-gram table stays out of the GPU's buffers: 28.8 GB in the Coder (more than the M1 Max's working set), of
    // which a token reads 16 rows of 90 bytes; Pass copies those rows from the mapping (Mat::host)
    host_ = m.ple_embd();
    arena_ = Arena::create(dev, m);
    add(m.gguf(), arena_ ? arena_->tensor_names() : std::vector<std::string>{});
}

namespace {

// Probe TONA_METAL_WINDOW_MB: smaller buffers, so that a command buffer binds only the pieces its layers read (a model
// over the working set, research/measurements/2026-11-15-mac-coder-baseline.md); default maxBufferLength
uint64_t window_limit(uint64_t max_buffer) {
    const int64_t mb = env::integer("TONA_METAL_WINDOW_MB", 0);
    return mb > 0 ? std::min<uint64_t>(max_buffer, (uint64_t)mb << 20) : max_buffer;
}

}  // namespace

void Weights::add(const Gguf& g, const std::vector<std::string>& skip) {
    Device& dev = dev_;
    const uint64_t page = (uint64_t)getpagesize();
    std::vector<std::vector<const TensorInfo*>> per_file(g.files());
    for (const TensorInfo& t : g.tensors())
        if (&t != host_ && std::find(skip.begin(), skip.end(), t.name) == skip.end()) per_file[t.file].push_back(&t);
    for (auto& ts : per_file) {
        if (ts.empty()) continue;
        std::ranges::sort(ts, {}, &TensorInfo::offset);  // file order (all tensors of ts share a file)
        const uint64_t limit = window_limit(dev.dev().maxBufferLength) / page * page;
        for (const Window& w : windows(ts, limit, page, arena_ ? page : UINT64_MAX)) {
            // The mapping covers whole pages, so a window may end past the end of the file inside its last page
            id<MTLBuffer> b = [dev.dev() newBufferWithBytesNoCopy:(void*)w.base
                                                           length:w.bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
            if (!b) throw std::runtime_error("Metal: cannot wrap " + g.file_path(ts.front()->file) + " as buffers");
            bufs_.push_back({b, w.base});
            mapped_ += w.bytes;
        }
        for (const TensorInfo* t : ts) {
            if (t->name.find("_exps") == std::string::npos) dense_.emplace_back(t->data, t->bytes);
            const auto& [b, base] = *std::find_if(bufs_.rbegin(), bufs_.rend(), [&](const auto& e) {
                return t->data >= e.second && t->data + t->bytes <= e.second + e.first.length;
            });
            at_[t] = {b, (uint64_t)(t->data - base)};
        }
    }
}

namespace {

// mlock of whole pages around [p, p + n); false if the system refuses (wire limit)
bool lock_pages(const uint8_t* p, uint64_t n) {
    const auto page = (uintptr_t)getpagesize();
    const uintptr_t a = (uintptr_t)p / page * page, e = ((uintptr_t)p + n + page - 1) / page * page;
    return mlock((const void*)a, e - a) == 0;
}

}  // namespace

uint64_t Weights::pin(const std::string& mode) {
    uint64_t bytes = 0;
    if (mode == "rset") {
        if (@available(macOS 15.0, *)) {
            MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
            d.initialCapacity = bufs_.size();
            NSError* err = nil;
            id<MTLResidencySet> s = [dev_.dev() newResidencySetWithDescriptor:d error:&err];
            if (!s) throw std::runtime_error("Metal: no residency set");
            for (const auto& b : bufs_) [s addAllocation:b.first], bytes += b.first.length;
            [s commit];
            [dev_.queue() addResidencySet:s];
            [s requestResidency];
            rset_ = s;
        }
    } else if (mode == "mlock" || mode == "mlock-dense") {
        if (mode == "mlock")
            for (const auto& b : bufs_) bytes += lock_pages(b.second, b.first.length) ? b.first.length : 0;
        else
            for (const auto& [p, n] : dense_) bytes += lock_pages(p, n) ? n : 0;
    } else if (!mode.empty() && mode != "0") {
        throw std::runtime_error("TONA_METAL_PIN: rset, mlock or mlock-dense, not " + mode);
    }
    return bytes;
}

Mat Weights::mat(T t) const {
    if (!t) throw std::logic_error("Metal: missing tensor");
    Mat w = shape(*t);
    if (t == host_)
        w.host = t->data;
    else
        w.w = at_.at(t);
    return w;
}

Mat Weights::shape(const TensorInfo& tensor) {
    const TensorInfo* t = &tensor;
    const GgmlTypeInfo* ti = ggml_type_info(t->type);
    Mat w;
    w.kernel_type = kernel_type(t->type);
    if (w.kernel_type.empty()) throw std::runtime_error("Metal: no kernel for type " + std::string(ti->name));
    w.k = (uint32_t)t->dims[0];
    w.rows = t->dims.size() > 1 ? (uint32_t)t->dims[1] : 1;
    w.experts = t->dims.size() > 2 ? (uint32_t)t->dims[2] : 1;
    if (w.k % ti->block_elems || w.k % 8)
        throw std::runtime_error("Metal: " + t->name + ": row length " + std::to_string(w.k) + " not whole blocks");
    w.row_bytes = (uint64_t)type_row_bytes(t->type, w.k);
    w.expert_bytes = (uint64_t)w.rows * w.row_bytes;
    return w;
}

Buf Weights::vec(T t) {
    if (!t) throw std::logic_error("Metal: missing tensor");
    auto it = vecs_.find(t);
    if (it != vecs_.end()) return it->second;
    const int64_t n = (int64_t)t->elements();
    Buf b = dev_.floats(n);
    dequantize_row(t->type, t->data, b.f32(), n);
    return vecs_[t] = b;
}

}  // namespace tona::metal
