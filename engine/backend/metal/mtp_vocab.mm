#include "backend/metal/mtp_vocab.h"

#include <algorithm>
#include <cstring>

#include "backend/env.h"

namespace tona::metal {

MtpVocab::MtpVocab(Device& dev, const Qwen4ExpModel& m) : out_(*m.output()) {
    const uint32_t vocab = m.hp().vocab;
    const size_t dyn = (size_t)std::max<int64_t>(0, env::integer("TONA_MTP_DYN", 8192));
    map_ = backend::MtpVocabMap(
        backend::mtp_vocab_list(vocab, backend::MtpList::Wide, env::text("TONA_MTP_VOCAB"), "Metal"), dyn, vocab);
    const Mat full = Weights::shape(out_);
    head_ = full;
    head_.rows = (uint32_t)map_.size();
    head_.w = dev.floats(((uint64_t)head_.rows * full.row_bytes + 3) / 4 + 64);  // slack past the last row
    rows_ = dev.floats(map_.size());                                             // int32 per row
    std::memcpy(rows_.i32(), map_.data(), map_.size() * sizeof(int32_t));
    for (size_t r = 0; r < map_.n_static(); r++)
        std::memcpy((uint8_t*)head_.w.b.contents + r * full.row_bytes, out_.data + (uint64_t)map_[r] * full.row_bytes,
                    full.row_bytes);
}

void MtpVocab::sync(const std::vector<int32_t>& tokens, int32_t next, size_t& scan) {
    map_.sync(tokens, next, scan, [&](size_t ring, int32_t, int32_t t) {
        std::memcpy((uint8_t*)head_.w.b.contents + (map_.n_static() + ring) * head_.row_bytes,
                    out_.data + (uint64_t)t * head_.row_bytes, head_.row_bytes);
        rows_.i32()[map_.n_static() + ring] = t;
        loaded_++;
    });
}

}  // namespace tona::metal
