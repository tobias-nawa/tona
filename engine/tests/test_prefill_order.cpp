// gpu/prefill_order.h: the layer order of the layer-by-layer prefill and the steps that must remake the layer input.
// The Bolt and Ornith prefill write the norm of the next layer's input into one buffer shared by all chunks; in the
// layer-by-layer order that norm belongs to another chunk at every step but the ones that follow their own chunk's
// previous layer (a missing remake gave KL 11.5 against the chunk-by-chunk result).
#include "gpu/prefill_order.h"
#include "tests/test_support.h"

int main() {
    using namespace tona::gpu;

    // chunk by chunk: each chunk runs through all layers, no step ever needs a remake
    auto order = pf_layer_order(5, 3, false);
    EXPECT(order.size() == 15);
    EXPECT(order[0] == (PfStep{0, 0}) && order[4] == (PfStep{4, 0}) && order[5] == (PfStep{0, 1}));
    for (size_t i = 0; i < order.size(); i++) EXPECT(!pf_needs_resume(order, i));

    // layer by layer: layers 0 and 1 per chunk, then every layer over all chunks
    order = pf_layer_order(5, 3, true);
    const std::vector<PfStep> want = {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {2, 0}, {2, 1},
                                      {2, 2}, {3, 0}, {3, 1}, {3, 2}, {4, 0}, {4, 1}, {4, 2}};
    EXPECT(order == want);
    size_t resumes = 0;
    for (size_t i = 0; i < order.size(); i++) {
        const bool r = pf_needs_resume(order, i);
        resumes += r;
        // layers 0 and 1 follow their own chunk's previous step; from layer 2 on no step does with three chunks
        EXPECT(r == (order[i].first >= 2));
    }
    EXPECT(resumes == 9);

    // one chunk: layered or not, every step follows its own previous layer
    order = pf_layer_order(5, 1, true);
    EXPECT(order == pf_layer_order(5, 1, false));
    for (size_t i = 0; i < order.size(); i++) EXPECT(!pf_needs_resume(order, i));

    // two chunks: the first step of layer 2 comes after the last chunk's layer 1
    order = pf_layer_order(4, 2, true);
    EXPECT(order[4] == (PfStep{2, 0}) && pf_needs_resume(order, 4));
    return tona::test::finish("prefill_order");
}
