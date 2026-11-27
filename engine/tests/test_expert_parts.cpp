// The slot of an expert (gpu/expert_parts.h): layout per layer from the layer's own tensors (a gated layer as the
// Coder's, an ungated latent layer of Bolt with its own width and expert count), the parts in slot order, and the
// size class of the router rows for 496 experts. Host only, the tensors are descriptions without data.
#include <cstdint>

#include "backend/expert_layout.h"
#include "gpu/expert_parts.h"
#include "gpu/mega_mailbox_fwd.h"
#include "tests/test_support.h"

namespace {

using namespace tona;
constexpr uint32_t Q8_0 = 8, IQ2_XXS = 16;

TensorInfo tensor(uint32_t type, std::vector<uint64_t> dims) {
    TensorInfo t;
    t.type = type;
    t.dims = std::move(dims);
    return t;
}

// A Bolt MoE layer: ungated experts of width ff in the latent space 1024, E experts, IQ2_XXS
struct BoltLayer {
    TensorInfo up, down;
    MoeWeights w;
    BoltLayer(uint64_t ff, uint64_t E) : up(tensor(IQ2_XXS, {1024, ff, E})), down(tensor(IQ2_XXS, {ff, 1024, E})) {
        w.up_exps = &up, w.down_exps = &down;
    }
};

void test_ungated_layers_differ() {
    const BoltLayer wide(2560, 496), narrow(1536, 240);
    const gpu::SlotLayout a = gpu::slot_layout(wide.w), b = gpu::slot_layout(narrow.w);
    // IQ2_XXS: 66 bytes per 256 weights; a row of the latent space 264 bytes, a down row of ff 2560: 660 bytes
    EXPECT(a.off_up == 0 && a.off_down == size_t{2560} * 264 && a.slot_bytes == size_t{2} * 2560 * 264);
    EXPECT(b.off_up == 0 && b.off_down == size_t{1536} * 264 && b.slot_bytes == size_t{2} * 1536 * 264);
    // The parts in slot order: up, down; the third is not used
    gpu::LayerCache C;
    C.off_up = a.off_up, C.off_down = a.off_down, C.slot_bytes = a.slot_bytes;
    const gpu::ExpertParts P = gpu::expert_parts(wide.w, C);
    EXPECT(P.n == 2 && P.t[0] == &wide.up && P.t[1] == &wide.down && P.off[0] == 0 && P.off[1] == a.off_down);
}

void test_gated_layer_keeps_the_coders_layout() {
    // d 2560, ff 512, IQ1-like sizes do not matter: Q8_0 rows (34 bytes per 32). gate | up | down, 256-aligned parts
    const TensorInfo gate = tensor(Q8_0, {2560, 512, 256}), up = tensor(Q8_0, {2560, 512, 256}),
                     down = tensor(Q8_0, {512, 2560, 256});
    MoeWeights w;
    w.gate_exps = &gate, w.up_exps = &up, w.down_exps = &down;
    const gpu::SlotLayout s = gpu::slot_layout(w);
    constexpr size_t kRowD = 2720, kRowFf = 544;  // bytes of a row of 2560 and of 512 values in Q8_0 (34 per 32)
    const size_t g = size_t{512} * kRowD, dn = size_t{2560} * kRowFf;
    EXPECT(s.off_up == (g + 255) / 256 * 256 && s.off_down == s.off_up + (g + 255) / 256 * 256);
    EXPECT(s.slot_bytes == s.off_down + (dn + 255) / 256 * 256);
    gpu::LayerCache C;
    C.off_up = s.off_up, C.off_down = s.off_down;
    const gpu::ExpertParts P = gpu::expert_parts(w, C);
    EXPECT(P.n == 3 && P.t[0] == &gate && P.off[0] == 0 && P.off[1] == s.off_up && P.off[2] == s.off_down);
}

// 496 experts need the wide size class: router rows of 512 floats; 240 and 256 stay in the Coder's layout
void test_router_rows_for_496_experts() {
    EXPECT(gpu::expert_cap(496) == gpu::MEGA_MAX_EXPERTS_WIDE && gpu::expert_cap(240) == gpu::MEGA_MAX_EXPERTS);
    EXPECT(gpu::expert_cap(256) == gpu::MEGA_MAX_EXPERTS && gpu::expert_cap(257) == gpu::MEGA_MAX_EXPERTS_WIDE);
    const backend::ExpertLayout layout({496, 240, 496, 320});
    EXPECT(layout.max_experts() == 496 && layout.offset(2) == 496 + 240 && layout.total() == 496 + 240 + 496 + 320);
}

}  // namespace

int main() {
    return tona::test::run("test_expert_parts", [] {
        test_ungated_layers_differ();
        test_gated_layer_keeps_the_coders_layout();
        test_router_rows_for_496_experts();
    });
}
