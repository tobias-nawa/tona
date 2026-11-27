#include "core/module.h"

namespace tona::install {

const std::vector<const Module*>& modules() {
    static const std::vector<const Module*> all = {&hip_gfx1200_module(), &metal_apple_module()};
    return all;
}

const Module* find_module(const Hardware& h) {
    for (const Module* m : modules())
        if (m->pick(h)) return m;
    return nullptr;
}

}  // namespace tona::install
