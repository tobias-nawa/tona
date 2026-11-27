// The offer data against the embedded catalog: every model the offer names can be fetched (pinned) and its size is the
// size of the pinned files, so the UI never offers what the installer cannot download.
#include <cstdio>
#include <string>

#include "core/catalog.h"
#include "offer/offer.h"

int main() {
    int fails = 0;
    auto expect = [&](bool ok, const std::string& what) {
        if (!ok) std::fprintf(stderr, "FAIL: %s\n", what.c_str()), fails++;
    };
    for (const auto& card : tona::offer::model_cards()) {
        const auto* e = tona::install::find_entry(card.id);
        expect(e != nullptr, card.id + " is in the embedded catalog");
        if (!e) continue;
        expect(tona::install::pinned(*e), card.id + " is pinned (repository and full commit hash)");
        expect(e->kind == "model" && e->tested, card.id + " is a tested model");
        uint64_t sum = 0;
        for (const auto& f : e->files) sum += f.size;
        expect(sum == card.size_bytes, card.id + ": card size equals the pinned files");
    }
    for (const char* id : {"ornith-1.5-35b-a3b", "nemotron-bolt-74b-a9b-24gb"})
        expect(tona::install::find_profile(id) != nullptr, std::string(id) + " has an expert profile");
    if (fails) return 1;
    std::puts("offer catalog: all passed");
    return 0;
}
