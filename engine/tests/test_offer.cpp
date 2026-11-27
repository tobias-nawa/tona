// The offer data (data/modeldb/offer.json as embedded by tools/gen_offer_data.py): lookups and the invariants the UI
// relies on. The values checked here are the facts of the three models of the first release; a change of the data that
// breaks one of them is a decision, not a refactor.
#include <cstdio>
#include <exception>
#include <string>

#include "offer/offer.h"

using namespace tona::offer;

namespace {

int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "FAIL: %s\n", what), fails++;
}

constexpr const char* kClass = "pc-16gb-vram-64gb-ram";

void test_tasks() {
    expect(tasks().size() == 6, "six tasks");
    const Task* coding = find_task("coding");
    expect(coding && coding->available, "coding is available");
    expect(coding && coding->label == "Coding with agents", "label");
    const Task* knowledge = find_task("knowledge");
    expect(knowledge && knowledge->available, "knowledge is available");
    for (const char* id : {"documents", "transcription", "speech", "images"}) {
        const Task* t = find_task(id);
        expect(t && !t->available && !t->coming_note.empty(), "a coming task says so");
        expect(offer_for(id, kClass) == nullptr, "a coming task has no offer");
    }
    expect(find_task("nonsense") == nullptr, "unknown task");
}

void test_offers() {
    const Offer* coding = offer_for("coding", kClass);
    expect(coding && coding->recommended == "coder-iq1_m", "coding recommends the Coder");
    expect(coding && coding->alternative == "ornith-1.5-35b-a3b", "coding offers Ornith as the alternative");
    const Offer* knowledge = offer_for("knowledge", kClass);
    expect(knowledge && knowledge->recommended == "nemotron-bolt-74b-a9b-24gb", "knowledge recommends Bolt");
    expect(knowledge && knowledge->alternative.empty(), "knowledge has no alternative yet");
    expect(offer_for("coding", "no-such-class") == nullptr, "unknown class has no offer");
    const HwClass* hw = find_hw_class(kClass);
    expect(hw && hw->measured, "the reference class is measured");
    for (const Offer& o : offers()) {
        expect(model_card(o.recommended) != nullptr, "the recommended model has a card");
        expect(o.alternative.empty() || (o.alternative != o.recommended && model_card(o.alternative)),
               "the alternative is another model with a card");
    }
}

void test_cards() {
    expect(model_cards().size() == 3, "three cards");
    expect(model_card("nonsense") == nullptr, "unknown model");
    const ModelCard* coder = model_card("coder-iq1_m");
    const ModelCard* ornith = model_card("ornith-1.5-35b-a3b");
    const ModelCard* bolt = model_card("nemotron-bolt-74b-a9b-24gb");
    expect(coder && ornith && bolt, "cards of all three models");
    if (!coder || !ornith || !bolt) return;
    expect(coder->name == "Coder" && ornith->name == "Ornith" && bolt->name == "Bolt", "display names");
    expect(coder->size_bytes == 29608446496ULL + 28800138432ULL, "Coder size is the sum of its two shards");
    expect(ornith->size_bytes == 21713463040ULL, "Ornith size");
    expect(bolt->size_bytes == 22018326528ULL, "Bolt size");
    expect(coder->verdict("rust")->level == Level::Weak, "the Coder is weak at Rust");
    expect(ornith->verdict("rust")->level == Level::Strong, "Ornith is strong at Rust");
    expect(coder->verdict("csharp")->level == Level::Strong, "the Coder is strong at C#");
    expect(ornith->verdict("go")->level == Level::Open, "Ornith's Go is open");
    expect(coder->verdict("cobol") == nullptr, "unknown language");
}

void test_invariants() {
    for (const ModelCard& c : model_cards()) {
        expect(c.languages.size() == languages().size(), "a verdict for every language");
        for (size_t i = 0; i < c.languages.size() && i < languages().size(); i++) {
            const LanguageVerdict& v = c.languages[i];
            expect(v.language == languages()[i].id, "verdicts follow the language list");
            expect(!v.source.empty() && !v.detail.empty(), "verdict has source and detail");
            expect((v.level == Level::Open) == (v.basis == Basis::None || v.basis == Basis::Provisional),
                   "open claims no basis, a verdict names one");
        }
        expect(!c.rates.empty() && !c.strengths.empty() && !c.weaknesses.empty(), "card is filled");
        for (const Rate& r : c.rates)
            expect((r.decode_tps || r.prefill_tps) && !r.source.empty(), "a rate has a number and a source");
        expect(!c.license.text.empty() && !c.license.source.empty(), "license line with source");
    }
}

void test_rate_at() {
    const ModelCard* ornith = model_card("ornith-1.5-35b-a3b");
    if (!ornith) return expect(false, "Ornith card");
    expect(ornith->rate_at(2000)->context_tokens == 2000, "exact context");
    expect(ornith->rate_at(20000)->context_tokens == 16000, "the largest context not above");
    expect(ornith->rate_at(1)->context_tokens == 2000, "below all: the smallest");
    const std::optional<double> decode = ornith->rate_at(2000)->decode_tps;
    expect(decode.has_value() && decode.value() > 100, "Ornith decodes above 100");
}

void test_words() {
    expect(basis_word(Basis::Measured) == "measured", "measured");
    expect(basis_word(Basis::Vendor) == "vendor", "vendor");
}

int run() {
    test_tasks();
    test_offers();
    test_cards();
    test_invariants();
    test_rate_at();
    test_words();
    if (fails) return 1;
    std::puts("offer: all passed");
    return 0;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }
}
