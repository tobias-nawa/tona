// Shared by the client tests: an offer and a catalog like the real ones (Coder, Ornith, Bolt and one model that is
// coming), recorded-shape /status documents, and a UI state built from them. No terminal, no server.
#pragma once

#include <fstream>
#include <sstream>
#include <string>

#include "client/text.h"
#include "client/ui.h"
#include "client/view.h"
#include "tests/test_support.h"

namespace tona::test {

namespace cl = tona::client;
namespace in = tona::install;

inline std::string read_text(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline bool has(const std::string& text, const std::string& part) { return text.contains(part); }

inline std::string joined(const cl::Lines& lines) {
    std::string s;
    for (const std::string& l : lines) s += l + "\n";
    return s;
}

inline cl::ModelCard card(const std::string& id, const std::string& name, double gb, std::vector<std::string> strong) {
    cl::ModelCard c;
    c.id = id, c.name = name, c.size_gb = gb, c.strong = std::move(strong), c.tasks = {"coding"};
    c.license = "MIT", c.license_checked = false;
    return c;
}

inline cl::Offer test_offer() {
    cl::Offer o;
    o.tasks = {{"coding", "Coden", false}, {"knowledge", "Wissen", false}};
    cl::ModelCard coder = card("coder-iq1_m", "Coder", 55, {"Python", "JavaScript", "C#"});
    coder.decode_tok_s = 96, coder.speed_context = 16000, coder.speed_basis = cl::Basis::Measured;
    cl::ModelCard ornith = card("ornith-1.5-35b-a3b", "Ornith", 21, {"Rust", "Go"});
    ornith.decode_tok_s = 120, ornith.speed_context = 16000, ornith.speed_basis = cl::Basis::Measured,
    ornith.weak_note = "Messung läuft";
    cl::ModelCard bolt = card("nemotron-bolt-74b-a9b-24gb", "Bolt", 41, {});
    bolt.tasks = {"knowledge"};
    cl::ModelCard coming;
    coming.name = "Qwen3.8 27B", coming.coming = true, coming.note = "laut Hersteller stark, wird gerade eingebaut";
    coming.tasks = {"coding"};
    o.cards = {coder, ornith, bolt, coming};
    return o;
}

inline in::ModelInfo info(const std::string& id, const std::string& file, bool present,
                          const std::string& kind = "model") {
    in::ModelInfo m;
    m.id = id, m.title = id, m.kind = kind, m.first_file = file, m.present = present, m.tested = true;
    return m;
}

inline std::vector<in::ModelInfo> test_catalog() {
    return {info("coder-iq1_m", "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf", true),
            info("ornith-1.5-35b-a3b", "ornith.gguf", true), info("nemotron-bolt-74b-a9b-24gb", "bolt.gguf", false),
            info("mtp-q8_0", "mtp.gguf", true, "drafter")};
}

// A state like the Start page of the design: Coder running with two conversations
inline cl::UiState test_state(const std::string& status_file, const std::string& dir) {
    cl::UiState s;
    s.offer = test_offer(), s.catalog = test_catalog();
    const nlohmann::json status = nlohmann::json::parse(read_text(dir + "/" + status_file));
    s.server = cl::parse_status(status, true, "");
    s.loaded_id = "coder-iq1_m";
    s.ram_total_gb = 64, s.vram_total_gb = 16;
    s.rates = {0, 20, 60, 90, 96, 96, 80, 96, 40, 96};
    s.projects = {"tona", "shop"};
    s.agents.project = "~/projekte/sensor-daemon";
    s.languages = {{"Rust", 0.81}, {"Shell", 0.12}};
    s.now = 1763925000;
    s.launches = {{"claude", "~/tona", "Coder", 1763924000}};
    return s;
}

}  // namespace tona::test
