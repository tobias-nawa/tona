// Reader for the agent corpus written by tools/agent_corpus.py, used by the simulators tona-spec-sim and
// tona-sfx-policy. Format: "C <name>" starts a session; "S <role> <bytes>" is followed by that many bytes of text (role
// "out" is model output, anything else is context).
#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tona::tools {

struct CorpusSegment {
    bool is_output;
    std::string text;
};

struct CorpusSession {
    std::string name;
    std::vector<CorpusSegment> segments;
};

inline std::vector<CorpusSession> read_corpus(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string(path) + ": cannot read");
    std::vector<CorpusSession> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.starts_with("C ")) {
            out.push_back({line.substr(2), {}});
        } else if (line.starts_with("S ")) {
            std::istringstream ls(line.substr(2));
            std::string role;
            size_t n;
            ls >> role >> n;
            std::string text(n, '\0');
            f.read(text.data(), n);
            out.back().segments.push_back({role == "out", std::move(text)});
        } else {
            throw std::runtime_error("corpus: unexpected line");
        }
    }
    return out;
}

}  // namespace tona::tools
