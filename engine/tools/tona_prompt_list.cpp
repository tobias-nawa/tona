#include "tools/tona_prompt_list.h"

#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace tona::tools {

std::vector<PromptEntry> parse_prompt_list(const std::string& text) {
    std::vector<PromptEntry> entries;
    std::istringstream in(text);
    std::string line;
    for (int n = 1; std::getline(in, line); n++) {
        if (line.empty()) continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos || tab == 0 || tab + 1 == line.size())
            throw std::invalid_argument("prompt list line " + std::to_string(n) + ": expected OUTPUT<TAB>PROMPT");
        entries.push_back({line.substr(tab + 1), line.substr(0, tab)});
    }
    return entries;
}

std::string read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

}  // namespace tona::tools
