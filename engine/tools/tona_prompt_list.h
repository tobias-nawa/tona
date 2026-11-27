// The prompt list of `tona-gpu-gen --prompts-file` (scripts/golden.sh): several prompts in one process, so the model
// loads once (18 s) instead of once per prompt.
#pragma once

#include <string>
#include <vector>

namespace tona::tools {

struct PromptEntry {
    std::string prompt_path;  // file with the raw prompt bytes
    std::string output_path;  // file that receives stdout of this prompt
};

// One entry per line: OUTPUT_PATH, a tab, PROMPT_PATH. Empty lines are skipped. Prompts live in files of their own
// because they hold newlines and quotes; a path may hold anything except a tab and a newline. Throws
// std::invalid_argument on a line without a tab or with an empty path.
std::vector<PromptEntry> parse_prompt_list(const std::string& text);

// Whole file as bytes, nothing stripped (the golden references depend on the exact prompt). Throws on a missing file.
std::string read_file_bytes(const std::string& path);

}  // namespace tona::tools
