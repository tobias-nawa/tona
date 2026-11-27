// Programming languages of a project folder, found by file extension and weighted by file size. The Agents page shows
// them and checks them against the strengths of the models (Tona only suggests, it never switches on its own).
#pragma once

#include <string>
#include <vector>

namespace tona::client {

struct LanguageShare {
    std::string language;  // "Rust", "Python", ... (names as the offer data uses them)
    double share = 0;      // 0..1 of the bytes of all source files found
};

// Language of one file name by its extension ("main.rs" -> "Rust"); empty for files that are no source code
std::string language_of(const std::string& file_name);

// Looks through `dir` (depth at most 6, at most 4000 files; hidden folders, build output and dependency folders are
// skipped) and returns the languages by falling share. A missing folder gives an empty list.
std::vector<LanguageShare> detect_languages(const std::string& dir);

// "~/x" to "$HOME/x"; other paths stay as they are
std::string expand_home(const std::string& path);

// "Rust 81 %, Shell 12 %" (the first `count` entries)
std::string languages_text(const std::vector<LanguageShare>& langs, size_t count = 3);

}  // namespace tona::client
