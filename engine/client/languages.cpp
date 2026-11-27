#include "client/languages.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>

namespace fs = std::filesystem;

namespace tona::client {

namespace {

constexpr int kMaxDepth = 6;
constexpr size_t kMaxFiles = 4000;  // enough for a large project's top levels, still instant on a slow disk

struct ExtensionLanguage {
    const char* extension;
    const char* language;
};

constexpr ExtensionLanguage kExtensions[] = {{".rs", "Rust"},       {".go", "Go"},           {".py", "Python"},
                                             {".js", "JavaScript"}, {".mjs", "JavaScript"},  {".jsx", "JavaScript"},
                                             {".ts", "TypeScript"}, {".tsx", "TypeScript"},  {".cs", "C#"},
                                             {".java", "Java"},     {".kt", "Kotlin"},       {".c", "C"},
                                             {".h", "C"},           {".cpp", "C++"},         {".cc", "C++"},
                                             {".hpp", "C++"},       {".hip", "C++"},         {".cu", "C++"},
                                             {".sh", "Shell/Bash"}, {".bash", "Shell/Bash"}, {".rb", "Ruby"},
                                             {".php", "PHP"},       {".swift", "Swift"},     {".lua", "Lua"},
                                             {".zig", "Zig"},       {".sql", "SQL"},         {".scala", "Scala"},
                                             {".hs", "Haskell"},    {".ex", "Elixir"},       {".dart", "Dart/Flutter"}};

bool skipped_folder(const std::string& name) {
    static const char* names[] = {"node_modules", "target",      "build",       "dist",
                                  "vendor",       "third_party", "__pycache__", "work"};
    if (name.size() > 1 && name[0] == '.') return true;  // .git, .venv, .cache
    return std::any_of(std::begin(names), std::end(names), [&](const char* n) { return name == n; });
}

}  // namespace

std::string language_of(const std::string& file_name) {
    const std::string ext = fs::path(file_name).extension().string();
    for (const ExtensionLanguage& e : kExtensions)
        if (ext == e.extension) return e.language;
    return {};
}

std::vector<LanguageShare> detect_languages(const std::string& dir) {
    std::map<std::string, double> bytes;
    size_t files = 0;
    std::error_code ec;
    fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
    for (; !ec && it != end && files < kMaxFiles; it.increment(ec)) {
        if (it.depth() >= kMaxDepth) it.disable_recursion_pending();
        std::error_code type_ec;
        if (it->is_directory(type_ec)) {
            if (skipped_folder(it->path().filename().string())) it.disable_recursion_pending();
            continue;
        }
        const std::string language = language_of(it->path().filename().string());
        if (language.empty() || !it->is_regular_file(type_ec)) continue;
        files++;
        std::error_code size_ec;
        const auto size = fs::file_size(it->path(), size_ec);
        bytes[language] += size_ec ? 0.0 : static_cast<double>(size);
    }
    double total = 0;
    for (const auto& [language, b] : bytes) total += b;
    std::vector<LanguageShare> out;
    for (const auto& [language, b] : bytes)
        if (total > 0) out.push_back({language, b / total});
    std::sort(out.begin(), out.end(), [](const LanguageShare& a, const LanguageShare& b) {
        return a.share != b.share ? a.share > b.share : a.language < b.language;
    });
    return out;
}

std::string expand_home(const std::string& path) {
    const char* home = std::getenv("HOME");
    if (path.empty() || path[0] != '~' || !home || !*home) return path;
    if (path.size() > 1 && path[1] != '/') return path;  // ~user: not supported
    return std::string(home) + path.substr(1);
}

std::string languages_text(const std::vector<LanguageShare>& langs, size_t count) {
    std::string out;
    for (size_t i = 0; i < langs.size() && i < count; i++)
        out +=
            (i ? ", " : "") + langs[i].language + " " + std::to_string((int)std::lround(langs[i].share * 100)) + " %";
    return out;
}

}  // namespace tona::client
