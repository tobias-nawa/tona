// tools/tona_prompt_list.h: the prompt list of tona-gpu-gen --prompts-file.
#include "tests/test_support.h"
#include "tools/tona_prompt_list.h"

namespace {

using tona::tools::parse_prompt_list;
using tona::tools::read_file_bytes;

bool throws_invalid(const std::string& text) {
    try {
        parse_prompt_list(text);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_entries_in_order() {
    const auto e = parse_prompt_list("out/a.txt\tin/a.prompt\nout/b.txt\tin/b.prompt\n");
    EXPECT(e.size() == 2);
    EXPECT(e[0].output_path == "out/a.txt" && e[0].prompt_path == "in/a.prompt");
    EXPECT(e[1].output_path == "out/b.txt" && e[1].prompt_path == "in/b.prompt");
}

void test_blank_lines_and_missing_final_newline() {
    const auto e = parse_prompt_list("\nx\ty\n\n\nz\tw");
    EXPECT(e.size() == 2);
    EXPECT(e[1].output_path == "z" && e[1].prompt_path == "w");
    EXPECT(parse_prompt_list("").empty());
}

void test_path_with_spaces() {
    const auto e = parse_prompt_list("out dir/a b.txt\tin dir/p q\n");
    EXPECT(e.size() == 1 && e[0].output_path == "out dir/a b.txt" && e[0].prompt_path == "in dir/p q");
}

void test_malformed_lines() {
    EXPECT(throws_invalid("no tab here\n"));
    EXPECT(throws_invalid("\tprompt-only\n"));
    EXPECT(throws_invalid("output-only\t\n"));
}

void test_file_bytes_unchanged() {
    tona::test::TempDir dir("prompt-list");
    const std::string bytes = std::string("line one\n\n\"quoted\" \xc3\xa4\n\n") + '\0' + "tail\n\n";
    const auto path = dir.path() / "prompt";
    std::ofstream(path, std::ios::binary) << bytes;
    EXPECT(read_file_bytes(path.string()) == bytes);
    bool threw = false;
    try {
        read_file_bytes((dir.path() / "missing").string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT(threw);
}

}  // namespace

int main() {
    return tona::test::run("test_prompt_list", [] {
        test_entries_in_order();
        test_blank_lines_and_missing_final_newline();
        test_path_with_spaces();
        test_malformed_lines();
        test_file_bytes_unchanged();
    });
}
