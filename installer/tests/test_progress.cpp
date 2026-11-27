// Progress of the download and the check (sizes, rate, remaining time, the sink) and the final page "What next"
// (fixed columns, PATH hint). No network, no model files. Usage: test_progress
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/progress.h"
#include "core/release_steps.h"
#include "core/space.h"
#include "tests/test_support.h"
#include "tui.h"

namespace in = tona::install;
using namespace tona::test;

namespace {

void test_formatting() {
    EXPECT(in::size_text(58e9) == "58.0 GB");
    EXPECT(in::size_text(12.34e9) == "12.3 GB");
    EXPECT(in::rate_text(45'200'000) == "45.2 MB/s");
    EXPECT(in::remaining_text(35) == "35 s");
    EXPECT(in::remaining_text(247) == "4 min 07 s");
    EXPECT(in::remaining_text(7380) == "2 h 03 min");
    const in::Progress p{in::Phase::Download, 29'000'000'000ULL, 58'000'000'000ULL};
    EXPECT(in::percent_of(p) == 50);
    EXPECT(in::percent_of({}) == 0);
}

void test_rate_and_eta() {
    in::RateMeter m;
    const uint64_t total = 100'000'000;
    for (int t = 0; t < 3; t++) m.add(t, static_cast<uint64_t>(t) * 10'000'000);
    EXPECT(m.rate() == 0 && m.remaining(total) < 0);  // 2 s of samples: not stable yet
    EXPECT(!in::progress_text({in::Phase::Download, 20'000'000, total}, m).contains("MB/s"));
    m.add(3, 30'000'000);
    EXPECT(m.rate() == 10'000'000);
    EXPECT(m.remaining(total) == 7);
    EXPECT(in::progress_text({in::Phase::Download, 30'000'000, total}, m) == "0.0 GB of 0.1 GB  10.0 MB/s  7 s left");
    // the window forgets the old speed: 30 s at 1 MB/s after the burst
    for (int s = 4; s <= 34; s++) m.add(s, 30'000'000 + static_cast<uint64_t>(s - 3) * 1'000'000);
    EXPECT(m.rate() < 1'100'000 && m.rate() > 900'000);
    m.add(35, 0);  // a restart from zero drops the samples
    EXPECT(m.rate() == 0);
    m.reset();
    EXPECT(m.rate() == 0);
}

void test_sink() {
    int n = 0;
    in::report_progress({});  // no sink: nothing happens
    in::set_progress_sink([&](const in::Progress& p) { n += static_cast<int>(p.done); });
    in::report_progress({in::Phase::Verify, 5, 10});
    EXPECT(n == 5);
    in::set_progress_sink({});
    in::report_progress({in::Phase::Verify, 5, 10});
    EXPECT(n == 5);
}

void test_what_next() {
    const std::vector<std::string> plain = in::what_next_lines({}, {"claude", "pi"});
    EXPECT(plain.size() == 7);
    EXPECT(plain[0].starts_with("tona ") && plain[1].starts_with("tona status") &&
           plain[2] == "Code with an agent against Tona");
    EXPECT(plain[3].starts_with("tona claude ~/project ") && plain[3].ends_with("Claude Code"));
    EXPECT(plain[4].ends_with("OpenCode (not installed)") && plain[6].ends_with("Pi"));
    const std::vector<std::string> none = in::what_next_lines({}, {});
    EXPECT(none.size() == 8 && none[7].contains("docs/use/client.md"));
    // every description starts in the same column
    for (size_t i : {0, 1, 3, 4, 5, 6}) EXPECT(plain[i].size() > 24 && plain[i][23] == ' ' && plain[i][24] != ' ');
    const std::vector<std::string> hint = in::path_hint("/home/u/.local/bin", "/usr/bin");
    EXPECT(!hint.empty());
    const std::vector<std::string> with = in::what_next_lines(hint, {"claude"});
    EXPECT(with.size() == 7 + 1 + hint.size() && with[7].empty() && with[8] == hint[0]);
    // the lookup uses PATH: a fake folder with one executable
    const std::string dir = "/tmp/tona-test-agents-" + std::to_string(::getpid());
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/opencode") << "#!/bin/sh\n";
    ::chmod((dir + "/opencode").c_str(), 0755);
    ::setenv("PATH", dir.c_str(), 1);
    EXPECT(in::installed_agents() == std::vector<std::string>{"opencode"});
    std::filesystem::remove_all(dir);
    EXPECT(in::path_hint("/home/u/.local/bin", "/usr/bin:/home/u/.local/bin").empty());
}

}  // namespace

int main() {
    return run("progress", [] {
        test_formatting();
        test_rate_and_eta();
        test_sink();
        test_what_next();
        return 0;
    });
}
