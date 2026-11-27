// Small helpers for the installer: read files, find programs, run commands.
#pragma once

#include <functional>
#include <optional>
#include <string>

namespace tona::install {

std::string read_file(const std::string& path);  // empty if unreadable
std::string trim(const std::string& s);
std::string home();
// path of the running program with its links resolved; empty if unknown
std::string self_exe();
// path of a program in PATH
std::optional<std::string> which(const std::string& cmd);

struct RunResult {
    int status = -1;  // exit status of the command (-1: could not start)
    std::string out;  // stdout and stderr
};
// Runs a command through sh -c; on_line gets each output line as soon as it is complete (\r also ends a line, so
// progress bars come through)
RunResult run(const std::string& cmd, const std::function<void(const std::string&)>& on_line = {});
// single-quoted word for the shell
std::string quote(const std::string& s);

}  // namespace tona::install
