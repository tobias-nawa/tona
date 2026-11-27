#include "core/sys.h"

#include <sys/wait.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace tona::install {

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

std::string home() {
    const char* h = std::getenv("HOME");
    return h ? h : "";
}

std::string self_exe() {
    std::error_code ec;
#ifdef __APPLE__
    // macOS has no /proc: the loader holds the path, and canonical resolves its links as readlink does on Linux
    uint32_t n = 0;
    _NSGetExecutablePath(nullptr, &n);
    std::string p(n, '\0');
    if (_NSGetExecutablePath(p.data(), &n) != 0) return {};
    p.resize(std::strlen(p.c_str()));
    const std::filesystem::path real = std::filesystem::canonical(p, ec);
    return ec ? std::string() : real.string();
#else
    const std::filesystem::path real = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : real.string();
#endif
}

std::optional<std::string> which(const std::string& cmd) {
    const char* path = std::getenv("PATH");
    if (!path) return std::nullopt;
    std::istringstream dirs(path);
    for (std::string d; std::getline(dirs, d, ':');) {
        std::string p = (d.empty() ? "." : d) + "/" + cmd;
        if (access(p.c_str(), X_OK) == 0) return p;
    }
    return std::nullopt;
}

namespace {

// Starts "sh -c cmd" with stdout and stderr on a pipe; returns the read end and the child's pid (-1 on failure).
std::pair<int, pid_t> spawn(const std::string& cmd) {
    int fds[2];
    if (pipe(fds) != 0) return {-1, -1};
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]), close(fds[1]);
        return {-1, -1};
    }
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO), dup2(fds[1], STDERR_FILENO);
        close(fds[0]), close(fds[1]);
        execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    close(fds[1]);
    return {fds[0], pid};
}

// Reads the pipe to its end, handing out each complete line (\n or \r ends one).
void read_lines(int fd, RunResult& r, const std::function<void(const std::string&)>& on_line) {
    std::string line;
    char buf[4096];
    for (ssize_t n; (n = read(fd, buf, sizeof buf)) != 0;) {
        if (n < 0 && errno == EINTR) continue;  // a signal (e.g. terminal resize) must not end the read early
        if (n < 0) break;
        for (ssize_t i = 0; i < n; i++) {
            const char c = buf[i];
            r.out.push_back(c);
            if (c != '\n' && c != '\r') {
                line.push_back(c);
                continue;
            }
            if (on_line && !line.empty()) on_line(line);
            line.clear();
        }
    }
    if (on_line && !line.empty()) on_line(line);
}

}  // namespace

RunResult run(const std::string& cmd, const std::function<void(const std::string&)>& on_line) {
    RunResult r;
    const auto [fd, pid] = spawn(cmd);
    if (fd < 0) return r;
    read_lines(fd, r, on_line);
    close(fd);
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
    }
    r.status = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    return r;
}

std::string quote(const std::string& s) {
    std::string q = "'";
    for (char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return q + "'";
}

}  // namespace tona::install
