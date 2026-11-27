// Formatting, ANSI drawing and terminal handling of tona.
#include "client/term.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <format>

#include "update/updater.h"

namespace tona::top {

double jnum(const json& j, const char* key, double fallback) {
    if (!j.is_object()) return fallback;
    const auto it = j.find(key);
    return it != j.end() && it->is_number() ? it->get<double>() : fallback;
}

std::string jstr(const json& j, const char* key, const std::string& fallback) {
    if (!j.is_object()) return fallback;
    const auto it = j.find(key);
    return it != j.end() && it->is_string() ? it->get<std::string>() : fallback;
}

std::string paint(const char* color, const std::string& text) { return color + text + kReset; }

namespace {

// Index of the final byte of the escape sequence that starts at s[i]
size_t escape_end(const std::string& s, size_t i) {
    while (i < s.size() && (s[i] < '@' || s[i] > '~' || s[i] == '[')) i++;
    return i;
}

bool starts_character(char c) { return ((unsigned char)c & 0xc0) != 0x80; }

}  // namespace

int width(const std::string& s) {
    int w = 0;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\x1b') {
            i = escape_end(s, i);
            continue;
        }
        if (starts_character(s[i])) w++;
    }
    return w;
}

std::string fit(const std::string& s, int w) {
    std::string o;
    int n = 0;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\x1b') {
            const size_t begin = i;
            i = escape_end(s, i);
            o.append(s, begin, i - begin + 1);
            continue;
        }
        if (starts_character(s[i])) {
            if (n == w) break;
            n++;
        }
        o += s[i];
    }
    return o + kReset + std::string(std::max(0, w - n), ' ');
}

std::string right(const std::string& s, int w) { return std::string(std::max(0, w - width(s)), ' ') + s; }

std::string repeat(const char* glyph, int n) {
    std::string o;
    for (int i = 0; i < n; i++) o += glyph;
    return o;
}

namespace {

termios saved_termios;
bool terminal_raw = false;

void write_terminal(const char* s) { (void)!write(STDOUT_FILENO, s, std::strlen(s)); }

void on_signal(int sig) {
    term_restore();
    signal(sig, SIG_DFL);
    raise(sig);
}

}  // namespace

void term_restore() {
    if (!terminal_raw) return;
    terminal_raw = false;
    tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios);
    write_terminal("\x1b[0m\x1b[?25h\x1b[?1049l");
}

void term_setup() {
    tcgetattr(STDIN_FILENO, &saved_termios);
    termios t = saved_termios;
    t.c_lflag &= ~(ICANON | ECHO);
    t.c_cc[VMIN] = 0, t.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &t);
    terminal_raw = true;
    std::atexit(term_restore);
    for (int s : {SIGINT, SIGTERM, SIGHUP, SIGQUIT}) signal(s, on_signal);
    write_terminal("\x1b[?1049h\x1b[?25l");
}

std::string run(const std::string& cmd, std::string* all) {
    std::string out;
    int fds[2];
    if (pipe(fds) == 0) {
        const pid_t pid = fork();
        if (pid == 0) {
            // child: stdin from /dev/null, stdout and stderr into the pipe
            const int null_fd = open("/dev/null", O_RDONLY);
            if (null_fd >= 0) dup2(null_fd, STDIN_FILENO);  // without /dev/null the child keeps our stdin
            dup2(fds[1], STDOUT_FILENO);
            dup2(fds[1], STDERR_FILENO);
            close(fds[0]);
            close(fds[1]);
            execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }
        close(fds[1]);
        char b[512];
        for (ssize_t n; (n = read(fds[0], b, sizeof b)) > 0;) out.append(b, (size_t)n);
        close(fds[0]);
        if (pid > 0) waitpid(pid, nullptr, 0);
    }
    if (all) *all = out;
    while (!out.empty() && out.back() == '\n') out.pop_back();
    const size_t nl = out.rfind('\n');
    return nl == std::string::npos ? out : out.substr(nl + 1);
}

int Shared::project(const std::string& name) {
    const std::string n = name.empty() ? "–" : name;
    for (size_t i = 0; i < projects.size(); i++)
        if (projects[i] == n) return (int)i;
    projects.push_back(n);
    return (int)projects.size() - 1;
}

namespace {

constexpr auto kConfirmWindow = std::chrono::seconds(5);

// First press arms, the same key again within the window confirms
bool confirmed(char key, KeyConfirm& c) {
    const auto now = std::chrono::steady_clock::now();
    if (c.key == key && now - c.at < kConfirmWindow) {
        c.key = 0;
        return true;
    }
    c = {key, now};
    return false;
}

}  // namespace

std::string update_key(tona::update::Updater* updater, char key, KeyConfirm& confirm) {
    if (!updater || !updater->view().enabled) return "Updates: this is a source checkout, use git pull and just build";
    const tona::update::UpdateView v = updater->view();
    if (key == 'c') {
        if (!confirmed(key, confirm))
            return v.checks_on
                       ? "Press c again to turn update checks off"
                       : "Press c again to turn update checks on (contacts GitHub, nothing else leaves this machine)";
        updater->set_checks(!v.checks_on);
    } else if (key == 'r') {
        if (!confirmed(key, confirm)) return "Press r again to go back to the previous version (restarts the server)";
        updater->roll_back();
    } else if (v.available) {
        updater->install();
    } else {
        return v.checks_on ? "No update known; checks run at start and once a day" : "Update checks are off, press c";
    }
    return "";
}

}  // namespace tona::top
