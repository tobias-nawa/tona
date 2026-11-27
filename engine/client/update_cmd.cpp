#include "client/update_cmd.h"

#include <algorithm>
#include <format>
#include <ostream>
#include <sstream>
#include <thread>

namespace tona::client {

namespace {

using update::UpdateView;

void wait_idle(const update::Updater& u, std::chrono::milliseconds poll, const std::function<void()>& tick = {}) {
    while (u.busy()) {
        if (tick) tick();
        std::this_thread::sleep_for(poll);
    }
}

// The lines of the changelog, at most `n`: what a user decides on
std::string first_lines(const std::string& text, int n) {
    std::istringstream in(text);
    std::string line, out;
    for (int i = 0; i < n && std::getline(in, line); i++) out += "  " + line + "\n";
    return out;
}

// Prints each new message of the updater once, and the download in steps of 10 %
class Progress {
public:
    Progress(const update::Updater& u, std::ostream& out) : u_(u), out_(out) {}
    void operator()() {
        const UpdateView v = u_.view();
        if (v.phase == UpdateView::Phase::Downloading && v.download_bytes > 0) {
            const int step = static_cast<int>(10 * v.done_bytes / v.download_bytes);
            if (step != step_) out_ << "downloaded " << 10 * step << " %\n", step_ = step;
        }
        if (!v.message.empty() && v.message != last_) out_ << v.message << "\n", last_ = v.message;
    }

private:
    const update::Updater& u_;
    std::ostream& out_;
    int step_ = -1;
    std::string last_;
};

int finish(const update::Updater& u, Io& io) {
    const UpdateView v = u.view();
    if (v.failed) return *io.err << "tona: " << v.message << "\n", kExitFailed;
    *io.out << v.message << "\n";
    return kExitOk;
}

// True: go on. False: stop with `code` (the answer was no, or there is nobody to ask)
bool confirmed(Io& io, bool yes, const std::string& question, int& code) {
    if (yes) return true;
    if (!io.ask)
        return *io.err << "tona: " << question << " Run it with --yes to answer yes.\n", code = kExitNeedsConfirm,
                                                                                         false;
    if (io.ask(question)) return true;
    return code = kExitOk, false;
}

int roll_back(update::Updater& u, bool yes, Io& io, std::chrono::milliseconds poll) {
    int code = kExitOk;
    if (!confirmed(io, yes, "Go back to the previous version and restart the server?", code)) return code;
    u.roll_back();
    Progress progress(u, *io.out);
    wait_idle(u, poll, std::ref(progress));
    progress();
    return finish(u, io);
}

}  // namespace

int run_update(update::Updater& u, const std::vector<std::string>& args, Io& io, std::chrono::milliseconds poll) {
    bool yes = false, rollback = false;
    for (const std::string& a : args) {
        if (a == "--yes")
            yes = true;
        else if (a == "--rollback")
            rollback = true;
        else
            return *io.err << "usage: tona update [--yes] [--rollback]\n", kExitUsage;
    }
    if (!u.view().enabled) {
        *io.err << "tona: updates are for a release installation; a checkout updates with git pull and just build\n";
        return kExitFailed;
    }
    if (rollback) return roll_back(u, yes, io, poll);
    *io.out << "Checking for a new release\n";
    u.check_now();
    wait_idle(u, poll);
    const UpdateView v = u.view();
    if (v.failed) return *io.err << "tona: " << v.message << "\n", kExitFailed;
    if (!v.available) return *io.out << "Up to date\n", kExitOk;
    *io.out << std::format("Version {} is available, {} MB{}\n", v.version, v.download_bytes / 1000000,
                           v.security ? " (security update)" : "")
            << first_lines(v.changelog, 12);
    int code = kExitOk;
    if (!confirmed(io, yes, "Install it and restart the server?", code)) return code;
    u.install();
    Progress progress(u, *io.out);
    wait_idle(u, poll, std::ref(progress));
    progress();
    return finish(u, io);
}

int run_update_command(const Env& env, const std::vector<std::string>& args, Io& io) {
    std::string note;
    const std::unique_ptr<update::Updater> u = update::make_default_updater(env.url(), note);
    if (!u) return *io.err << "tona: " << note << "\n", kExitFailed;
    return run_update(*u, args, io);
}

}  // namespace tona::client
