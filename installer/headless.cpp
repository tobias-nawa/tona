// tona setup without a screen: --print, --dry-run, --json, --yes, --config. It builds the plan from the same start
// and the same choices as the pages (core/start.h) and runs the steps one by one.
#include "headless.h"

#include <cstdio>
#include <string>

#include "core/cli_plan.h"
#include "core/exit_codes.h"
#include "core/release_steps.h"
#include "core/shell_path.h"
#include "core/start.h"

namespace tona::install {

namespace {

// The exit code of a step that failed: a download has its own, so a script can retry only that
int failed_code(const Step& st) { return st.title.starts_with("Fetch and verify") ? kExitDownload : kExitFailed; }

// Runs the steps one by one; stops at the first failure.
int run_steps(const std::vector<Step>& st, const std::string& repo) {
    for (size_t i = 0; i < st.size(); i++) {
        if (st[i].done) continue;
        std::printf("\n== %zu/%zu %s\n", i + 1, st.size(), st[i].title.c_str());
        std::fflush(stdout);
        const int r = run_step(st[i], repo, [](const std::string& l) {
            std::printf("%s\n", l.c_str());
            std::fflush(stdout);
        });
        if (r != 0) {
            std::fprintf(stderr, "Step failed (status %d): %s\n", r, st[i].cmd.c_str());
            return failed_code(st[i]);
        }
    }
    return kExitOk;
}

const char* done_text(bool release, bool macos) {
    if (release && macos)
        return "\nDone. Start the server with tona start (stop: tona stop). Start at login: tona autostart on.\n";
    return release ? "\nDone. Start the server with tona (also updates) or tona start. Start at login: tona autostart "
                     "on.\n"
                   : "\nDone. Start: tona claude (Claude Code with the engine), control centre: tona\n";
}

}  // namespace

int run_headless(const Options& o, const Target& t, bool dry) {
    Start st;
    FlowError err;
    if (!make_start(o, t, false, st, err)) {
        std::fprintf(stderr, "tona setup: %s\n", err.text.c_str());
        return err.code;
    }
    PlanDoc plan = make_plan(st.h, st.module, st.selection, st.settings, t.release, models_dir_of(t));
    const PathEnv env = current_path_env();
    if (st.module) {
        plan.steps = plan_steps(st.h, st.settings, t);
        for (Step& f : finish_steps(st.settings, t, env)) plan.steps.push_back(std::move(f));
    }
    const int code = plan_exit_code(plan);
    std::fputs((o.json ? plan_json(plan) : plan_text(plan)).c_str(), stdout);
    if (dry) return code;
    if (code != kExitOk) {
        std::fprintf(stderr, "\nAborted: fix the errors above first, then run again.\n");
        return code;
    }
    const int r = run_steps(plan.steps, t.dir());
    if (r == kExitOk) {
        std::fputs(done_text(t.release, t.macos), stdout);
        const std::string bin = t.bin_dir.empty() ? env.home + "/.local/bin" : t.bin_dir;
        if (st.settings.shell)
            for (const std::string& line : path_advice(env, bin, st.settings.modify_path))
                std::printf("%s\n", line.c_str());
    }
    return r;
}

}  // namespace tona::install
