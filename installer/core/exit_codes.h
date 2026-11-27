// Exit codes of tona setup. Fixed: scripts rely on them, docs/use/installer.md and --help list them.
#pragma once

namespace tona::install {

enum ExitCode {
    kExitOk = 0,
    kExitFailed = 1,        // a step failed, or errors in the plan have to be fixed first
    kExitUsage = 2,         // unknown option or name, no terminal and nothing named
    kExitNoHardware = 3,    // no backend for this hardware
    kExitNoSpace = 4,       // not enough free space for what is named
    kExitDownload = 5,      // a download failed (the next run resumes it)
    kExitTaskComing = 6,    // a named task exists but cannot be installed yet
    kExitMissingTools = 7,  // a required package is missing
};

}  // namespace tona::install
