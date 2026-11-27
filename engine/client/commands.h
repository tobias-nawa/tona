// The commands without a screen: tona status | models | use | config | doctor. Plain text when nothing else is asked,
// --json for data; the exit codes are fixed (below). They work over SSH and in scripts and share the logic of the
// screens: the same ServerView, offer, settings table and checks.
#pragma once

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#include "client/control.h"
#include "client/doctor.h"
#include "client/offer.h"
#include "client/server_view.h"
#include "client/settings.h"

namespace tona::client {

// Exit codes, documented in docs/use/client.md
enum ExitCode {
    kExitOk = 0,
    kExitFailed = 1,        // the action failed (message on stderr)
    kExitUsage = 2,         // unknown command, option or value
    kExitServerDown = 3,    // status: the server does not answer
    kExitNeedsConfirm = 4,  // use: conversations are running and neither --yes nor an answer was given
    kExitWarnings = 5       // doctor: at least one check is not ok
};

// What the commands need from the outside; tests replace each part
struct Services {
    Control* control = nullptr;
    std::function<ServerView()> status;                                  // GET /status once
    std::function<std::vector<Check>(const install::Settings&)> doctor;  // facts and checks
    std::function<Offer()> offer;  // the offer, with the installed flags in the catalog
    std::function<std::vector<install::ModelInfo>()> catalog;
    std::function<void(install::Settings&)> resolve;  // the context of "auto"; empty: this machine's backend module
};

struct Io {
    std::ostream* out = nullptr;
    std::ostream* err = nullptr;
    // Asks a yes/no question; empty function = no terminal to ask on
    std::function<bool(const std::string&)> ask;
};

// args without the program name ("status", "--json", ...). Returns an ExitCode.
int run_command(const std::vector<std::string>& args, const Env& env, Services& svc, Io& io);

// True for the words that run_command knows ("status", "models", "use", "config", "doctor", "help")
bool is_command(const std::string& word);

// The real services of this machine
Services system_services(const Env& env, Control& control);

// GET /status of a server once; `up` false and `error` filled when it does not answer
ServerView fetch_server_view(const std::string& url);

// The usage text
std::string usage_text();

}  // namespace tona::client
