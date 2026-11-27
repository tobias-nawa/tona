// Words for the numbers of the offer data, shared by the terminal pages and the plan on the command line. Every number
// is shown with its context and its basis word (measured, provisional, vendor claim), never bare.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/hw.h"
#include "core/plan.h"
#include "core/steps.h"
#include "offer/offer.h"

namespace tona::install {

// The context of the speed in a list line: a coding agent works with 16K tokens in the window most of the time, and
// the three measured points (2K, 16K, 64K) are all shown by the details
constexpr uint32_t kListContext = 16384;

// "2K", "16K", "64K"
std::string context_label(uint32_t tokens);

// The basis word: measured / provisional / vendor / independent / open
std::string basis_text(offer::Basis b);

// Speed of a model on this machine as the screens show it: "55–75 tok/s" (decode at 16K, a range), "unknown" when the
// module has no estimate. No exact number is shown anywhere (docs/use/installer.md, "Speed estimate").
std::string speed_text(const Estimate& e);
// The one line of the model page that says which rate this is and for which machine
std::string speed_footer(bool mac);
// "this PC" or "this Mac": what the screens call the machine the user sits at
std::string this_machine(bool mac);
// "other PCs" or "other computers": the machines that may connect with a key
std::string other_machines(bool mac);
// The class of the machine as the plan and the machine page show it. exact: the class label alone. Not exact: the
// label with the reason (a PC that is no class of its own, a Mac below the smallest Mac class)
std::string class_text(const offer::HwClass& c, bool exact, const Hardware& h);
std::string class_title(const offer::HwClass& c, bool exact, bool mac);
// Decode under the chosen agents and context: "57-79 tok/s each, 110-150 together" (together only for more than one
// agent), "unknown" without an estimate
std::string load_speed_text(const Estimate& e);
// The same in the words of the command line: "decode 55-75 tok/s at 16K"
std::string speed_plain(const Estimate& e);
// Lines of the details (key i): decode and prefill range, the inputs used, the unknown ones, the anchor
std::vector<std::string> speed_detail_lines(const Estimate& e);

// Labels of the languages where the card says strong / weak / open, joined by ", "
std::string languages_with(const offer::ModelCard& c, offer::Level level);

// "stark: Rust, Go   schwach: Python" without the empty parts; falls back to the written strengths
struct VerdictParts {
    std::string strong, weak;  // "strong: Python, C#" and "weak: Go"; empty if there is none
};
VerdictParts verdict_parts(const offer::ModelCard& c);
std::string verdict_line(const offer::ModelCard& c);

// Lines of the details (key i): license and the verdicts with their sources; the speed comes from speed_detail_lines
std::vector<std::string> detail_lines(const offer::ModelCard& c);

// A step of the plan with model names as the model page has them ("Download and verify:
// Coder · 58 GB"); the title of a step this does not know
// With a models folder the model steps say what is there: "Verify: Coder · 62.5 GB (already here)" instead of
// "Download and verify", and the drafter step is empty (the size and the status of the model include it)
std::string step_text(const Step& st, const std::string& models_dir = "");

// "strong" for a measured verdict, "weak (provisional)" otherwise: a basis word only where it is not the usual one
std::string level_text(offer::Level l, offer::Basis b);

}  // namespace tona::install
