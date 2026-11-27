// The flow of the installer without a screen: which page follows which, what is pre-selected per task, language and
// hardware class, and what a choice means for the settings. The terminal pages (tui*.cpp) and the commands without a
// screen (cli.h) both call this, so a choice on a page and the same choice on the command line give the same plan.
#pragma once

#include <string>
#include <vector>

#include "core/hw.h"
#include "core/plan.h"
#include "offer/offer.h"

namespace tona::install {

enum class Page { Machine, Tasks, Languages, Models, Room, Go };

struct HwMatch {
    std::string cls;     // id of a class of the offer data
    bool exact = false;  // false: this machine is no class of the offer data, the numbers are those of `cls`
};

// The class of the offer data this machine belongs to; else the measured class with exact = false
HwMatch match_hw_class(const Hardware& h);

// What the user chose. All ids are ids of the offer data; `models` is in the order of the offer (task order, the
// recommendation before the alternative), so the first one is the model the server starts with.
struct Selection {
    std::vector<std::string> tasks, languages, models;
};

bool contains(const std::vector<std::string>& v, const std::string& id);

// Coding and knowledge with their recommendations: what Enter on every page gives. The alternatives stay off.
Selection default_selection(const std::string& cls);

// A task that is "coming" or unknown cannot be switched on (returns false). Switching a task on adds its
// recommendation, switching it off removes the models only that task offered.
bool toggle_task(Selection& sel, const std::string& task, const std::string& cls);
void toggle_language(Selection& sel, const std::string& language);
// Only a model that a chosen task offers can be switched
bool toggle_model(Selection& sel, const std::string& model, const std::string& cls);

struct TaskModels {
    const offer::Task* task = nullptr;
    const offer::Offer* offer = nullptr;
    const offer::ModelCard* recommended = nullptr;
    const offer::ModelCard* alternative = nullptr;  // nullptr: none
};
// One entry per chosen task that has an offer on this class
std::vector<TaskModels> task_models(const Selection& sel, const std::string& cls);

// The page with languages exists only for coding (documents and audio come with their models)
bool needs_languages(const Selection& sel);
Page next_page(Page from, const Selection& sel);
Page prev_page(Page from, const Selection& sel);
// The pages of this selection, in order (the breadcrumb of the design)
std::vector<Page> page_order(const Selection& sel);

// The model the server starts with: the first chosen one; empty if none is chosen
std::string primary_model(const Selection& sel);
// The drafter of a model: only the Coder has an MTP head in the catalog; the others run without (models.toml)
std::string default_drafter(const std::string& model);

// Per chosen language the model of the coding offer that is strongest for it
struct LanguageProposal {
    std::string language, model;
    offer::Level level = offer::Level::Open;
    offer::Basis basis = offer::Basis::None;
};
std::vector<LanguageProposal> language_proposals(const Selection& sel, const std::string& cls);

// The models the language page proposed, once each, in the order of the languages; the model page switches them on
std::vector<std::string> proposed_models(const Selection& sel, const std::string& cls);

// Models of the offer that are not chosen but strong in a chosen language where no chosen model is: the hint under
// the model list. Never switched on by itself, a model that large is the user's choice.
std::vector<std::string> models_to_consider(const Selection& sel, const std::string& cls);

// The id of a model from what a user types: catalog id, card name ("coder"), or the start of an id; empty if unknown
std::string resolve_model(const std::string& name);

// Settings for a selection: model, models, drafter, languages
Settings apply_selection(Settings s, const Selection& sel);

// Comma separated ids; empty parts are dropped
std::vector<std::string> split_list(const std::string& csv);
std::string join_list(const std::vector<std::string>& v);

// A selection from names (command line)
struct NamedChoice {
    std::vector<std::string> tasks, languages, models;
};
struct FlowError {
    int code = 0;  // an ExitCode (exit_codes.h)
    std::string text;
};
// Nothing named: the default selection. Models named: they decide, and the tasks follow from them. Else the named
// tasks with their recommendations. False with `err` set if a name is unknown or a task is not available yet.
bool selection_from_names(const NamedChoice& in, const std::string& cls, Selection& out, FlowError& err);

// The selection that belongs to settings (an existing configuration, or the defaults of a fresh machine): the models
// of the settings with the tasks that offer them. Falls back to the default selection if none of them is offered.
Selection selection_from_settings(const Settings& s, const std::string& cls);

}  // namespace tona::install
