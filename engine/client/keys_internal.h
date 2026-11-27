// Shared between keys.cpp (keys of every page, dialogs) and keys_pages.cpp (the keys of each page)
#pragma once

#include "client/settings.h"
#include "client/ui.h"

namespace tona::client {

// Handles a key of the current page; returns true if the page used it (then `out` is what to do)
bool handle_page(UiState& s, Key key, Action& out);
// `w` on every page: choose a model (on the Agents page: take the suggestion)
Action switch_key(UiState& s);
// Opens a Confirm dialog for `ask`
Action ask_for(UiState& s, Ask ask, const std::string& arg);

}  // namespace tona::client
