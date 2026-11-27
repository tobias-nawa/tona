// The data feed of the centre: the poller thread of tona (client/poller.cpp) keeps a `Shared` current; this
// turns what it holds into the typed fields of the UI state. Separate from the poller so the screens are tested with a
// hand-filled Shared and a recorded /status JSON, and from the views so they never see the lock.
#pragma once

#include "client/term.h"
#include "client/ui.h"

namespace tona::client {

// Copies status, rates, projects and log from the poller's state into `ui` (under its lock).
// The rate series follows ui.project_view (0: all projects added up).
void fill_from_feed(UiState& ui, top::Shared& sh);

}  // namespace tona::client
