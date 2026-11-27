#include "client/feed.h"

namespace tona::client {

void fill_from_feed(UiState& ui, top::Shared& sh) {
    std::lock_guard lk(sh.mu);
    ui.server = parse_status(sh.status, sh.up, sh.error);
    ui.projects = sh.projects;
    if (ui.project_view > (int)ui.projects.size()) ui.project_view = 0;  // a project cannot disappear, but be safe
    ui.rates.clear();
    for (const auto& parts : sh.rate) {
        double sum = 0;
        for (const auto& [project, rate] : parts)
            if (ui.project_view == 0 || project == ui.project_view - 1) sum += rate;
        ui.rates.push_back(sum);
    }
    ui.log = sh.log;
}

}  // namespace tona::client
