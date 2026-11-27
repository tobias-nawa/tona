#include "client/view.h"

#include <algorithm>
#include <format>

#include "client/format.h"
#include "client/term.h"
#include "client/text.h"

namespace tona::client {

namespace {

std::string tab_bar(const UiState& s) {
    static const char* keys[] = {"tab_start", "tab_models", "tab_agents", "tab_settings"};
    std::string bar = " " + bold("tona") + "   ";
    for (int i = 0; i < 4; i++) {
        const std::string label = std::format(" {} {} ", i + 1, tr(keys[i]));
        bar += (int)s.tab == i ? top::paint(top::kInverse, label) : label;
        bar += ' ';
    }
    return bar;
}

std::string conversations_text(int n) { return fill(tr(n == 1 ? "conv_one" : "conv_many"), fnum(n)); }

std::string status_line(const UiState& s) {
    const ServerView& v = s.server;
    std::string line = " " + tr("status_server") + " ";
    if (nothing_installed(s)) return line + paint_info(tr("status_ready")) + " · " + tr("status_no_model_installed");
    if (!v.up && restarting(s)) return line + paint_info(tr("status_restarting"));
    if (!v.up) return line + paint_bad(tr("status_down")) + " · " + tr("status_no_model");
    if (v.loading()) {
        line += paint_info(tr("status_loading") + " " + std::format("{:.0f} %", 100 * v.load_fraction()));
    } else if (!v.error.empty()) {
        return line + paint_bad(tr("status_load_failed"));
    } else {
        line += paint_ok(tr("status_up"));
    }
    const std::string name = loaded_name(s);
    if (!name.empty()) line += " · " + tr("status_model") + " " + bold(name);
    line += " · " + conversations_text(v.conversation_count());
    return line + " · " + tr("status_priority") + " " + tr("priority_" + get_value(s.config, "priority"));
}

KeyHints page_hints(const UiState& s) {
    switch (s.tab) {
        case Tab::Start:
            if (nothing_installed(s))
                return {{"↑↓", tr("k_choose")}, {"Enter", tr("k_continue")}, {"q", tr("k_close")}, {"?", tr("k_help")}};
            if (!s.server.up)
                return {{"↑↓", tr("k_choose")},
                        {"Enter", tr("k_run")},
                        {"l", tr("k_log")},
                        {"q", tr("k_close")},
                        {"?", tr("k_help")}};
            return {{"a", tr("k_agent")},   {"w", tr("k_switch")}, {"u", tr("k_update")},
                    {"r", tr("k_restart")}, {"q", tr("k_close")},  {"?", tr("k_help")}};
        case Tab::Models:
            return {{"↑↓", tr("k_choose")}, {"Enter", tr("k_switch_to")}, {"d", tr("k_download")},
                    {"x", tr("k_remove")},  {"f", tr("k_filter")},        {"?", tr("k_help")}};
        case Tab::Agents:
            return {{"Tab", tr("k_field")},   {"←→", tr("k_pick")},  {"w", tr("k_take_suggestion")},
                    {"Enter", tr("k_start")}, {"Esc", tr("k_back")}, {"?", tr("k_help")}};
        case Tab::Settings:
            return {{"↑↓", tr("k_row")}, {"←→", tr("k_change")},     {"Tab", tr("k_group")},
                    {"s", tr("k_save")}, {"z", tr("k_recommended")}, {"?", tr("k_help")}};
    }
    return {};
}

KeyHints footer_hints(const UiState& s) {
    if (s.modal == Modal::Help) return {{"Esc", tr("k_back")}};
    if (s.modal == Modal::Confirm) return {{"Enter", tr("k_do")}, {"Esc", tr("k_cancel")}};
    if (s.modal == Modal::Close)
        return {{"↑↓", tr("k_choose")}, {"Enter", tr("k_run")}, {"s", tr("k_stop_server")}, {"Esc", tr("k_cancel")}};
    return page_hints(s);
}

Lines content(const UiState& s, int rows) {
    if (s.modal == Modal::Help) return view_help(s, rows);
    switch (s.tab) {
        case Tab::Start:
            if (nothing_installed(s)) return view_empty(s, rows);
            return s.server.up ? view_start(s, rows) : view_server_down(s, rows);
        case Tab::Models:
            return view_models(s, rows);
        case Tab::Agents:
            return view_agents(s, rows);
        case Tab::Settings:
            return view_settings(s, rows);
    }
    return {};
}

void draw_dialog(const UiState& s, Lines& screen) {
    if (s.modal == Modal::Confirm) overlay(screen, dialog_confirm(s), 4, 6);
    if (s.modal == Modal::Close) overlay(screen, dialog_close(s), 4, 3);
}

}  // namespace

Lines render(const UiState& s) {
    if (s.width < kMinWidth || s.height < kMinHeight) return view_too_small(s);
    const int rows = s.height - 4;  // header, status line, message line, keys
    Lines screen;
    screen.push_back(tab_bar(s));
    screen.push_back(status_line(s));
    Lines body = content(s, rows);
    body.resize((size_t)rows);
    for (std::string& row : body) screen.push_back(std::move(row));
    screen.push_back(s.message.empty() ? "" : " " + paint_warn(s.message));
    screen.push_back(hint_row(footer_hints(s)));
    draw_dialog(s, screen);
    for (std::string& row : screen) row = top::fit(row, s.width);
    return screen;
}

}  // namespace tona::client
