// The pages that replace the Start page in special states: server not running, first start with nothing installed;
// and the help page and the "terminal too small" page
#include <algorithm>
#include <format>

#include "client/format.h"
#include "client/term.h"
#include "client/text.h"
#include "client/view.h"

namespace tona::client {

namespace {

std::string option_row(const UiState& s, int index, const std::string& text, const std::string& note = "") {
    const bool on = s.start_cursor == index;
    return std::string(on ? " ▸ " : "   ") + (on ? bold(text) : text) + (note.empty() ? "" : "    " + dim(note));
}

std::string check_row(const Check& c) {
    const std::string state = c.ok ? paint_ok(tr("check_ok")) : paint_warn(tr("check_warn"));
    std::string row = "   " + cell(tr("check_" + c.name), 16) + " " + top::fit(state, 6) + c.detail;
    if (!c.ok && !c.fix.empty()) row += "  " + dim("→ " + c.fix);
    return row;
}

const ModelCard* recommended_card(const UiState& s) {
    for (const ModelCard& c : s.offer.cards)
        if (!c.coming && !c.id.empty()) return &c;
    return nullptr;
}

const ModelCard* smaller_card(const UiState& s, const ModelCard& than) {
    const ModelCard* best = nullptr;
    for (const ModelCard& c : s.offer.cards)
        if (!c.coming && c.id != than.id && c.size_gb > 0 && c.size_gb < than.size_gb &&
            (!best || c.size_gb < best->size_gb))
            best = &c;
    return best;
}

}  // namespace

Lines view_server_down(const UiState& s, int rows) {
    Lines out;
    // A planned restart is not a failure: no list of checks and no options, only what is going on
    if (restarting(s)) return {" " + paint_info(tr("down_restarting"))};
    out.push_back(" " +
                  paint_bad(s.server.error.empty() ? tr("down_title") : fill(tr("down_title_why"), s.server.error)));
    out.push_back(section_row(tr("down_found")));
    for (const Check& c : s.checks) out.push_back(check_row(c));
    if (s.checks.empty()) out.push_back("   " + dim(tr("down_checking")));
    if (s.show_log) {
        out.push_back(section_row(tr("sec_log")));
        const int room = rows - (int)out.size();
        for (int i = std::max(0, (int)s.log.size() - room); i < (int)s.log.size(); i++)
            out.push_back("   " + dim(s.log[(size_t)i]));
        return out;
    }
    out.push_back(section_row(tr("down_options")));
    const int port = s.config.s.port;
    out.push_back(option_row(s, 0, fill(tr("down_free_port"), std::to_string(s.free_port ? s.free_port : port + 1))));
    out.push_back(option_row(s, 1, fill(tr("down_retry"), std::to_string(port))));
    out.push_back(option_row(s, 2, tr("down_log")));
    out.emplace_back("");
    out.push_back(" " + dim(tr("down_waiting")));
    out.push_back(" " + dim(tr("down_ssh")));
    return out;
}

Lines view_empty(const UiState& s, int) {
    Lines out;
    out.push_back(" " + bold(tr("empty_welcome")));
    out.push_back(" " + tr("empty_need_model"));
    out.emplace_back("");
    const ModelCard* rec = recommended_card(s);
    out.push_back(option_row(s, 0, tr("empty_choose"), tr("empty_choose_note")));
    out.push_back(option_row(s, 1, tr("empty_load"),
                             rec ? fill(tr("empty_load_note"), rec->name, fnum(rec->size_gb)) : tr("empty_load_none")));
    out.emplace_back("");
    if (rec)
        if (const ModelCard* small = smaller_card(s, *rec))
            out.push_back(" " + fill(tr("empty_tip"), small->name, fnum(small->size_gb)));
    out.emplace_back("");
    out.push_back(" " + tr("empty_after"));
    for (const char* key : {"empty_after_1", "empty_after_2", "empty_after_3"}) out.push_back("   " + tr(key));
    return out;
}

Lines view_help(const UiState& s, int) {
    static const char* tab_keys[] = {"tab_start", "tab_models", "tab_agents", "tab_settings"};
    Lines out;
    out.push_back(" " + bold(tr("help_title")) + dim(" · " + tr("help_tab") + " " + tr(tab_keys[(int)s.tab])));
    out.push_back(section_row(tr("help_here")));
    const auto row = [&](const std::string& key, const std::string& text, const std::string& note = "") {
        out.push_back("   " + bold(top::fit(key, 8)) + cell(text, 28) + (note.empty() ? "" : dim(note)));
    };
    if (s.tab == Tab::Start) {
        row("a", tr("k_agent")), row("w", tr("k_switch"), tr("help_asks")),
            row("u", tr("help_update"), tr("help_asks"));
        row("r", tr("help_restart"), tr("help_asks")), row("p", tr("help_project")), row("l", tr("k_log"));
    } else if (s.tab == Tab::Models) {
        row("Enter", tr("k_switch_to"), tr("help_asks")), row("d", tr("k_download")),
            row("x", tr("k_remove"), tr("help_asks"));
        row("f t m", tr("help_filters"));
    } else if (s.tab == Tab::Agents) {
        row("Tab", tr("k_field")), row("←→ Space", tr("k_pick")), row("w", tr("k_take_suggestion")),
            row("Enter", tr("k_start"));
    } else {
        row("↑↓", tr("k_row")), row("←→", tr("k_change")), row("Tab", tr("k_group")), row("s", tr("k_save")),
            row("z", tr("k_recommended"));
    }
    out.push_back(section_row(tr("help_everywhere")));
    // Two entries per row so that the page fits 80 x 24
    const auto row2 = [&](const std::string& k1, const std::string& t1, const std::string& k2, const std::string& t2) {
        out.push_back("   " + bold(top::fit(k1, 8)) + cell(t1, 24) + bold(top::fit(k2, 8)) + t2);
    };
    row2("↑↓ j k", tr("help_move"), "Enter", tr("help_main"));
    row2("Esc ⌫", tr("help_back"), "Space", tr("help_toggle"));
    row2("1-4", tr("help_tabs"), "Tab", tr("help_next_field"));
    row2("?", tr("help_this"), "q", tr("help_close"));
    out.push_back(section_row(tr("help_words")));
    for (const char* key : {"help_w_token", "help_w_writes", "help_w_reads"}) out.push_back("   " + tr(key));
    for (const std::string& l : wrap_lines(tr("help_w_measured"), 74)) out.push_back("   " + l);
    out.push_back(" " + dim(tr("help_q_asks")));
    return out;
}

Lines view_too_small(const UiState& s) {
    Lines out;
    out.push_back(" " + bold(tr("small_title")));
    out.emplace_back("");
    out.push_back(" " + fill(tr("small_need"), std::to_string(kMinWidth), std::to_string(kMinHeight)));
    out.push_back(" " + fill(tr("small_now"), std::to_string(s.width), std::to_string(s.height)));
    for (const std::string& l : wrap_lines(tr("small_hint"), std::max(s.width - 2, 10))) out.push_back(" " + l);
    out.emplace_back("");
    out.push_back(" " + bold("q") + " " + dim(tr("k_close")));
    out.resize((size_t)std::max(s.height, 1));
    for (std::string& row : out) row = top::fit(row, std::max(s.width, 1));
    return out;
}

}  // namespace tona::client
