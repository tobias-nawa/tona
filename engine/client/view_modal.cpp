// The dialogs drawn over a page: ask before switching, restarting or updating (and say which conversations are
// affected), and the Close dialog
#include <algorithm>

#include "client/format.h"
#include "client/term.h"
#include "client/text.h"
#include "client/view.h"

namespace tona::client {

namespace {

constexpr int kDialogInner = 66;

std::string conversation_line(const Conversation& c) {
    const std::string who = c.agent.empty() ? "–" : c.agent;
    const std::string what = c.state == "decode" ? tr("conv_writes_now") : tr("conv_reads");
    return "  " + cell(who, 10) + " " + cell(project_label(c.project), 16) + " " + what;
}

std::string title_for(const UiState& s) {
    const ModelCard* c = s.offer.find(s.ask_arg);
    switch (s.ask) {
        case Ask::SwitchModel:
            return fill(tr("ask_switch_title"), c ? c->name : s.ask_arg);
        case Ask::Restart:
            return tr("ask_restart_title");
        case Ask::Update:
            return fill(tr("ask_update_title"), s.update.version);
        case Ask::Remove:
            return fill(tr("ask_remove_title"), c ? c->name : s.ask_arg);
        case Ask::StopServer:
            return tr("ask_stop_title");
        case Ask::None:
            break;
    }
    return "";
}

std::string effect_for(const UiState& s) {
    switch (s.ask) {
        case Ask::SwitchModel:
            return tr("ask_switch_effect");
        case Ask::Restart:
            return tr("ask_restart_effect");
        case Ask::Update:
            return tr("ask_update_effect");
        case Ask::StopServer:
            return tr("ask_stop_effect");
        case Ask::Remove: {
            const ModelCard* c = s.offer.find(s.ask_arg);
            return fill(tr("ask_remove_effect"), fnum(c ? c->size_gb : 0));
        }
        case Ask::None:
            break;
    }
    return "";
}

// The release notes: control characters replaced (the text is drawn into a terminal), each line wrapped to the dialog
// width (box_lines would cut it), at most `max_rows` rows; "..." marks what is left out
Lines changelog_lines(const std::string& text, size_t max_rows) {
    Lines rows;
    bool cut = false;
    std::string line;
    const auto flush = [&] {
        // Blank lines of the Markdown source would only use up the few rows
        for (std::string& w : wrap_lines(line, kDialogInner - 4)) {
            if (w.empty()) continue;
            if (rows.size() < max_rows)
                rows.push_back("  " + w);
            else
                cut = true;
        }
        line.clear();
    };
    for (const char c : text) {
        if (c == '\n')
            flush();
        else
            line += static_cast<unsigned char>(c) < 0x20 ? ' ' : c;
    }
    if (!line.empty()) flush();
    if (cut) rows.emplace_back("  ...");
    return rows;
}

bool affects_conversations(Ask ask) { return ask != Ask::Remove && ask != Ask::None; }

}  // namespace

Lines dialog_confirm(const UiState& s) {
    Lines body{""};
    const std::vector<Conversation> running = running_conversations(s);
    if (affects_conversations(s.ask) && !running.empty()) {
        body.push_back(tr("ask_running"));
        for (const Conversation& c : running) body.push_back(conversation_line(c));
        body.emplace_back("");
    } else if (affects_conversations(s.ask)) {
        body.push_back(tr("ask_none_running"));
    }
    if (s.ask == Ask::Update) {
        body.push_back(fill(tr("ask_update_size"), fdec(s.update.download_mb)));
        for (std::string& l : changelog_lines(s.update.changelog, 8)) body.push_back(std::move(l));
    }
    for (std::string& l : wrap_lines(effect_for(s), kDialogInner - 2)) body.push_back(std::move(l));
    body.emplace_back("");
    body.push_back(bold("Enter") + " " + tr("k_do") + "   " + bold("Esc") + " " + tr("k_cancel"));
    return box_lines(title_for(s), body, kDialogInner);
}

Lines dialog_close(const UiState& s) {
    const Memory& m = s.server.mem;
    const std::vector<int> options = close_options(s);
    const int n = (int)running_conversations(s).size();
    Lines body{"", tr("close_explain"), tr("close_explain2"),
               fill(tr("close_memory"), fnum(m.vram_used / 1e9), fnum(m.ram_tona / 1e9)), ""};
    for (size_t i = 0; i < options.size(); i++) {
        const bool on = (int)i == s.modal_cursor;
        static const char* text[] = {"close_keep", "close_stop", "close_cancel"};
        static const char* key[] = {"Enter", "s", "Esc"};
        const std::string line = std::string(on ? "▸ " : "  ") + cell(tr(text[options[i]]), 52) + key[options[i]];
        body.push_back(on ? paint_selected(line, kDialogInner + 4) : line);
        if (options[i] == 1 && n > 0) body.push_back("    " + paint_warn(fill(tr("close_cut"), fnum(n))));
    }
    body.emplace_back("");
    body.push_back(std::string(s.remember_choice ? "[x] " : "[ ] ") + tr("close_remember") + dim("  (Space)"));
    body.push_back("    " + dim(tr("close_remember_where")));
    return box_lines(tr("close_title"), body, kDialogInner + 6);
}

}  // namespace tona::client
