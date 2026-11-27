// The keys of each page of the control centre
#include <algorithm>

#include "client/keys_internal.h"
#include "client/text.h"

namespace tona::client {

namespace {

Action act(ActionKind k, std::string arg = "") { return {k, std::move(arg)}; }

void move_cursor(int& cursor, Key key, int count) {
    if (key.code == KeyCode::Up || (key.code == KeyCode::Char && key.ch == 'k')) cursor = std::max(0, cursor - 1);
    if (key.code == KeyCode::Down || (key.code == KeyCode::Char && key.ch == 'j'))
        cursor = std::min(std::max(0, count - 1), cursor + 1);
}

bool is_char(Key key, char c) { return key.code == KeyCode::Char && key.ch == c; }
bool is_move(Key key) {
    return key.code == KeyCode::Up || key.code == KeyCode::Down || is_char(key, 'j') || is_char(key, 'k');
}

// Start

// Server down: three ways forward (design: start on a free port, try again, read the log)
bool start_server_down(UiState& s, Key key, Action& out) {
    if (is_move(key)) return move_cursor(s.start_cursor, key, 3), true;
    if (is_char(key, 'l')) return out = act(ActionKind::ToggleLog), true;
    if (key.code != KeyCode::Enter) return false;
    static const ActionKind kinds[] = {ActionKind::StartOnFreePort, ActionKind::StartServer, ActionKind::ToggleLog};
    out = act(kinds[std::clamp(s.start_cursor, 0, 2)]);
    return true;
}

// First start: nothing installed
bool start_empty(UiState& s, Key key, Action& out) {
    if (is_move(key)) return move_cursor(s.start_cursor, key, 2), true;
    if (key.code != KeyCode::Enter) return false;
    out = act(s.start_cursor == 0 ? ActionKind::RunInstaller : ActionKind::DownloadRecommended);
    return true;
}

bool start_running(UiState& s, Key key, Action& out) {
    if (key.code != KeyCode::Char) return false;
    switch (key.ch) {
        case 'a':
            s.tab = Tab::Agents;
            return true;
        case 'u':
            if (s.update.available)
                out = ask_for(s, Ask::Update, "");
            else
                out = act(ActionKind::UpdateKey, "u");
            return true;
        case 'c':
            out = act(ActionKind::UpdateKey, "c");
            return true;
        case 'R':
            out = act(ActionKind::UpdateKey, "r");
            return true;
        case 'r':
            out = ask_for(s, Ask::Restart, "");
            return true;
        case 'x':
            out = ask_for(s, Ask::StopServer, "");
            return true;
        case 'l':
            out = act(ActionKind::ToggleLog);
            return true;
        case 'p':
            s.project_view = (s.project_view + 1) % ((int)s.projects.size() + 1);
            return true;
        case 's':
            if (!s.server.up) out = act(ActionKind::StartServer);
            return true;
        default:
            return false;
    }
}

bool handle_start(UiState& s, Key key, Action& out) {
    if (nothing_installed(s)) return start_empty(s, key, out);
    if (!s.server.up) return start_server_down(s, key, out);
    return start_running(s, key, out);
}

// Models

void cycle(size_t& index, size_t count) { index = count ? (index + 1) % count : 0; }

bool models_enter(UiState& s, const ModelRow& row, Action& out) {
    const ModelCard& c = *row.card;
    if (row.section == ModelRow::Section::Coming)
        s.message = tr("msg_coming");
    else if (row.loaded)
        s.message = tr("msg_already_loaded");
    else if (row.section == ModelRow::Section::ToDownload)
        s.message = tr("msg_download_first");
    else
        out = ask_for(s, Ask::SwitchModel, c.id);
    return true;
}

// f task, t language, m size: each starts the list again at the top
bool models_filter_key(UiState& s, char ch) {
    if (ch == 'f')
        cycle(s.filter.task, s.offer.tasks.size() + 1);
    else if (ch == 't')
        cycle(s.filter.language, filter_languages(s).size());
    else if (ch == 'm')
        s.filter.fits_only = !s.filter.fits_only;
    else
        return false;
    s.models_cursor = 0;
    return true;
}

bool handle_models(UiState& s, Key key, Action& out) {
    const std::vector<ModelRow> rows = model_rows(s);
    if (is_move(key)) return move_cursor(s.models_cursor, key, (int)rows.size()), true;
    const ModelRow* row = selected_model(s, rows);
    if (key.code == KeyCode::Enter) return row ? models_enter(s, *row, out) : true;
    if (key.code != KeyCode::Char) return false;
    if (models_filter_key(s, key.ch)) return true;
    switch (key.ch) {
        case 'd':
            if (row && row->section == ModelRow::Section::ToDownload) out = act(ActionKind::Download, row->card->id);
            return true;
        case 'x':
            if (row && row->section == ModelRow::Section::Installed && !row->loaded)
                out = ask_for(s, Ask::Remove, row->card->id);
            else if (row)
                s.message = tr("msg_cannot_remove");
            return true;
        default:
            return false;
    }
}

// Agents

// Tab and the arrows between the fields; the value of the field changes with left, right and Space
bool agents_move_field(AgentForm& f, Key key) {
    const bool next = key.code == KeyCode::Tab || key.code == KeyCode::Down || is_char(key, 'j');
    const bool prev = key.code == KeyCode::BackTab || key.code == KeyCode::Up || is_char(key, 'k');
    if (next) f.field = (f.field + 1) % AgentForm::FieldCount;
    if (prev) f.field = (f.field + AgentForm::FieldCount - 1) % AgentForm::FieldCount;
    return next || prev;
}

bool agents_change_value(AgentForm& f, Key key) {
    const int step = key.code == KeyCode::Right ? 1 : key.code == KeyCode::Left ? -1 : 0;
    const bool space = is_char(key, ' ');
    if (f.field == AgentForm::Agent) f.agent = std::clamp(f.agent + step, 0, 2);
    if (f.field == AgentForm::Effort) f.effort = std::clamp(f.effort + step, 0, 2);
    if (f.field == AgentForm::Tools && (step || space)) f.all_tools = !f.all_tools;
    return step != 0 || space;
}

bool handle_agents(UiState& s, Key key, Action& out) {
    if (agents_move_field(s.agents, key)) return true;
    if (key.code == KeyCode::Enter) return out = act(ActionKind::LaunchAgent), true;
    return agents_change_value(s.agents, key);
}

// Settings

bool settings_change(UiState& s, int direction) {
    const std::vector<const SettingSpec*> rows = group_rows(s.settings.group);
    if (rows.empty()) return true;
    const SettingSpec& spec = *rows[std::min(s.settings.row, rows.size() - 1)];
    const std::string value = step_value(s.config, spec, direction);
    if (const std::string err = set_value(s.config, spec.key, value); !err.empty())
        s.message = err;
    else
        s.settings.dirty = true;
    return true;
}

bool handle_settings(UiState& s, Key key, Action& out) {
    SettingsForm& f = s.settings;
    const size_t groups = setting_groups().size();
    if (key.code == KeyCode::Tab) return f.group = (f.group + 1) % groups, f.row = 0, true;
    if (key.code == KeyCode::BackTab) return f.group = (f.group + groups - 1) % groups, f.row = 0, true;
    if (is_move(key)) {
        int row = (int)f.row;
        move_cursor(row, key, (int)group_rows(f.group).size());
        return f.row = (size_t)row, true;
    }
    if (key.code == KeyCode::Right || key.code == KeyCode::Left)
        return settings_change(s, key.code == KeyCode::Right ? 1 : -1);
    if (is_char(key, ' ')) return settings_change(s, 1);
    if (is_char(key, 's')) return out = act(ActionKind::SaveConfig), true;
    if (is_char(key, 'z')) return out = act(ActionKind::ApplyRecommendation), true;
    if (is_char(key, 'r')) return out = ask_for(s, Ask::Restart, ""), true;
    return false;
}

}  // namespace

bool handle_page(UiState& s, Key key, Action& out) {
    switch (s.tab) {
        case Tab::Start:
            return handle_start(s, key, out);
        case Tab::Models:
            return handle_models(s, key, out);
        case Tab::Agents:
            return handle_agents(s, key, out);
        case Tab::Settings:
            return handle_settings(s, key, out);
    }
    return false;
}

Action switch_key(UiState& s) {
    if (s.tab == Tab::Agents) {
        if (const ModelCard* c = suggest_model(s)) return ask_for(s, Ask::SwitchModel, c->id);
    }
    s.tab = Tab::Models, s.message.clear();
    return {};
}

}  // namespace tona::client
