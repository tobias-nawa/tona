#include "client/ui.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>

#include "client/term.h"
#include "client/text.h"

namespace tona::client {

bool is_installed(const UiState& s, const ModelCard& c) {
    if (c.coming || c.id.empty()) return false;
    if (c.id == s.loaded_id) return true;  // the server runs it: it is on the disk, whatever the verification mark says
    for (const install::ModelInfo& m : s.catalog)
        if (m.id == c.id) return m.present;
    return false;
}

std::vector<std::string> filter_languages(const UiState& s) {
    std::set<std::string> all;
    for (const ModelCard& c : s.offer.cards)
        for (const std::string& l : c.strong) all.insert(l);
    std::vector<std::string> out{""};  // "" = any language
    out.insert(out.end(), all.begin(), all.end());
    return out;
}

namespace {

bool passes_filter(const UiState& s, const ModelCard& c, const std::vector<std::string>& languages) {
    if (s.filter.task > 0 && s.filter.task <= s.offer.tasks.size()) {
        const std::string& task = s.offer.tasks[s.filter.task - 1].id;
        if (std::find(c.tasks.begin(), c.tasks.end(), task) == c.tasks.end()) return false;
    }
    if (!c.coming && s.filter.language > 0 && s.filter.language < languages.size() &&
        !card_strong_in(c, languages[s.filter.language]))
        return false;
    // Rule of thumb, not a measurement: Tona keeps the experts in RAM and the rest in graphics memory, so the file may
    // be larger than either; RAM plus graphics memory minus 8 GB for the system has to hold it
    const bool fits = s.offer.fits_by_class || s.ram_total_gb <= 0 || c.size_gb <= s.ram_total_gb + s.vram_total_gb - 8;
    return c.coming || !s.filter.fits_only || fits;
}

}  // namespace

std::vector<ModelRow> model_rows(const UiState& s) {
    const std::vector<std::string> languages = filter_languages(s);
    std::vector<ModelRow> rows;
    for (const ModelCard& c : s.offer.cards) {
        if (!passes_filter(s, c, languages)) continue;
        ModelRow r;
        r.card = &c;
        r.loaded = !c.id.empty() && c.id == s.loaded_id;
        r.section = c.coming             ? ModelRow::Section::Coming
                    : is_installed(s, c) ? ModelRow::Section::Installed
                                         : ModelRow::Section::ToDownload;
        rows.push_back(r);
    }
    std::stable_sort(rows.begin(), rows.end(), [](const ModelRow& a, const ModelRow& b) {
        return a.section != b.section ? a.section < b.section : a.loaded > b.loaded;
    });
    return rows;
}

const ModelRow* selected_model(const UiState& s, const std::vector<ModelRow>& rows) {
    if (rows.empty()) return nullptr;
    return &rows[(size_t)std::clamp(s.models_cursor, 0, (int)rows.size() - 1)];
}

const ModelCard* suggest_model(const UiState& s) {
    if (s.languages.empty() || s.loaded_id.empty()) return nullptr;
    const std::string& language = s.languages.front().language;
    const ModelCard* loaded = s.offer.find(s.loaded_id);
    if (!loaded || card_strong_in(*loaded, language)) return nullptr;
    for (const ModelCard& c : s.offer.cards)
        if (c.id != s.loaded_id && is_installed(s, c) && card_strong_in(c, language)) return &c;
    return nullptr;
}

std::string loaded_name(const UiState& s) {
    if (const ModelCard* c = s.offer.find(s.loaded_id)) return c->name;
    return s.server.model_file.empty() ? std::string() : model_stem(s.server.model_file);
}

std::vector<Conversation> running_conversations(const UiState& s) {
    std::vector<Conversation> out;
    for (const Conversation& c : s.server.running)
        if (c.state != "waiting") out.push_back(c);
    return out;
}

std::string agent_name(int agent) {
    static const char* names[] = {"claude", "opencode", "junie"};
    return names[std::clamp(agent, 0, 2)];
}

namespace {

// ~ for the home folder, quotes for a path with spaces
std::string shown_path(const std::string& path) {
    std::string p = path;
    if (const char* home = std::getenv("HOME");
        home && *home && p.starts_with(home) && (p.size() == strlen(home) || p[strlen(home)] == '/'))
        p = "~" + p.substr(strlen(home));
    return !p.contains(' ') ? p : "\"" + p + "\"";
}

}  // namespace

std::string agent_command(const UiState& s) {
    const AgentForm& f = s.agents;
    std::string cmd;
    if (!f.all_tools) cmd += "TONA_TOOLS=lean ";
    if (f.effort != 1) cmd += std::string("TONA_EFFORT=") + (f.effort == 0 ? "low" : "high") + " ";
    cmd += "tona " + agent_name(f.agent);
    if (!f.project.empty()) cmd += " " + shown_path(f.project);
    return cmd;
}

bool in_text_field(const UiState& s) { return s.tab == Tab::Agents && s.agents.field == AgentForm::Project; }

bool nothing_installed(const UiState& s) {
    if (s.catalog.empty() || !s.loaded_id.empty()) return false;  // unread catalog, or a model runs
    return std::none_of(s.catalog.begin(), s.catalog.end(),
                        [](const install::ModelInfo& m) { return m.kind == "model" && m.present; });
}

std::vector<std::string> setting_groups() {
    std::vector<std::string> groups;
    for (const SettingSpec& spec : setting_specs())
        if (std::find(groups.begin(), groups.end(), spec.group) == groups.end()) groups.push_back(spec.group);
    return groups;
}

std::vector<const SettingSpec*> group_rows(size_t group) {
    const std::vector<std::string> groups = setting_groups();
    std::vector<const SettingSpec*> rows;
    for (const SettingSpec& spec : setting_specs())
        if (group < groups.size() && spec.group == groups[group]) rows.push_back(&spec);
    return rows;
}

std::vector<int> close_options(const UiState& s) {
    // Without a running conversation nothing is lost by stopping: that choice comes first
    return running_conversations(s).empty() ? std::vector<int>{1, 0, 2} : std::vector<int>{0, 1, 2};
}

}  // namespace tona::client
