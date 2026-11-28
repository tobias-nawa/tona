// The Agents page: choose agent, project, tools and thinking effort; shows the command that Enter runs
#include <algorithm>

#include "client/format.h"
#include "client/term.h"
#include "client/text.h"
#include "client/view.h"

namespace tona::client {

namespace {

std::string radio(bool on, const std::string& label) { return std::string(on ? "(•) " : "( ) ") + label; }

std::string field_row(const UiState& s, int field, const char* label, const std::string& value) {
    const bool focus = s.agents.field == field;
    if (focus) return " " + paint_selected("▸ " + cell(tr(label), 12) + " " + value, 78);
    return "   " + cell(tr(label), 12) + " " + value;
}

Lines project_rows(const UiState& s) {
    const bool focus = s.agents.field == AgentForm::Project;
    const std::string text = s.agents.project.empty() ? dim(tr("agents_project_hint")) : s.agents.project;
    std::string line;
    if (focus) {
        const std::string head = "▸ " + cell(tr("agents_project"), 12) + " " +
                                 (s.agents.project.empty() ? tr("agents_project_hint") : s.agents.project);
        const int used = top::width(head);
        line = " " + paint_selected(head, used) + top::paint(top::kInverse, " ") +
               (used + 1 < 78 ? paint_selected("", 78 - used - 1) : "");
    } else {
        line = field_row(s, AgentForm::Project, "agents_project", text);
    }
    Lines out{std::move(line)};
    const std::string langs = languages_text(s.languages);
    out.push_back("                " +
                  dim(langs.empty() ? tr("agents_no_languages") : tr("agents_detected") + " " + langs));
    return out;
}

Lines model_rows_agent(const UiState& s) {
    const std::string name = loaded_name(s);
    std::string value = name.empty() ? dim(tr("agents_no_model")) : name + " " + dim("(" + tr("agents_loaded") + ")");
    Lines out{field_row(s, AgentForm::Model, "agents_model", value)};
    if (const ModelCard* c = suggest_model(s))
        out.push_back("                " +
                      paint_info(fill(tr("agents_suggest"), c->name, s.languages.front().language)) + " " +
                      dim(tr("agents_key_w")));
    return out;
}

Lines recent_launch_rows(const UiState& s) {
    Lines out;
    out.push_back(section_row(tr("agents_recent")));
    int shown = 0;
    for (const LaunchEntry& l : s.launches) {
        if (shown++ >= 3) break;
        const long long secs = std::max(0LL, s.now - l.time);
        out.push_back("   " + cell(l.agent, 9) + " " + cell(project_label(l.project), 28) + " " + cell(l.model, 10) +
                      " " + dim(l.time > 0 ? fill(tr("agents_ago"), age((double)secs)) : ""));
    }
    return out;
}

}  // namespace

Lines view_agents(const UiState& s, int) {
    const AgentForm& f = s.agents;
    Lines out;
    out.push_back(field_row(s, AgentForm::Agent, "agents_agent",
                            radio(f.agent == 0, "Claude Code") + "   " + radio(f.agent == 1, "OpenCode") + "   " +
                                radio(f.agent == 2, "Junie")));
    for (std::string& r : project_rows(s)) out.push_back(std::move(r));
    for (std::string& r : model_rows_agent(s)) out.push_back(std::move(r));
    out.push_back(field_row(
        s, AgentForm::Tools, "agents_tools",
        std::string(f.all_tools ? "[x] " : "[ ] ") + tr(f.all_tools ? "agents_tools_all" : "agents_tools_lean")));
    out.push_back(field_row(s, AgentForm::Effort, "agents_effort",
                            radio(f.effort == 0, tr("effort_short")) + "   " +
                                radio(f.effort == 1, tr("effort_medium")) + "   " +
                                radio(f.effort == 2, tr("effort_long"))));
    out.emplace_back("");
    out.push_back("                " + tr("agents_command") + " " + paint_info(cell(agent_command(s), 80 - 17 - 9)));
    const int n = (int)running_conversations(s).size();
    if (n > 0) out.push_back("                " + dim(fill(tr("agents_switch_cuts"), fnum(n))));
    if (!s.launches.empty()) {
        out.emplace_back("");
        for (std::string& r : recent_launch_rows(s)) out.push_back(std::move(r));
    }
    return out;
}

}  // namespace tona::client
