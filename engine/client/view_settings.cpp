// The Settings page: groups on the left, the rows of the chosen group on the right
#include <algorithm>
#include <format>

#include "client/format.h"
#include "client/term.h"
#include "client/text.h"
#include "client/view.h"

namespace tona::client {

namespace {

constexpr int kGroupWidth = 20;
constexpr int kLabelWidth = 18;
constexpr int kHelpWidth = 54;

std::string value_label(const std::string& choice) {
    const std::string t = tr("val_" + choice);
    return t == "val_" + choice ? choice : t;
}

std::string value_text(const Config& c, const SettingSpec& spec) {
    const std::string value = get_value(c, spec.key);
    if (spec.kind == Kind::Number) return "◀ " + value + (spec.unit.empty() ? "" : " " + spec.unit) + " ▶";
    if (spec.kind == Kind::Toggle)
        return std::string(value == "on" ? "[x] " : "[ ] ") + tr("val_" + std::string(value == "on" ? "on" : "off"));
    std::string out;
    for (const std::string& choice : spec.choices)
        out += (out.empty() ? "" : "  ") + std::string(choice == value ? "(•) " : "( ) ") + value_label(choice);
    return out;
}

Lines right_pane(const UiState& s, const std::vector<const SettingSpec*>& rows) {
    Lines out;
    for (size_t i = 0; i < rows.size(); i++) {
        const SettingSpec& spec = *rows[i];
        const bool focus = i == s.settings.row;
        const std::string label = cell(tr("set_" + spec.key), kLabelWidth);
        const std::string row = label + " " + value_text(s.config, spec);
        out.push_back(focus ? paint_selected(row, 57) : row);
        // the help of the row under the cursor only: the others are one line each
        if (focus)
            for (const std::string& l : wrap_lines(tr(spec.help), kHelpWidth)) out.push_back("   " + dim(l));
    }
    return out;
}

}  // namespace

Lines view_settings(const UiState& s, int rows_available) {
    const std::vector<std::string> groups = setting_groups();
    const std::vector<const SettingSpec*> rows = group_rows(s.settings.group);
    const Lines right = right_pane(s, rows);
    Lines out;
    const int body = rows_available - 4;
    for (int i = 0; i < body; i++) {
        std::string left(kGroupWidth, ' ');
        if ((size_t)i < groups.size()) {
            const bool active = (size_t)i == s.settings.group;
            left = active ? " " + paint_selected("▸ " + cell(tr(groups[(size_t)i]), kGroupWidth - 4), kGroupWidth - 2) +
                                " "
                          : "   " + cell(tr(groups[(size_t)i]), kGroupWidth - 3);
        }
        out.push_back(left + dim("│") + " " + ((size_t)i < right.size() ? right[(size_t)i] : ""));
    }
    out.emplace_back("");
    out.push_back(" " + (s.settings.dirty ? paint_warn(tr("set_unsaved")) : dim(tr("set_saved"))) + "  " +
                  dim(tr("set_applies")));
    const size_t row = std::min(s.settings.row, rows.empty() ? 0 : rows.size() - 1);
    if (!rows.empty()) {
        const SettingSpec& spec = *rows[row];
        out.push_back(" " + dim(tr("set_command") + " ") +
                      paint_info("tona config set " + spec.key + " " + get_value(s.config, spec.key)));
    }
    return out;
}

}  // namespace tona::client
