// The Models page: filter, the list in three sections, the card of the selected model
#include <algorithm>
#include <format>

#include "client/format.h"
#include "client/term.h"
#include "client/text.h"
#include "client/view.h"

namespace tona::client {

namespace {

constexpr int kCardRows = 8;

std::string join(const std::vector<std::string>& v) {
    std::string out;
    for (const std::string& s : v) out += (out.empty() ? "" : ", ") + s;
    return out;
}

std::string filter_line(const UiState& s) {
    const std::vector<std::string> languages = filter_languages(s);
    std::string task = tr("filter_all_tasks");
    if (s.filter.task > 0 && s.filter.task <= s.offer.tasks.size()) task = s.offer.tasks[s.filter.task - 1].label;
    const std::string language = s.filter.language > 0 && s.filter.language < languages.size()
                                     ? languages[s.filter.language]
                                     : tr("filter_any_language");
#ifdef __APPLE__
    const char* fits_key = "filter_fits_mac";
#else
    const char* fits_key = "filter_fits";
#endif
    const std::string fits = s.filter.fits_only ? tr(fits_key) : tr("filter_all_sizes");
    return " " + tr("filter_title") + " " + bold(task + " · " + language + " · " + fits) + "   " +
           dim("f " + tr("k_filter") + "  t " + tr("k_language") + "  m " + tr("k_fits"));
}

std::string row_note(const UiState& s, const ModelRow& r) {
    const ModelCard& c = *r.card;
    if (r.section == ModelRow::Section::Coming) return "";  // the section header says it
    if (r.loaded) return tr("row_running");
    const std::vector<std::string> languages = filter_languages(s);
    if (s.filter.language > 0 && s.filter.language < languages.size() &&
        card_strong_in(c, languages[s.filter.language]))
        return fill(tr("row_best_for"), languages[s.filter.language]);
    return c.note;
}

std::string model_row_text(const UiState& s, const ModelRow& r, bool cursor) {
    const ModelCard& c = *r.card;
    // The section header says installed or to download, and every model in Tona is measured: a basis word
    // appears only where it is another one
    const bool usual = c.coming || c.speed_basis == Basis::Measured;
    const std::string basis = usual ? std::string() : basis_word(c.speed_basis);
    const std::string size = c.size_gb > 0 ? fnum(c.size_gb) + " GB" : "";
    std::string row = std::string(cursor ? " ▸ " : "   ") + cell(c.name, 22) + " " + cell(row_note(s, r), 32) + " " +
                      top::right(size, 7) + "  " + basis;
    return cursor ? bold(row) : row;
}

// All rows of the list with their section headers; `cursor_line` receives the line of the cursor row
Lines list_lines(const UiState& s, const std::vector<ModelRow>& rows, int& cursor_line) {
    Lines out;
    static const struct {
        ModelRow::Section section;
        const char* title;
    } sections[] = {{ModelRow::Section::Installed, "sec_installed"},
                    {ModelRow::Section::ToDownload, "sec_to_download"},
                    {ModelRow::Section::Coming, "sec_coming"}};
    for (const auto& [section, title] : sections) {
        out.push_back(section_row(tr(title)));
        bool any = false;
        for (size_t i = 0; i < rows.size(); i++) {
            if (rows[i].section != section) continue;
            any = true;
            if ((int)i == std::clamp(s.models_cursor, 0, (int)rows.size() - 1)) cursor_line = (int)out.size();
            out.push_back(model_row_text(s, rows[i], (int)i == s.models_cursor));
        }
        if (!any && section == ModelRow::Section::ToDownload) out.push_back("   " + dim(tr("list_none_to_download")));
    }
    return out;
}

// The basis word only where it is not the usual one: a model enters Tona only after it is measured
std::string speed_text(const ModelCard& c) {
    if (c.decode_tok_s <= 0 && c.decode_tok_s_hi <= 0) return tr("card_speed_unknown");
    if (c.decode_tok_s_hi > 0)
        return fill(tr("card_speed_range"), fdec(c.decode_tok_s, 0) + "–" + fdec(c.decode_tok_s_hi, 0),
                    fctx(c.speed_context));
    const std::string unusual = c.speed_basis == Basis::Measured ? "" : ", " + basis_word(c.speed_basis);
    return fill(tr("card_speed_value"), fdec(c.decode_tok_s), fctx(c.speed_context)) + unusual;
}

// The licence text is long: it wraps under its label, never cut
Lines license_lines(const ModelCard& c) {
    const std::string license = c.license.empty()
                                    ? tr("card_open")
                                    : c.license + (c.license_checked ? "" : ", " + tr("card_license_unchecked"));
    Lines out;
    const Lines wrapped = wrap_lines(license, 80 - 3 - 13 - 1);
    for (size_t i = 0; i < wrapped.size(); i++)
        out.push_back("   " + cell(i ? "" : tr("card_license"), 13) + wrapped[i]);
    return out;
}

Lines card_lines(const UiState& s, const ModelRow* row) {
    Lines out;
    if (!row) return {" " + dim(tr("card_none"))};
    const ModelCard& c = *row->card;
    out.push_back(" " + bold(c.name));
    const auto line = [&](const char* label, const std::string& text) {
        out.push_back("   " + cell(tr(label), 13) + text);
    };
    if (c.coming) {
        line("card_state", c.note.empty() ? tr("card_coming") : c.note);
        return out;
    }
    line("card_good", c.strong.empty() ? tr("card_open") : join(c.strong));
    line("card_weak", !c.weak.empty() ? join(c.weak) : c.weak_note.empty() ? tr("card_open") : c.weak_note);
    if (!c.open.empty()) line("card_open_langs", join(c.open));
    line("card_speed", speed_text(c));
    const double vram = s.vram_total_gb;
    line("card_size",
         fill(tr(vram > 0 && c.size_gb <= vram * 0.9 ? "card_size_fits" : "card_size_ram"), fnum(c.size_gb)));
    for (std::string& l : license_lines(c)) out.push_back(std::move(l));
    return out;
}

// The last line: what Enter does, or the load progress of a switch
std::string action_line(const UiState& s, const ModelRow* row) {
    if (!s.switching_to.empty()) {
        const ModelCard* to = s.offer.find(s.switching_to);
        const ModelCard* from = s.offer.find(s.switching_from);
        return " " +
               fill(tr("switching_progress"), to ? to->name : s.switching_to,
                    std::format("{:.0f}", 100 * s.server.load_fraction())) +
               " " + meter(s.server.load_fraction(), 16) + "   " +
               dim(fill(tr("switching_cancel"), from ? from->name : s.switching_from));
    }
    if (!row || row->section != ModelRow::Section::Installed || row->loaded) return "";
    const int n = (int)running_conversations(s).size();
    return " " + (n > 0 ? fill(tr("enter_switch_asks"), row->card->name, fnum(n))
                        : fill(tr("enter_switch"), row->card->name));
}

}  // namespace

Lines view_models(const UiState& s, int rows_available) {
    const std::vector<ModelRow> rows = model_rows(s);
    const ModelRow* row = selected_model(s, rows);
    Lines out{filter_line(s)};
    const int room = rows_available - 1 - kCardRows - 2;  // filter line, card, rule and action line
    int cursor_line = 0;
    Lines list = list_lines(s, rows, cursor_line);
    // Keep the cursor row in view when the list is longer than its room
    const int first = std::clamp(cursor_line - room + 2, 0, std::max(0, (int)list.size() - room));
    for (int i = first; i < (int)list.size() && i < first + room; i++) out.push_back(list[(size_t)i]);
    while ((int)out.size() < 1 + room) out.emplace_back("");
    out.push_back(" " + dim(top::repeat("─", 78)));
    for (std::string& l : card_lines(s, row)) out.push_back(std::move(l));
    out.resize((size_t)(rows_available - 1));
    out.push_back(action_line(s, row));
    return out;
}

}  // namespace tona::client
