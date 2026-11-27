// The Start page: what tona showed (running requests, write rate, memory, recent requests, update) in the layout of
// the design; plus the load progress, the log and the per-project view
#include <algorithm>
#include <format>

#include "client/format.h"
#include "client/term.h"
#include "client/text.h"
#include "client/view.h"

namespace tona::client {

namespace {

constexpr int kShownConversations = 3, kShownRecent = 3;
constexpr int kBarWidth = 20;

bool in_view(const UiState& s, const std::string& project) {
    if (s.project_view == 0) return true;
    return (size_t)s.project_view <= s.projects.size() && s.projects[(size_t)s.project_view - 1] == project;
}

std::string conversation_row(const Conversation& c) {
    std::string state, rate;
    if (c.state == "decode") {
        state = tr("conv_writes"), rate = std::format("{} {}", fnum(c.tok_s), tr("unit_tok_s"));
    } else if (c.state == "prefill") {
        state = tr("conv_reads"),
        rate = std::format("{:.0f} % · {} {}", 100 * c.prefill_fraction, fnum(c.prefill_tok_s), tr("unit_tok_s"));
    } else {
        state = tr("conv_waits"), rate = "";
    }
    const std::string who = c.agent.empty() ? (c.role.empty() ? "–" : c.role) : c.agent;
    std::string role = c.role == "auxiliary" || c.role == "subagent" ? " · " + c.role : "";
    return "   " + cell(who, 10) + " " + cell(project_label(c.project) + role, 16) + " " + cell(state, 11) + " " + rate;
}

void now_rows(const UiState& s, Lines& out) {
    out.push_back(section_row(s.project_view == 0 ? tr("sec_now")
                                                  : tr("sec_now") + " · " + s.projects[(size_t)s.project_view - 1]));
    int shown = 0, hidden = 0;
    for (const Conversation& c : s.server.running) {
        if (!in_view(s, c.project)) continue;
        if (shown++ < kShownConversations)
            out.push_back(conversation_row(c));
        else
            hidden++;
    }
    if (shown == 0) out.push_back("   " + dim(tr("now_idle")));
    if (hidden > 0) out.push_back("   " + dim(fill(tr("now_more"), fnum(hidden))));
}

void loading_rows(const UiState& s, Lines& out) {
    out.push_back(section_row(tr("sec_loading")));
    for (const LoadPhase& p : s.server.phases) {
        const std::string mark = p.done ? paint_ok("✓") : paint_info("▸");
        out.push_back("   " + mark + " " + cell(p.name, 30) + " " + meter(p.fraction, kBarWidth) +
                      std::format(" {:>3.0f} %", 100 * p.fraction));
    }
    if (s.server.phases.empty()) out.push_back("   " + dim(tr("loading_starting")));
}

std::string rate_summary(const std::vector<double>& rates) {
    double sum = 0, last = 0;
    size_t n = 0;
    for (double r : rates) {
        if (r > 0) sum += r, n++;
    }
    for (auto it = rates.rbegin(); it != rates.rend(); ++it)
        if (*it > 0) {
            last = *it;
            break;
        }
    return fill(tr("spark_summary"), fnum(last), fnum(n ? sum / (double)n : 0));
}

void spark_rows(const UiState& s, Lines& out) {
    constexpr size_t kWindowPolls = 480;  // two minutes at four polls per second
    // The newest value is at the right edge; a server that is only a few seconds old fills only the last columns
    std::vector<double> window(s.rates.size() > kWindowPolls ? s.rates.end() - (long)kWindowPolls : s.rates.begin(),
                               s.rates.end());
    window.insert(window.begin(), kWindowPolls - window.size(), 0.0);
    double top = 20;  // the scale never falls below 20 tok/s, so a quiet server does not look busy
    for (double r : window) top = std::max(top, r);
    out.push_back(section_row(tr("sec_spark")));
    out.push_back("   " + paint_ok(sparkline(window, 28, top)) + "   " + dim(rate_summary(window)));
}

void memory_rows(const UiState& s, Lines& out) {
    const Memory& m = s.server.mem;
    const double gb = 1e9;
    out.push_back(section_row(tr("sec_memory")));
    // A graphics memory the server could not read (no sysfs) gets no row; on unified memory (Apple silicon) the one
    // memory row counts the model's weights and GPU buffers as Tona's
    if (!m.unified && m.vram_total > 0)
        out.push_back("   " + cell(tr("mem_vram"), 16) + " " + meter(m.vram_used / m.vram_total, kBarWidth) + "  " +
                      fill(tr("mem_vram_of"), fdec(m.vram_used / gb), fdec(m.vram_total / gb, 0)));
    const double tona = m.unified ? m.tona_gpu : m.ram_tona;
    const double used = m.ram_total - m.ram_available, other = std::max(0.0, used - tona);
    out.push_back(
        "   " + cell(tr(m.unified ? "mem_shared" : "mem_ram"), 16) + " " +
        meter2(m.ram_total > 0 ? tona / m.ram_total : 0, m.ram_total > 0 ? other / m.ram_total : 0, kBarWidth) + "  " +
        fill(tr("mem_ram_split"), fnum(tona / gb), fnum(other / gb), fnum(m.ram_available / gb)));
    const int slots = std::max(s.server.slots, 1);
    out.push_back("   " + cell(tr("mem_chats"), 16) + " " +
                  meter((double)s.server.conversation_count() / slots, kBarWidth) + "  " +
                  fill(tr("mem_chats_of"), fnum(s.server.conversation_count()), fnum(slots)));
}

std::string recent_row(const RecentRequest& r) {
    return "   " + clock_hm(r.time) + "  " + cell(r.agent.empty() ? project_label(r.project) : r.agent, 10) + " " +
           cell(fill(tr("recent_written"), fnum(r.out_tokens)), 24) + " " + fnum(r.tok_s) + " " + tr("unit_tok_s");
}

void recent_rows(const UiState& s, Lines& out) {
    out.push_back(section_row(tr("sec_recent")));
    int shown = 0;
    for (auto it = s.server.recent.rbegin(); it != s.server.recent.rend() && shown < kShownRecent; ++it) {
        if (!in_view(s, it->project)) continue;
        out.push_back(recent_row(*it));
        shown++;
    }
    if (shown == 0) out.push_back("   " + dim(tr("recent_none")));
}

void log_rows(const UiState& s, Lines& out, int room) {
    out.push_back(section_row(tr("sec_log")));
    const int n = (int)s.log.size(), first = std::max(0, n - std::max(room - 1, 1));
    for (int i = first; i < n; i++) out.push_back("   " + dim(s.log[(size_t)i]));
    if (s.log.empty()) out.push_back("   " + dim(tr("log_none")));
}

std::string update_hint(const UpdateInfo& u) {
    if (u.downloading) return fill(tr("update_loading"), u.version, std::format("{:.0f}", 100 * u.fraction));
    if (u.available) return fill(tr(u.security ? "update_ready_security" : "update_ready"), u.version);
    if (u.checking) return tr("update_checking");
    if (!u.message.empty()) return u.message;
    return "";
}

void hint_rows(const UiState& s, Lines& out) {
    Lines hints;
    if (s.update.enabled)
        if (const std::string h = update_hint(s.update); !h.empty())
            hints.push_back(s.update.failed ? paint_bad(h) : paint_info(h));
    if (!s.job.empty()) hints.push_back(paint_info(s.job));
    if (hints.empty()) return;
    out.emplace_back();
    for (const std::string& h : hints) out.push_back(" " + h);
}

}  // namespace

Lines view_start(const UiState& s, int rows) {
    Lines out;
    if (s.server.loading())
        loading_rows(s, out);
    else if (!s.server.error.empty())
        out.push_back(" " + paint_bad(fill(tr("load_failed"), s.server.error)));
    else
        now_rows(s, out), spark_rows(s, out);
    memory_rows(s, out);
    if (s.show_log) {
        log_rows(s, out, rows - (int)out.size() - 1);
        return out;
    }
    recent_rows(s, out);
    hint_rows(s, out);
    return out;
}

}  // namespace tona::client
