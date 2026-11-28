// The Room page (how much Tona may take) and the advanced settings behind the key `e`.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <functional>

#include "core/context_plan.h"
#include "core/describe.h"
#include "core/port.h"
#include "tui.h"
#include "tui_widgets.h"

using namespace ftxui;

namespace tona::install {

namespace {

constexpr int kBarWidth = 48;

std::string radios(const std::vector<std::string>& labels, int selected) {
    std::string out;
    for (size_t i = 0; i < labels.size(); i++)
        out += (i ? "  " : "") + ui::radio(static_cast<int>(i) == selected) + " " + labels[i];
    return out;
}

std::string gb_text(double v) {
    std::string t = std::format("{:g}", std::round(v * 10) / 10);
    return t + " GB";
}

// One row of the advanced settings: label, help, the value as text, and what the arrows and a typed number do
struct AdvRow {
    std::string label, help, value;
    std::function<void(int)> change;     // dir -1 or +1, or a toggle for 0
    std::function<void(double)> number;  // empty: not a number
};

template <typename T>
void cycle(T& v, const std::vector<T>& options, int dir) {
    const auto at = std::find(options.begin(), options.end(), v);
    const int i = at == options.end() ? 0 : static_cast<int>(at - options.begin());
    v = options[static_cast<size_t>((i + dir + static_cast<int>(options.size())) % static_cast<int>(options.size()))];
}

// The context choices: 0 is auto, then the sizes in ascending order, as far as the backend offers them. A size the
// settings hold that is no choice (a configuration written by hand) is listed too, so the radios always mark the
// current choice.
std::vector<int> context_options(const Settings& s, const Limits& l) {
    std::vector<int> o;
    if (l.auto_context) o.push_back(0);
    const std::vector<int>& sizes = l.contexts.empty() ? context_steps() : l.contexts;
    o.insert(o.end(), sizes.begin(), sizes.end());
    if (!s.ctx_auto && std::find(o.begin(), o.end(), s.ctx) == o.end())
        o.insert(std::upper_bound(o.begin() + (l.auto_context ? 1 : 0), o.end(), s.ctx), s.ctx);
    return o;
}

std::vector<std::string> agent_labels(int max) {
    std::vector<std::string> out;
    for (int i = 1; i <= max; i++) out.push_back(std::to_string(i));
    return out;
}

std::string context_label(int ctx) { return ctx == 0 ? "auto" : context_text(ctx); }

std::vector<std::string> context_labels(const std::vector<int>& options) {
    std::vector<std::string> out(options.size());
    std::transform(options.begin(), options.end(), out.begin(), context_label);
    return out;
}

int context_choice(const Settings& s, const std::vector<int>& options) {
    const int want = s.ctx_auto && options.front() == 0 ? 0 : s.ctx;
    const auto at = std::find(options.begin(), options.end(), want);
    return at == options.end() ? 0 : static_cast<int>(at - options.begin());
}

std::string on_off(bool b) { return b ? ui::box(true) + " " + "on" : ui::box(false) + " " + "off"; }

}  // namespace

// The speed of every chosen model under the agents and context chosen above, one line each. The field is one for all
// models; "auto" is planned per model (its own pinning limit), so the sizes may differ. A conversation's own `tona use`
// plans "auto" again for the model it switches to (client/control.cpp).
std::string Wizard::speed_line() const {
    std::string out = "Decode on " + this_machine(st_.h.os == "macos") + ", estimated";
    for (const std::string& id : models_of(s())) {
        Settings t = s();
        t.model = id;
        if (t.ctx_auto && st_.module) t.ctx = st_.module->auto_context(t);
        const Estimate e = st_.module ? st_.module->estimate(st_.h, t, id) : Estimate{};
        const offer::ModelCard* card = offer::model_card(id);
        std::string line = std::format("  {}{} x {}   {}", ui::pad(card ? card->name : id, 8), t.seqs,
                                       context_text(t.ctx), load_speed_text(e));
        if (!t.ctx_auto && e.decode.hi > 0 && !e.pinned) line += ": experts not pinned, try auto";
        out += "\n" + line;
    }
    return out;
}

void Wizard::room_changed() {
    if (!target_.release) s().network = false;  // the checkout's scripts do not read a host
    apply_room(s(), room_plan(st_.h, server_only_, priority_, s().vram_gb), server_only_, priority_);
    resolve_context(st_.module, s());
}

// The rows in the order of the page
enum class Row { Use, Access, Port, Login, Tight, Agents, Context };

std::vector<Row> room_rows() {
    return {Row::Use, Row::Access, Row::Port, Row::Login, Row::Tight, Row::Agents, Row::Context};
}

// One line for the row under the cursor; the long explanations are behind `e`
std::string room_help(Row row, const Limits& l) {
    switch (row) {
        case Row::Use:
            return "A server dedicated to Tona may take more of the memory.";
        case Row::Access:
            return "Without a key every program on this machine can use the engine; other machines need one.";
        case Row::Port:
            return "Type digits or use the arrows; the page checks the port as you go.";
        case Row::Login:
            return "The server then starts at login and holds its memory.";
        case Row::Tight:
            return "Not active yet; your choice is stored.";
        case Row::Agents:
            return l.max_agents == 1 ? "This backend is measured with one agent."
                                     : "Every agent has its own context; more agents need more memory.";
        case Row::Context:
            break;
    }
    return l.auto_context ? "auto: the largest context that keeps all experts in RAM (full speed)."
                          : "The size this backend is measured with.";
}

// Cached port probe so key presses do not stall for a second on an unresponsive port
PortState cached_port_state(int port) {
    static int last_port = 0;
    static std::chrono::steady_clock::time_point at;
    static PortState state = PortState::Free;
    const auto now = std::chrono::steady_clock::now();
    if (port != last_port || now - at > std::chrono::seconds(1)) state = probe_port(port), last_port = port, at = now;
    return state;
}

// "8662  free", "8662  yours (running)" or "8662  in use -> 8663"
Element port_value_el(int port, bool here) {
    const PortState state = cached_port_state(port);
    const std::string num = std::format("{}  ", port);
    const std::string status = state == PortState::Busy   ? std::format("in use → {}", next_free_port(port + 1))
                               : state == PortState::Ours ? "yours (running)"
                                                          : "free";
    if (here) return ui::plain(num + status);
    return hbox({ui::plain(num), state == PortState::Busy ? ui::warn(status) : ui::good(status)});
}

// Two-line bars (label + blue/dim bar, then legend with matching swatches underneath)
Elements room_bars(const RoomPlan& r) {
    Elements out;
    out.push_back(hbox({ui::dim(ui::pad(std::format("Memory {:g} GB", r.ram_total), 15)),
                        ui::bar_el({r.tona, r.programs, r.system}, kBarWidth)}));
    out.push_back(hbox({ui::plain(ui::pad("", 15)), ui::swatch("█", true),
                        ui::plain(std::format(" Tona {:g} GB   ", std::round(r.tona))), ui::swatch("▒", false),
                        ui::dim(std::format(" your programs {:g} GB   ", std::round(r.programs))),
                        ui::swatch("░", false), ui::dim(std::format(" system {:g} GB", std::round(r.system)))}));
    if (!r.unified && r.vram_total > 0) {
        out.push_back(hbox({ui::dim(ui::pad(std::format("Video {:g} GB", r.vram_total), 15)),
                            ui::bar_el({r.vram_tona, r.vram_screen}, kBarWidth)}));
        out.push_back(
            hbox({ui::plain(ui::pad("", 15)), ui::swatch("█", true),
                  ui::plain(std::format(" Tona {:g} GB   ", std::round(r.vram_tona * 10) / 10)), ui::swatch("▒", false),
                  ui::dim(std::format(" screen {:g} GB", std::round(r.vram_screen * 10) / 10))}));
    }
    return out;
}

// Digits type a port, Backspace deletes, Esc drops it, any other key keeps it and does its own work (Enter, `e`, the
// arrows); true if the key was used up
bool port_key(const Event& e, bool& editing, std::string& edit, int& port) {
    const bool digit =
        e.is_character() && e.character().find_first_of("0123456789") != std::string::npos && e.character().size() == 1;
    if (!editing && !digit) return false;
    if (!editing) return editing = true, edit = e.character(), true;
    if (e == Event::Escape) return editing = false, true;
    if (e == Event::Backspace) return edit.empty() ? (editing = false, true) : (edit.pop_back(), true);
    if (digit) return edit.size() < 5 ? (edit += e.character(), true) : true;
    if (const long n = std::strtol(edit.c_str(), nullptr, 10); n >= 1024 && n <= 65535) port = static_cast<int>(n);
    editing = false;
    return e == Event::Return;  // Enter only keeps the number; other keys go on to their own work
}

Element Wizard::room_body() {
    const RoomPlan r = room_plan(st_.h, server_only_, priority_, s().vram_gb);
    const bool mac = st_.h.os == "macos";
    const std::vector<Row> rows = room_rows();
    const auto is_at = [&](Row id) { return rows[static_cast<size_t>(cursor_)] == id; };
    const auto row_el = [&](Row id, const std::string& label, Element val) {
        const bool here = is_at(id);
        return ui::highlight(hbox({ui::plain(ui::cursor(here) + ui::pad(label, 20)), std::move(val)}), here);
    };
    const auto row = [&](Row id, const std::string& label, const std::string& value) {
        return row_el(id, label, ui::plain(value));
    };
    const Limits lim = st_.module ? st_.module->limits() : Limits{};
    const std::vector<int> ctx = context_options(s(), lim);
    Elements l = {ui::title("What for")};
    l.push_back(
        row(Row::Use, "Use", radios({"I work on " + this_machine(mac), "only for Tona, a server"}, server_only_)));
    l.push_back(row(Row::Access, "Access",
                    radios({this_machine(mac) + " only", other_machines(mac) + ", with a key"}, s().network ? 1 : 0)));
    l.push_back(
        row_el(Row::Port, "Port", editing_ ? ui::plain(edit_ + "_") : port_value_el(s().port, is_at(Row::Port))));
    l.push_back(row_el(Row::Login, "At login",
                       is_at(Row::Login) ? ui::plain(ui::box(s().service) + " start the server (then it holds memory)")
                                         : hbox({ui::plain(ui::box(s().service) + " start the server "),
                                                 ui::dim("(then it holds memory)")})));
    l.push_back(ui::blank());
    l.push_back(ui::title("How much"));
    const std::string tight_radios = radios({"work first", "balanced", "Tona first"}, static_cast<int>(priority_));
    l.push_back(row_el(Row::Tight, "When it gets tight",
                       is_at(Row::Tight) ? ui::plain(tight_radios + "  (coming)")
                                         : hbox({ui::plain(tight_radios), ui::dim("  (coming)")})));
    l.push_back(row(Row::Agents, "Parallel agents", radios(agent_labels(lim.max_agents), s().seqs - 1)));
    l.push_back(row(Row::Context, "Context per agent", radios(context_labels(ctx), context_choice(s(), ctx))));
    if (models_of(s()).size() <= 2 || message_.empty()) l.push_back(ui::blank());
    for (Element& b : room_bars(r)) l.push_back(std::move(b));
    const std::string speed = speed_line();
    for (size_t at = 0, end; at < speed.size(); at = end + 1) {
        end = std::min(speed.find('\n', at), speed.size());
        l.push_back(at == 0 ? ui::dim(speed.substr(at, end - at)) : ui::plain(speed.substr(at, end - at)));
    }
    if (l.size() < static_cast<size_t>(ui::kBody - (message_.empty() ? 0 : 1)))
        l.push_back(ui::dim(room_help(rows[static_cast<size_t>(cursor_)], lim)));
    if (!message_.empty()) l.push_back(ui::warn(message_));
    return vbox(l);
}

// The sizes in a ring; the Mac has one
static void change_context(Settings& s, const Limits& l, int dir) {
    const std::vector<int> o = context_options(s, l);
    const int n = static_cast<int>(o.size());
    const int next = o[static_cast<size_t>((context_choice(s, o) + dir + n) % n)];
    s.ctx_auto = next == 0;
    if (next > 0) s.ctx = next;
}

// The left and right arrows (dir) and the space bar change the answer of the row under the cursor
void Wizard::change_room_row(int dir) {
    const bool mac = st_.h.os == "macos";
    const Limits lim = st_.module ? st_.module->limits() : Limits{};
    switch (room_rows()[static_cast<size_t>(cursor_)]) {
        case Row::Use:
            server_only_ = !server_only_;
            break;
        case Row::Access:
            if (target_.release)
                s().network = !s().network;
            else
                message_ = "Other " + std::string(mac ? "computers" : "PCs") +
                           ": only with the installed release, not in a checkout.";
            break;
        case Row::Port:
            s().port = std::clamp(s().port + dir, 1024, 65535);
            break;
        case Row::Login:
            s().service = !s().service;
            break;
        case Row::Tight:
            priority_ = static_cast<Priority>((static_cast<int>(priority_) + dir + 3) % 3);
            break;
        case Row::Agents:
            s().seqs = (s().seqs - 1 + dir + lim.max_agents) % lim.max_agents + 1;
            break;
        case Row::Context:
            change_context(s(), lim, dir);
            break;
    }
    room_changed();
}

bool Wizard::key_room(const Event& e) {
    const std::vector<Row> rows = room_rows();
    if (rows[static_cast<size_t>(cursor_)] == Row::Port && port_key(e, editing_, edit_, s().port)) return true;
    if (ui::is_down(e) && cursor_ + 1 < static_cast<int>(rows.size())) return ++cursor_, true;
    if (ui::is_up(e) && cursor_ > 0) return --cursor_, true;
    if (ui::is_left(e) || ui::is_right(e) || ui::is_space(e)) return change_room_row(ui::is_left(e) ? -1 : 1), true;
    if (e == Event::Character('e')) return view_ = View::Advanced, adv_cursor_ = 0, true;
    if (e == Event::Return) return forward(), true;
    if (ui::is_back(e)) return back(), true;
    return true;
}

namespace {

std::vector<AdvRow> engine_rows(Settings& s, bool coder) {
    // Static, not captured: a lambda holding a copy of a const vector copies it again when the row is moved, and that
    // copy may throw
    static const std::vector<std::string> drafters = {"mtp", "dflash", "none"};
    static const std::vector<std::string> efforts = {"low", "medium", "high"};
    return {
        {"RAM limit of the engine",
         "How much memory the engine may use, page cache included. From 31.6 GB all experts are pinned in RAM; more "
         "gains nothing.",
         gb_text(s.ram_gb), [&s](int d) { s.ram_gb = std::max(8.0, s.ram_gb + d); },
         [&s](double v) { s.ram_gb = std::max(8.0, v); }},
        {"VRAM limit of the engine",
         "How much video memory the engine may take. The screen needs its share on the same card.", gb_text(s.vram_gb),
         [&s](int d) { s.vram_gb = std::max(6.0, s.vram_gb + d * 0.5); },
         [&s](double v) { s.vram_gb = std::max(6.0, v); }},
        {"Drafter",
         "Proposes several tokens at once, the model checks them in one pass. Only the Coder has a matching MTP "
         "head.",
         coder ? s.drafter : "none",
         [&s, coder](int d) {
             if (coder) cycle(s.drafter, drafters, d);
         },
         {}},
        {"Effort for Claude Code",
         "medium solved 21 of 23 tasks in 8 min, high 20 of 23 in 38 min "
         "(research/measurements/2026-10-26-effort.md).",
         s.effort,
         [&s](int d) { cycle(s.effort, efforts, d); },
         {}},
    };
}

std::vector<AdvRow> option_rows(Settings& s) {
    const auto toggle = [](bool& b) { return [&b](int) { b = !b; }; };
    return {
        {"Shared experts (OEA)",
         "+4–8 % speed with two conversations, quality within noise. Off only if every output must be independent.",
         on_off(s.oea),
         toggle(s.oea),
         {}},
        {"Subagents think briefly",
         "Subagents answer faster (subagent=low). What that costs in quality is not measured.",
         on_off(!s.role_effort.empty()),
         [&s](int) { s.role_effort = s.role_effort.empty() ? "subagent=low" : ""; },
         {}},
        {"Keep the GPU warm",
         "After half a second of pause the GPU clock drops, the next answer starts about 0.1 s later. On costs "
         "about +27 W (40 instead of 13 W idle).",
         on_off(s.clock_hold),
         toggle(s.clock_hold),
         {}},
        {"Port", "Where the server listens (Anthropic and OpenAI API).", std::to_string(s.port),
         [&s](int d) { s.port = std::max(1024, s.port + d); },
         [&s](double v) { s.port = std::clamp(static_cast<int>(v), 1024, 65535); }},
        {"API key",
         "Without a key every process on the machine can use the engine. tona and the service pass it on "
         "themselves.",
         on_off(s.api_key),
         toggle(s.api_key),
         {}},
        {"Checkpoints on the SSD",
         "Conversations that no longer fit in RAM go to the SSD and are back at once after a restart.",
         gb_text(s.prompt_cache_gb), [&s](int d) { s.prompt_cache_gb = std::max(1.0, s.prompt_cache_gb + d * 2); },
         [&s](double v) { s.prompt_cache_gb = std::max(1.0, v); }},
        {"Command tona", "Links the command tona into ~/.local/bin.", on_off(s.shell), toggle(s.shell), {}},
    };
}

}  // namespace

int Wizard::advanced_rows() const { return 11; }

Element Wizard::advanced_body() {
    std::vector<AdvRow> rows = engine_rows(s(), s().model == "coder-iq1_m");
    for (AdvRow& r : option_rows(s())) rows.push_back(std::move(r));
    Elements l = {ui::title("Advanced")};
    for (size_t i = 0; i < rows.size(); i++) {
        const bool here = static_cast<int>(i) == adv_cursor_;
        const std::string value = here && editing_ ? edit_ + "_" : rows[i].value;
        l.push_back(ui::highlight(ui::plain(ui::cursor(here) + ui::pad(rows[i].label, 32) + value), here));
    }
    l.push_back(ui::blank());
    if (adv_cursor_ < static_cast<int>(rows.size()))
        for (Element& w : ui::wrap(rows[adv_cursor_].help)) l.push_back(std::move(w));
    return vbox(l);
}

namespace {

// A digit starts typing a number into a number row; Enter keeps it, Esc drops it
bool edit_key(const Event& e, bool& editing, std::string& edit, const AdvRow& row) {
    if (e == Event::Return) {
        std::replace(edit.begin(), edit.end(), ',', '.');
        if (!edit.empty()) row.number(std::strtod(edit.c_str(), nullptr));
        return editing = false, true;
    }
    if (e == Event::Escape) return editing = false, true;
    if (e == Event::Backspace) return edit.empty() ? (editing = false, true) : (edit.pop_back(), true);
    if (e.is_character() && e.character().find_first_of("0123456789.,") != std::string::npos)
        return edit += e.character(), true;
    return true;
}

}  // namespace

namespace {

// Arrows and space change a row, a digit starts typing into a number row; true if the key was used
bool row_key(const Event& e, const AdvRow& row, bool& editing, std::string& edit) {
    const bool change_key = ui::is_left(e) || ui::is_right(e) || ui::is_space(e);
    if (row.change && change_key) return row.change(ui::is_left(e) ? -1 : 1), true;
    const bool digit = e.is_character() && e.character().find_first_of("0123456789") != std::string::npos;
    if (row.number && digit) return editing = true, edit = e.character(), true;
    return false;
}

}  // namespace

bool Wizard::key_advanced(const Event& e) {
    const bool used = key_advanced_row(e);
    resolve_context(st_.module, s());  // the RAM limit moves the context of "auto"
    return used;
}

bool Wizard::key_advanced_row(const Event& e) {
    std::vector<AdvRow> rows = engine_rows(s(), s().model == "coder-iq1_m");
    for (AdvRow& r : option_rows(s())) rows.push_back(std::move(r));
    if (editing_) return edit_key(e, editing_, edit_, rows[adv_cursor_]);
    if (ui::is_down(e) && adv_cursor_ + 1 < static_cast<int>(rows.size())) return ++adv_cursor_, true;
    if (ui::is_up(e) && adv_cursor_ > 0) return --adv_cursor_, true;
    if (row_key(e, rows[adv_cursor_], editing_, edit_)) return true;
    if (ui::is_back(e) || e == Event::Return) return view_ = View::Pages, true;
    return true;
}

}  // namespace tona::install
