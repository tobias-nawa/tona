// Keys: bytes from the terminal to Key values, and the reducer for the keys that work on every page and in the dialogs.
// The pages' own keys are in keys_pages.cpp.
#include <algorithm>

#include "client/keys_internal.h"
#include "client/text.h"

namespace tona::client {

namespace {

// ESC [ A..D arrows, ESC [ Z shift+tab, ESC O A..D the same in application mode; a lone ESC is Esc.
// Returns how many bytes after the ESC belong to the key, or -1 for a sequence we do not know.
int escape_key(const std::string& bytes, size_t i, Key& out) {
    const bool seq = i + 2 < bytes.size() && (bytes[i + 1] == '[' || bytes[i + 1] == 'O');
    if (!seq) return (out = {KeyCode::Esc, 0}, 0);
    switch (bytes[i + 2]) {
        case 'A':
            return (out = {KeyCode::Up, 0}, 2);
        case 'B':
            return (out = {KeyCode::Down, 0}, 2);
        case 'C':
            return (out = {KeyCode::Right, 0}, 2);
        case 'D':
            return (out = {KeyCode::Left, 0}, 2);
        case 'Z':
            return (out = {KeyCode::BackTab, 0}, 2);
        default:
            return -1;
    }
}

Key plain_key(unsigned char c) {
    if (c == '\r' || c == '\n') return {KeyCode::Enter, 0};
    if (c == 0x7f || c == 0x08) return {KeyCode::Backspace, 0};
    if (c == '\t') return {KeyCode::Tab, 0};
    if (c >= 0x20) return {KeyCode::Char, (char)c};
    return {KeyCode::None, 0};
}

}  // namespace

std::vector<Key> parse_keys(const std::string& bytes) {
    std::vector<Key> keys;
    for (size_t i = 0; i < bytes.size(); i++) {
        Key k{KeyCode::None, 0};
        if ((unsigned char)bytes[i] == 0x1b) {
            const int skip = escape_key(bytes, i, k);
            if (skip < 0) break;  // an escape sequence we do not know: drop the rest of this read
            i += (size_t)skip;
        } else {
            k = plain_key((unsigned char)bytes[i]);
        }
        if (k.code != KeyCode::None) keys.push_back(k);
    }
    return keys;
}

namespace {

Action act(ActionKind k, std::string arg = "") { return {k, std::move(arg)}; }

// What Enter does in a Confirm dialog
Action confirmed(UiState& s) {
    const Ask ask = s.ask;
    const std::string arg = s.ask_arg;
    s.modal = Modal::None, s.ask = Ask::None;
    switch (ask) {
        case Ask::SwitchModel:
            return act(ActionKind::SwitchModel, arg);
        case Ask::Restart:
            return act(ActionKind::RestartServer);
        case Ask::Update:
            return act(ActionKind::UpdateKey, "u");
        case Ask::Remove:
            return act(ActionKind::Remove, arg);
        case Ask::StopServer:
            return act(ActionKind::StopServer);
        case Ask::None:
            break;
    }
    return {};
}

Action handle_confirm(UiState& s, Key key) {
    if (key.code == KeyCode::Enter) return confirmed(s);
    if (key.code == KeyCode::Esc || key.code == KeyCode::Backspace || (key.code == KeyCode::Char && key.ch == 'n'))
        s.modal = Modal::None, s.ask = Ask::None;
    return {};
}

// The Close dialog: options as close_options() orders them; Space remembers the choice
Action close_choice(UiState& s, int option) {
    s.modal = Modal::None;
    const std::string arg = s.remember_choice ? "remember" : "";
    if (option == 0) return act(ActionKind::Quit, arg);
    if (option == 1) return act(ActionKind::QuitAndStop, arg);
    return {};
}

Action handle_close(UiState& s, Key key) {
    const std::vector<int> options = close_options(s);
    if (key.code == KeyCode::Up) s.modal_cursor = std::max(0, s.modal_cursor - 1);
    if (key.code == KeyCode::Down) s.modal_cursor = std::min((int)options.size() - 1, s.modal_cursor + 1);
    if (key.code == KeyCode::Esc || key.code == KeyCode::Backspace) return close_choice(s, 2);
    if (key.code == KeyCode::Enter) return close_choice(s, options[(size_t)s.modal_cursor]);
    if (key.code == KeyCode::Char && key.ch == 's') return close_choice(s, 1);
    if (key.code == KeyCode::Char && key.ch == ' ') s.remember_choice = !s.remember_choice;
    return {};
}

Action handle_modal(UiState& s, Key key) {
    switch (s.modal) {
        case Modal::Help:
            if (key.code == KeyCode::Esc || key.code == KeyCode::Backspace ||
                (key.code == KeyCode::Char && key.ch == '?'))
                s.modal = Modal::None;
            return {};
        case Modal::Close:
            return handle_close(s, key);
        case Modal::Confirm:
            return handle_confirm(s, key);
        case Modal::None:
            break;
    }
    return {};
}

void edit_text(std::string& text, Key key) {
    if (key.code == KeyCode::Backspace) {
        // Remove one UTF-8 character: continuation bytes (10xxxxxx) go with their first byte
        while (!text.empty() && ((unsigned char)text.back() & 0xc0) == 0x80) text.pop_back();
        if (!text.empty()) text.pop_back();
    } else if (key.code == KeyCode::Char) {
        text += key.ch;
    }
}

// Leaves the project field; returns whether the folder text may have changed
bool leaves_field(Key key) {
    return key.code == KeyCode::Esc || key.code == KeyCode::Enter || key.code == KeyCode::Tab ||
           key.code == KeyCode::BackTab || key.code == KeyCode::Up || key.code == KeyCode::Down;
}

Action handle_text_field(UiState& s, Key key) {
    AgentForm& f = s.agents;
    if (!leaves_field(key)) {
        edit_text(f.project, key);
        return {};
    }
    f.field = key.code == KeyCode::Up || key.code == KeyCode::BackTab || key.code == KeyCode::Esc ? AgentForm::Agent
                                                                                                  : AgentForm::Model;
    return {ActionKind::ProjectChanged, ""};
}

void go_tab(UiState& s, Tab tab) { s.tab = tab, s.message.clear(); }

Action ask_confirm(UiState& s, Ask ask, std::string arg) {
    s.modal = Modal::Confirm, s.ask = ask, s.ask_arg = std::move(arg);
    return {};
}

// Close: the saved choice (Settings > Start) decides without asking; a remote server is never ours to stop
Action handle_quit(UiState& s) {
    if (!s.server.up) return act(ActionKind::Quit);
    const std::string choice = get_value(s.config, "close");
    if (choice == "keep") return act(ActionKind::Quit);
    if (choice == "stop") return act(ActionKind::QuitAndStop);
    s.modal = Modal::Close, s.modal_cursor = 0, s.remember_choice = false;
    return {};
}

Action handle_global(UiState& s, Key key) {
    if (key.code == KeyCode::Esc || key.code == KeyCode::Backspace) {
        // While a model loads after a switch, Esc on the Models page takes the switch back
        if (s.tab == Tab::Models && !s.switching_from.empty()) return act(ActionKind::SwitchModel, s.switching_from);
        if (s.tab != Tab::Start) go_tab(s, Tab::Start);
        return {};
    }
    if (key.code != KeyCode::Char) return {};
    switch (key.ch) {
        case '1':
            go_tab(s, Tab::Start);
            return {};
        case '2':
            go_tab(s, Tab::Models);
            return {};
        case '3':
            go_tab(s, Tab::Agents);
            return {};
        case '4':
            go_tab(s, Tab::Settings);
            return {};
        case '?':
            s.modal = Modal::Help;
            return {};
        case 'q':
            return handle_quit(s);
        case 'w':
            return switch_key(s);
        default:
            break;
    }
    return {};
}

}  // namespace

Action handle_key(UiState& s, Key key) {
    if (s.modal != Modal::None) return handle_modal(s, key);
    if (in_text_field(s)) return handle_text_field(s, key);
    Action out;
    if (handle_page(s, key, out)) return out;
    return handle_global(s, key);
}

Action ask_for(UiState& s, Ask ask, const std::string& arg) { return ask_confirm(s, ask, arg); }

}  // namespace tona::client
