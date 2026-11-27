#include "server/job.h"

#include <algorithm>

namespace tona::server {

namespace {

// System prompt text of a request body, from the system field and mid-history system messages
std::string system_text(const json& body) {
    std::string sys;
    auto add = [&](const json& c) {
        if (c.is_string())
            sys += c.get<std::string>();
        else if (c.is_array())
            for (const auto& b : c)
                if (b.is_object() && b.contains("text") && b["text"].is_string()) sys += b["text"].get<std::string>();
    };
    if (body.contains("system")) add(body["system"]);
    if (body.contains("messages") && body["messages"].is_array())
        for (const auto& m : body["messages"])
            if (m.is_object() && m.value("role", "") == "system" && m.contains("content")) add(m["content"]);
    return sys;
}

// Name of the last path component after "Primary working directory: ", empty if absent
std::string project_from_system(const std::string& sys) {
    const std::string key = "Primary working directory: ";
    const size_t p = sys.find(key);
    if (p == std::string::npos) return "";
    std::string dir = sys.substr(p + key.size(), sys.find('\n', p) - p - key.size());
    while (!dir.empty() && (dir.back() == '/' || dir.back() == ' ' || dir.back() == '\r')) dir.pop_back();
    return dir.substr(dir.find_last_of('/') + 1);
}

}  // namespace

std::deque<std::shared_ptr<Job>>::iterator next_job(std::deque<std::shared_ptr<Job>>& q, bool busy) {
    auto it = std::min_element(q.begin(), q.end(), [](const auto& a, const auto& b) { return a->prio() < b->prio(); });
    return it != q.end() && (*it)->warm && busy ? q.end() : it;
}

void tag_origin(Job& j, const httplib::Request& req, const json& body) {
    j.session = req.get_header_value("x-claude-code-session-id");
    j.agent = req.get_header_value("x-claude-code-agent-id");
    const std::string cls = req.get_header_value("x-claude-code-request-class");
    j.role = cls == "auxiliary" || cls == "subagent" ? cls : !j.agent.empty() ? "subagent" : "main";
    j.project = req.get_header_value("x-tona-project");
    if (!j.project.empty()) return;
    j.project = project_from_system(system_text(body));
}

}  // namespace tona::server
