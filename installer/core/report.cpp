#include "core/report.h"

#include <format>

#include "core/context_plan.h"
#include "core/flow.h"
#include "core/speed.h"

namespace tona::install {

std::string gb(double bytes) { return std::format("{:.1f} GB", bytes / 1e9); }

std::string hardware_text(const Hardware& h, const Module* m) {
    std::string o = std::format("System:  {}\n", h.distro_name);
    o += std::format("CPU:     {} ({} cores, {} threads{})\n", h.cpu, h.cores, h.threads, h.avx512 ? ", AVX-512" : "");
    o += std::format("RAM:     {} ({} free){}\n", gb(h.ram_total), gb(h.ram_avail),
                     h.ram_gbs > 0 ? std::format(", reads {:.0f} GB/s", h.ram_gbs) : "");
    for (const Gpu& g : h.gpus)
        o += g.vendor == "Apple"
                 ? std::format("GPU:     {} (shares the RAM), GPU working set {}\n", g.name, gb(g.vram_total))
                 : std::format("GPU:     {} {} [{:04x}:{:04x}]{}{}, VRAM {} ({} used){}\n", g.vendor,
                               g.name.empty() ? "?" : g.name, g.pci_vendor, g.pci_device, g.arch.empty() ? "" : " ",
                               g.arch, gb(g.vram_total), gb(g.vram_used), g.integrated ? ", integrated" : "");
    const std::string drive =
        h.disk.kind.empty() ? h.disk.model : h.disk.kind + (h.disk.model.empty() ? "" : " " + h.disk.model);
    o += std::format("Space:   {} free under {}{}\n", gb(h.disk_free), h.disk_path.empty() ? "work/" : h.disk_path,
                     drive.empty() ? "" : " (" + drive + ")");
    if (h.disk_speed.measured)
        o += std::format("Drive:   {:.0f}K random reads/s (4 KiB, {:.0f} µs each), {:.0f} MB/s sequential\n",
                         h.disk_speed.rand_iops / 1e3, h.disk_speed.rand_lat_us, h.disk_speed.seq_mb_s);
    o += std::format("Backend: {}\n", m ? m->title() : "none for this hardware");
    return o;
}

std::string settings_text(const Settings& s) {
    std::string o;
    o += std::format("RAM limit        {:g} GB (page cache included)\n", s.ram_gb);
    o += std::format("VRAM limit       {:g} GB\n", s.vram_gb);
    const std::vector<std::string> all = models_of(s);
    o += std::format("Model            {}{}\n", s.model, all.size() > 1 ? " (starts the server)" : "");
    if (all.size() > 1) o += std::format("Also downloaded  {}\n", join_list({all.begin() + 1, all.end()}));
    if (!s.languages.empty()) o += std::format("Languages        {}\n", join_list(s.languages));
    o += std::format("Use              {}, priority in a shortage: {}\n", s.use, s.priority);
    o += std::format("Drafter          {}\n", s.drafter);
    o += std::format("Agents           {} in parallel\n", s.seqs);
    o += std::format("Context          {} per agent{}\n", context_text(s.ctx), s.ctx_auto ? " (auto)" : "");
    o += std::format("Port             {}\n", s.port);
    o += std::format("Effort           {}{}\n", s.effort, s.role_effort.empty() ? "" : ", per role " + s.role_effort);
    o += std::format("OEA              {}\n", s.oea ? "on" : "off");
    o += std::format("Keep GPU warm    {}\n", s.clock_hold ? "yes (about +27 W while held)" : "no");
    o += std::format("API key          {}\n", s.api_key ? "yes" : "no");
    o += std::format("Checkpoints      {:g} GB on the SSD\n", s.prompt_cache_gb);
    o += std::format("Service          {}\n", s.service ? "yes, starts at login" : "no");
    o += std::format("Command tona       {}\n", s.shell ? "yes" : "no");
    return o;
}

std::string estimate_text(const Estimate& e) {
    const offer::ModelCard* c = offer::model_card(e.model);
    const std::string name = c ? c->name : e.model;
    if (e.decode.hi <= 0) return std::format("{}: unknown ({})\n", name, e.setup);
    std::string rates = std::format("decode {} tok/s at {}K context", range_text(e.decode),
                                    e.context_tokens % 1000 == 0 ? e.context_tokens / 1000 : e.context_tokens / 1024);
    if (e.prefill.hi > 0) rates += std::format(", prefill {} tok/s", range_text(e.prefill));
    const std::string load = e.load_s <= 0    ? ""
                             : e.load_s < 120 ? std::format(", start ~{:.0f} s", e.load_s)
                                              : std::format(", start ~{:.0f} min", e.load_s / 60);
    std::string unknown;
    for (const std::string& k : e.unknown) unknown += (unknown.empty() ? "" : ", ") + k;
    return std::format("{}: {}{}, estimate +-{:.0f} % (anchor: measured on {}{})\n", name, rates, load, e.width * 100,
                       e.anchor, unknown.empty() ? "" : "; not found: " + unknown);
}

std::string packages_text(const Hardware& h, const std::vector<Package>& p) {
    std::string o;
    for (const Package& k : p) {
        const char* mark = k.found && k.problem.empty() ? "ok     " : k.required ? "MISSING" : "missing";
        o += std::format("  {} {:<10} {}{}\n", mark, k.id, k.problem.empty() ? k.purpose : k.problem,
                         k.version.empty() ? "" : " (" + k.version + ")");
        if ((!k.found || !k.problem.empty()) && !k.hint.empty()) o += "          → " + k.hint + "\n";
    }
    if (const std::string c = install_command(h, p); !c.empty()) o += "  Install the missing: " + c + "\n";
    return o;
}

std::string steps_text(const std::vector<Step>& st) {
    std::string o;
    for (size_t i = 0; i < st.size(); i++)
        o += std::format("  {}. {}{}\n     {}\n", i + 1, st[i].title, st[i].done ? " (already done)" : "", st[i].cmd);
    return o;
}

bool packages_ok(const std::vector<Package>& p) {
    for (const Package& k : p)
        if (k.required && (!k.found || !k.problem.empty())) return false;
    return true;
}

}  // namespace tona::install
