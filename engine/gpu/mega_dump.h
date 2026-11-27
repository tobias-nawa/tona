// Diagnostic: TONA_MEGA_DUMP_PROGRAM=<file> writes every op program Mega::build() produced as text. Pointers are
// printed as "<region>+<offset>" (region = allocation order index), so the dumps of two builds of the engine
// compare equal with cmp when the programs are the same.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/mega.h"

namespace tona::gpu {

struct PfDumpSink;

class ProgramDump {
public:
    ProgramDump(const Mega& g, const char* path);
    ~ProgramDump();
    ProgramDump(const ProgramDump&) = delete;
    ProgramDump& operator=(const ProgramDump&) = delete;

    // ops of one program with their labels
    void ops(std::string_view name, const std::vector<MegaOp>& ops, const std::vector<std::string>& labels);
    // the matvec tasks as the device gets them
    template <class Dev>
    void dev_tasks(std::string_view name, const std::vector<Dev>& dev) {
        refresh();
        std::fprintf(f_, "tasks %.*s %zu\n", (int)name.size(), name.data(), dev.size());
        for (size_t i = 0; i < dev.size(); i++) {
            const Dev& t = dev[i];
            std::fprintf(
                f_, "T%zu type=%u in=%d rows=%lld units=%lld rb=%lld base=%lld L=%d w=%s x=%s y=%s xq=%s xm=%s\n", i,
                t.type, t.in, (long long)t.rows, (long long)t.units, (long long)t.row_bytes, (long long)t.base, t.L,
                ptr(t.w).c_str(), ptr(t.x).c_str(), ptr(t.y).c_str(), ptr(t.xq).c_str(), ptr(t.xm).c_str());
            std::fprintf(f_,
                         "  slot_e=%d slot_stride=%lld y_pos=%lld act=%a n_exp=%d upe=%d x_e=%lld y_e=%lld xb=%lld "
                         "yb=%lld tsplit=%d wide=%d\n",
                         t.slot_e, (long long)t.slot_stride, (long long)t.y_pos_stride, (double)t.act_scale, t.n_exp,
                         t.upe, (long long)t.x_e, (long long)t.y_e, (long long)t.xb, (long long)t.yb, t.tsplit, t.wide);
        }
    }
    // a table uploaded next to the program (dependency masks), as 32-bit words
    void table(std::string_view name, const void* data, size_t bytes);
    // sizes and settings the programs depend on, the prefill records, the programs of several sequences
    void state();

private:
    struct Region {
        uintptr_t base;
        std::string name;
    };
    std::string ptr(const void* p) const;
    void refresh();
    void pf_weight(const char* what, const PfW& w);
    void pf_layers();
    PfDumpSink sink();
    void patches(const char* name, const std::vector<XPatch>& v);

    const Mega& g_;
    FILE* f_;
    std::vector<Region> regions_;
};

}  // namespace tona::gpu
