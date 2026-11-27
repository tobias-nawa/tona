// gpu/kfd_smi.h: parsing of the KFD SMI event lines, on the CPU. The lines are the formats of KFD_EVENT_FMT_* in
// include/uapi/linux/kfd_ioctl.h, with the event number and a space in front (kfd_smi_event_add).
#include "gpu/kfd_smi.h"
#include "tests/test_support.h"

using namespace tona::gpu::smi;

namespace {

void test_queue_events() {
    Line l;
    // QUEUE_EVICTION "%lld -%d %x %d" with trigger 2 (TTM)
    EXPECT(parse_line("9 123456789 -4242 d3f1 2\n", l));
    EXPECT(l.event == kQueueEviction && l.ns == 123456789);
    EXPECT(l.pid == 4242 && l.trigger == 2);
    EXPECT(format_log_line(123457789, l) ==
           "123457789 QUEUE_EVICTION trigger=ttm kernel_ns=123456789 latency_us=1 | 123456789 -4242 d3f1 2");
    // QUEUE_RESTORE "%lld -%d %x %c": '0' or 'R' (rescheduled)
    EXPECT(parse_line("a 99 -7 1 0", l));
    EXPECT(l.event == kQueueRestore && l.tag == '0' && l.pid == 7 && l.ns == 99);
    EXPECT(parse_line("a 99 -7 1 R", l));
    EXPECT(l.tag == 'R');
}

void test_other_events() {
    Line l;
    // MIGRATE_END "%lld -%d @%lx(%lx) %x->%x %d %d", UNMAP "%lld -%d @%lx(%lx) %x %d"
    EXPECT(parse_line("6 500 -12 @7f00(10) 1->0 3 0", l));
    EXPECT(l.event == kMigrateEnd && l.ns == 500 && l.pid == 12);
    EXPECT(parse_line("b 600 -12 @7f00(10) 1 1", l));
    EXPECT(l.event == kUnmapFromGpu && l.ns == 600);
    // Events without a timestamp keep the defaults; the text stays in rest
    EXPECT(parse_line("1 1a2b:kitty\n", l));
    EXPECT(l.event == kVmFault && l.ns == -1 && l.rest == "1a2b:kitty");
    EXPECT(parse_line("3 7 reset", l));
    EXPECT(l.event == kGpuPreReset && l.ns == -1);
}

void test_bad_lines() {
    Line l;
    EXPECT(!parse_line("", l));
    EXPECT(!parse_line("\n", l));
    EXPECT(!parse_line("zz", l));
    EXPECT(!parse_line("0 1", l));
    // Damaged payload: the event is still known, the fields are not invented
    EXPECT(parse_line("9 12", l));
    EXPECT(l.event == kQueueEviction && l.pid == 0 && l.trigger == -1);
    EXPECT(parse_line("9 12 -x", l));
    EXPECT(l.pid == 0);
}

void test_masks() {
    // Bit i-1; the logged set holds eviction and restore, not the page-fault events 7 and 8
    EXPECT(mask_of(kVmFault) == 1);
    EXPECT(mask_of(kQueueEviction) == (1ULL << 8));
    EXPECT(kLoggedEvents & mask_of(kQueueEviction));
    EXPECT(kLoggedEvents & mask_of(kQueueRestore));
    EXPECT(!(kLoggedEvents & mask_of(7)));
    EXPECT(!(kLoggedEvents & mask_of(8)));
}

void test_splitter() {
    // A read may end inside a line
    LineSplitter sp;
    auto a = sp.feed("9 1 -2 3 2\na 5 -2 3");
    EXPECT(a.size() == 1 && a[0] == "9 1 -2 3 2");
    auto b = sp.feed(" 0\n9 6 -2 3 2\n");
    EXPECT(b.size() == 2 && b[0] == "a 5 -2 3 0" && b[1] == "9 6 -2 3 2");
    EXPECT(sp.feed("partial").empty());
    EXPECT(sp.feed("").empty());
    EXPECT(std::string(eviction_trigger_name(2)) == "ttm");
    EXPECT(std::string(eviction_trigger_name(99)) == "unknown");
}

}  // namespace

int main() {
    test_queue_events();
    test_other_events();
    test_bad_lines();
    test_masks();
    test_splitter();
    return tona::test::finish("kfd_smi");
}
