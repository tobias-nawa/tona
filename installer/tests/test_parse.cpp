// Parsers for system text (core/parse.h) on invented machines: cpuinfo, os-release, meminfo, mountinfo, KFD node
// properties and nvidia-smi output.
#include <cstdio>
#include <exception>

#include "core/parse.h"

using namespace tona::install::parse;

namespace {

int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "FAIL: %s\n", what), fails++;
}

const char* const kCpuInfo =
    "processor\t: 0\n"
    "vendor_id\t: AuthenticAMD\n"
    "cpu family\t: 26\n"
    "model name\t: AMD Ryzen 5 9600X 6-Core Processor\n"
    "cpu cores\t: 6\n"
    "flags\t\t: fpu vme avx2 avx512f avx512dq avx512_vnni avx512_bf16\n"
    "\n"
    "processor\t: 1\n"
    "model name\t: second block is never read\n";

void test_field() {
    expect(field(kCpuInfo, "vendor_id", ':') == "AuthenticAMD", "key with tab before the colon");
    expect(field(kCpuInfo, "model name", ':') == "AMD Ryzen 5 9600X 6-Core Processor", "the first block wins");
    expect(field(kCpuInfo, "cpu", ':').empty(), "a prefix of a key is not the key");
    expect(field(kCpuInfo, "missing", ':').empty(), "missing key");
    expect(field("", "x", ':').empty(), "empty text");
    const char* osr =
        "NAME=\"Fedora Linux\"\nID=fedora\nID_LIKE=\"rhel centos\"\nPRETTY_NAME=\"Fedora Linux 44\"\nVERSION='x'\n";
    expect(field(osr, "ID", '=') == "fedora" && field(osr, "ID_LIKE", '=') == "rhel centos",
           "os-release ID and ID_LIKE");
    expect(field(osr, "PRETTY_NAME", '=') == "Fedora Linux 44", "quotes are removed");
    expect(field(osr, "VERSION", '=') == "'x'", "single quotes stay");
    expect(field("a = \"\"\n", "a", '=').empty(), "empty quoted value");
    expect(field("key=\"\n", "key", '=') == "\"", "a lone quote stays");
    expect(field("MemTotal:       65718552 kB\nMemAvailable:   52000000 kB\n", "MemAvailable", ':') == "52000000 kB",
           "meminfo keeps the unit");
    expect(field("gfx_target_version 120000\ndrm_render_minor 128\n", "drm_render_minor", ' ') == "128",
           "KFD properties use a space");
}

void test_num() {
    expect(num("42") == 42 && num("-7") == -7, "decimal");
    expect(num("0x1002") == 0x1002 && num("0x10de") == 0x10de, "hexadecimal, as in the sysfs vendor files");
    expect(num("65718552 kB") == 65718552, "trailing unit is ignored");
    expect(num("") == 0 && num("abc") == 0, "no number: 0");
    expect(num("010") == 8, "leading zero is octal (stoll base 0)");
}

void test_flags() {
    const std::string flags = field(kCpuInfo, "flags", ':');
    expect(has_flag(flags, "avx512f") && has_flag(flags, "avx512_vnni") && has_flag(flags, "fpu"), "present flags");
    expect(!has_flag(flags, "avx512") && !has_flag(flags, "avx") && !has_flag(flags, "vnni"), "no partial matches");
    expect(!has_flag("", "avx512f"), "no flags");
    expect(has_flag("avx512f", "avx512f"), "single flag");
}

const char* const kMountInfo =
    "22 1 0:21 / /sys rw,nosuid shared:2 - sysfs sysfs rw\n"
    "30 1 259:3 /root / rw,relatime shared:1 - btrfs /dev/nvme0n1p3 rw,subvol=/root\n"
    "31 30 259:3 /home /home rw,relatime shared:1 - btrfs /dev/nvme0n1p3 rw,subvol=/home\n"
    "40 30 8:17 / /mnt/data rw,relatime - ext4 /dev/sdb1 rw\n"
    "41 30 0:44 / /mnt/data2 rw - tmpfs tmpfs rw\n"
    "42 30 253:0 / /srv rw,relatime shared:5 master:9 - xfs /dev/mapper/vg-srv rw\n";

void test_mount_source() {
    expect(mount_source(kMountInfo, "/") == "/dev/nvme0n1p3", "the root file system");
    expect(mount_source(kMountInfo, "/usr/lib") == "/dev/nvme0n1p3", "a path below the root");
    expect(mount_source(kMountInfo, "/home/user/work") == "/dev/nvme0n1p3", "a nested mount");
    expect(mount_source(kMountInfo, "/mnt/data") == "/dev/sdb1", "the mount point itself");
    expect(mount_source(kMountInfo, "/mnt/data/models/x.gguf") == "/dev/sdb1", "below a mount point");
    expect(mount_source(kMountInfo, "/mnt/data2/x") == "tmpfs",
           "a mount point that only shares a prefix is another one");
    expect(mount_source(kMountInfo, "/mnt/dat") == "/dev/nvme0n1p3", "a prefix of a mount point is not inside it");
    expect(mount_source(kMountInfo, "/srv/a") == "/dev/mapper/vg-srv",
           "optional fields before the dash (shared, master)");
    expect(mount_source("", "/x").empty(), "no mounts");
    expect(mount_source("garbage line\n\n", "/x").empty(), "unparsable lines");
}

void test_gfx_name() {
    expect(gfx_name(120000) == "gfx1200", "RDNA4 (reference)");
    expect(gfx_name(110000) == "gfx1100", "RDNA3");
    expect(gfx_name(90010) == "gfx90a", "CDNA-style versions use hex digits");
    expect(gfx_name(100302) == "gfx1032", "RDNA2");
    expect(gfx_name(90012) == "gfx90c", "hex minor");
}

void test_nvidia_rows() {
    const auto rows = nvidia_rows("NVIDIA GeForce RTX 4090, 24564, 1021\nNVIDIA RTX A2000, 12282, 0\n");
    expect(rows.size() == 2, "two cards");
    expect(rows[0].name == "NVIDIA GeForce RTX 4090" && rows[0].total_mib == 24564 && rows[0].used_mib == 1021,
           "first");
    expect(rows[1].name == "NVIDIA RTX A2000" && rows[1].total_mib == 12282 && rows[1].used_mib == 0, "second");
    expect(nvidia_rows("").empty(), "no output");
    const auto odd = nvidia_rows("only a name\n");
    expect(odd.size() == 1 && odd[0].total_mib == 0, "a short line gives zeros, not a crash");
}

}  // namespace

int main() {
    try {
        test_field();
        test_num();
        test_flags();
        test_mount_source();
        test_gfx_name();
        test_nvidia_rows();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }
    if (!fails) std::printf("ok\n");
    return fails ? 1 : 0;
}
