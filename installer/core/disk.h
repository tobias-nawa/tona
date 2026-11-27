// The drive under work/: which device, which kind, how fast. With little RAM the engine reads experts and, for every
// token, rows of the n-gram table (PLE, 28.8 GB) at random offsets from it; then the drive decides the speed.
#pragma once

#include <string>

namespace tona::install {

struct Disk {
    std::string device;  // e.g. "nvme0n1"
    std::string model;
    std::string kind;  // "nvme", "ssd", "hdd" or empty (unknown)
};

struct DiskSpeed {
    bool measured = false;
    double rand_iops = 0;    // random 4 KiB reads per second, 32 in flight, bypassing the page cache
    double rand_lat_us = 0;  // mean duration of one such read
    double seq_mb_s = 0;     // sequential 1 MiB reads
    std::string file;        // measured on this file
    // Reads of 2 MiB blocks with 16 in flight, as the engine reads experts: the Mac backend's drive figure (macOS only,
    // 0: not measured). Sequential single reads are no stand-in: they run at a fraction of this on Apple's SSD.
    double expert_mb_s = 0;
};

// largest .gguf under models_dir; empty if there is none of at least 1 GiB
std::string largest_model_file(const std::string& models_dir);
// device behind a path (through /proc/self/mountinfo, device mapper and partitions down to the disk)
Disk detect_disk(const std::string& path);
// measures on an existing large file (read only, O_DIRECT; F_NOCACHE on macOS); empty file: largest .gguf under
// repo/work/models
DiskSpeed measure_disk(const std::string& repo, std::string file = {}, double seconds = 3);

}  // namespace tona::install
