// The CU count of the GPU before the HIP runtime starts, and the HSA_CU_MASK for TONA_FREE_CUS. The mask has to be set
// before hipInit, so the count cannot come from hipDeviceGetAttribute; the kernel driver's topology in sysfs has it.
// The functions that parse and format are pure (unit-tested without a GPU); the sysfs side is gpu/cu_mask.cpp.
#pragma once

#include <string>
#include <string_view>

namespace tona::gpu {

// The value of "<key> <number>" in a KFD node's properties file; -1 if the key is missing.
long long kfd_property(std::string_view properties, std::string_view key);

// CUs of one KFD node from its properties (simd_count / simd_per_cu); 0 for a CPU node or an unreadable one.
int kfd_node_cus(std::string_view properties);

// The CU count of the first GPU node in the topology below `root` (/sys/class/kfd/kfd/topology/nodes): the order of
// the nodes is the order of the GPUs for HSA_CU_MASK ("0:" is the first one). 0 if no GPU node can be read.
int first_gpu_cus(const std::string& root = "/sys/class/kfd/kfd/topology/nodes");

// "0:0-<cus-free-1>" for HSA_CU_MASK: the lowest CUs stay with the process, the `free` highest go to the desktop.
// Empty if `free` leaves no CU or nothing is to be freed.
std::string cu_mask_string(int cus, int free);

}  // namespace tona::gpu
