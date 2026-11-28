// Windows counterparts of the /proc reads in backend/budget.h. Win32 kernel32 calls, no extra libraries needed.
#pragma once

#include <psapi.h>
#include <windows.h>

namespace tona::backend::win32 {

// Private bytes committed by this process: the closest counterpart of RssAnon + RssShmem on Linux
inline long long held_bytes() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof pmc))
        return 0;
    return static_cast<long long>(pmc.PrivateUsage);
}

// Physical memory available without paging out: the counterpart of MemAvailable
inline double available_bytes() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return 0;
    return static_cast<double>(ms.ullAvailPhys);
}

}  // namespace tona::backend::win32
