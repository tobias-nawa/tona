// Stand-in for a compositor in the VRAM pressure test (research/62-compositor-survival-test-plan.md): a
// headless Vulkan client that keeps one 64 MiB device-local buffer and one recorded command buffer, submits a fill of
// the buffer every 100 ms and waits for it. The command submission needs the buffer resident, the path that failed
// for the bar and the terminal on 2026-11-15 ("Not enough memory for command submission").
// It prints one JSON line per submission (time and latency in microseconds, no program or window names) and ends with
// exit code 2 and a line {"error": ...} at the first failed call, which is the result of the test, not an accident.
// Usage: vram_canary [--seconds N] [--stop-file PATH] [--period-ms N] [--priority low|medium|high|realtime]
// --priority sets the global queue priority (VK_KHR_global_priority); high and realtime need CAP_SYS_NICE or the DRM
// master (a compositor is the master): without them the device creation fails and the canary exits with code 2. Build:
// g++ -O2 -std=c++20 vram_canary.cpp -lvulkan -o vram_canary   (no shader, no window, no GPU work beyond a fill)
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr VkDeviceSize kBufferBytes = 64ULL << 20;
constexpr uint64_t kFenceTimeoutNs = 5ULL * 1000 * 1000 * 1000;
constexpr VkDeviceSize kBigHeap = 4ULL << 30;  // the discrete GPU has over 4 GiB; the iGPU has 2

volatile std::sig_atomic_t g_stop = 0;

[[noreturn]] void fail(const char* what, int code) {
    std::printf("{\"error\": \"%s\", \"code\": %d}\n", what, code);
    std::fflush(stdout);
    std::exit(2);
}

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) fail(what, static_cast<int>(r));
}

struct Canary {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t family = 0;
};

VkDeviceSize device_local_bytes(VkPhysicalDevice p) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(p, &mp);
    VkDeviceSize best = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) best = std::max(best, mp.memoryHeaps[i].size);
    return best;
}

void pick_device(Canary& c) {
    uint32_t n = 0;
    check(vkEnumeratePhysicalDevices(c.instance, &n, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> devs(n);
    check(vkEnumeratePhysicalDevices(c.instance, &n, devs.data()), "vkEnumeratePhysicalDevices");
    VkDeviceSize best = kBigHeap;
    for (VkPhysicalDevice p : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(p, &props);
        const VkDeviceSize size = device_local_bytes(p);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && size > best) {
            c.phys = p;
            best = size;
        }
    }
    if (c.phys == VK_NULL_HANDLE) fail("no discrete GPU with more than 4 GiB", 0);
}

void make_device(Canary& c, VkQueueGlobalPriorityKHR priority) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &n, fams.data());
    for (uint32_t i = 0; i < n; i++) {
        if (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {  // a graphics queue: the compositor's kind of submission
            c.family = i;
            break;
        }
    }
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = c.family;
    q.queueCount = 1;
    q.pQueuePriorities = &prio;
    VkDeviceQueueGlobalPriorityCreateInfoKHR gp{VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR};
    gp.globalPriority = priority;
    q.pNext = &gp;
    const char* ext = VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME;
    VkDeviceCreateInfo d{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    d.enabledExtensionCount = 1;
    d.ppEnabledExtensionNames = &ext;
    d.queueCreateInfoCount = 1;
    d.pQueueCreateInfos = &q;
    check(vkCreateDevice(c.phys, &d, nullptr, &c.device), "vkCreateDevice");
    vkGetDeviceQueue(c.device, c.family, 0, &c.queue);
}

// A device-local type without host access: the buffer lives in VRAM only, as a compositor's render target does.
uint32_t vram_type(const Canary& c, uint32_t allowed) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(c.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((allowed & (1u << i)) && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            return i;
    }
    fail("no device-local memory type", 0);
}

void make_buffer(Canary& c) {
    VkBufferCreateInfo b{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    b.size = kBufferBytes;
    b.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(c.device, &b, nullptr, &c.buffer), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c.device, c.buffer, &req);
    VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    a.allocationSize = req.size;
    a.memoryTypeIndex = vram_type(c, req.memoryTypeBits);
    check(vkAllocateMemory(c.device, &a, nullptr, &c.memory), "vkAllocateMemory");
    check(vkBindBufferMemory(c.device, c.buffer, c.memory, 0), "vkBindBufferMemory");
}

void record(Canary& c) {
    VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    p.queueFamilyIndex = c.family;
    check(vkCreateCommandPool(c.device, &p, nullptr, &c.pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    a.commandPool = c.pool;
    a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    a.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(c.device, &a, &c.cmd), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(c.cmd, &begin), "vkBeginCommandBuffer");
    vkCmdFillBuffer(c.cmd, c.buffer, 0, kBufferBytes, 0xCAFE);
    check(vkEndCommandBuffer(c.cmd), "vkEndCommandBuffer");
    VkFenceCreateInfo f{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(c.device, &f, nullptr, &c.fence), "vkCreateFence");
}

void setup(Canary& c, VkQueueGlobalPriorityKHR priority) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    check(vkCreateInstance(&ci, nullptr, &c.instance), "vkCreateInstance");
    pick_device(c);
    make_device(c, priority);
    make_buffer(c);
    record(c);
}

double submit_once(Canary& c) {
    const auto t0 = std::chrono::steady_clock::now();
    VkSubmitInfo s{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    s.commandBufferCount = 1;
    s.pCommandBuffers = &c.cmd;
    check(vkQueueSubmit(c.queue, 1, &s, c.fence), "vkQueueSubmit");
    check(vkWaitForFences(c.device, 1, &c.fence, VK_TRUE, kFenceTimeoutNs), "vkWaitForFences");
    check(vkResetFences(c.device, 1, &c.fence), "vkResetFences");
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

struct Options {
    double seconds = 900;
    int period_ms = 100;
    std::string stop_file;
    VkQueueGlobalPriorityKHR priority = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
};

double number(const char* text) {
    char* end = nullptr;
    const double v = std::strtod(text, &end);
    if (end == text || *end != '\0') fail("not a number", 0);
    return v;
}

VkQueueGlobalPriorityKHR priority_of(const std::string& name) {
    if (name == "low") return VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR;
    if (name == "medium") return VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
    if (name == "high") return VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
    if (name == "realtime") return VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR;
    fail("unknown priority", 0);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--seconds")
            o.seconds = number(argv[i + 1]);
        else if (k == "--period-ms")
            o.period_ms = static_cast<int>(number(argv[i + 1]));
        else if (k == "--priority")
            o.priority = priority_of(argv[i + 1]);
        else if (k == "--stop-file")
            o.stop_file = argv[i + 1];
        else
            fail("unknown option", 0);
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    std::signal(SIGTERM, [](int) { g_stop = 1; });
    Canary c;
    setup(c, opt.priority);
    const auto start = std::chrono::steady_clock::now();
    std::printf("{\"event\": \"ready\", \"buffer_mib\": %llu}\n", static_cast<unsigned long long>(kBufferBytes >> 20));
    std::fflush(stdout);
    while (!g_stop) {
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (t > opt.seconds || (!opt.stop_file.empty() && std::filesystem::exists(opt.stop_file))) break;
        const double us = submit_once(c);
        std::printf("{\"t\": %.3f, \"us\": %.0f}\n", t, us);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(opt.period_ms));
    }
    std::printf("{\"event\": \"end\"}\n");
    return 0;
}
