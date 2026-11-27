// Stands in for tona-server in the updater's end-to-end test (test_update_script_service): a process named
// tona-server that answers /status as a loaded, idle server on --port. Ends itself after two minutes, so a test that
// dies never leaves it behind for long.
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include "third_party/httplib.h"

int main(int argc, char** argv) try {
    int port = 0;
    for (int i = 1; i + 1 < argc; i++)
        if (std::string(argv[i]) == "--port") {
            const std::string v = argv[i + 1];
            std::from_chars(v.data(), v.data() + v.size(), port);
        }
    if (port <= 0) return 2;
    httplib::Server svr;
    svr.Get("/status", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"ready":true,"engine":{"state":"idle","queue":0,"active":[]}})", "application/json");
    });
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::minutes(2));
        std::_Exit(0);
    }).detach();
    return svr.listen("127.0.0.1", port) ? 0 : 1;
} catch (...) {
    return 1;
}
