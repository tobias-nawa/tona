// Host and Origin checks of the local server (DNS rebinding, cross-site requests)
#include "server/request_guard.h"
#include "tests/test_support.h"

namespace {

using tona::server::host_allowed;
using tona::server::origin_allowed;

void test_host() {
    EXPECT(host_allowed("127.0.0.1:8662", 8662));
    EXPECT(host_allowed("localhost:8662", 8662));
    EXPECT(host_allowed("LOCALHOST:8662", 8662));
    EXPECT(host_allowed("[::1]:8662", 8662));
    EXPECT(!host_allowed("127.0.0.1:8663", 8662));
    EXPECT(!host_allowed("127.0.0.1", 8662));
    EXPECT(host_allowed("localhost", 80));
    // A rebound name resolves to 127.0.0.1 but the browser sends the attacker's name
    EXPECT(!host_allowed("evil.example:8662", 8662));
    EXPECT(!host_allowed("localhost.evil.example:8662", 8662));
    EXPECT(!host_allowed("127.0.0.1.nip.io:8662", 8662));
    EXPECT(!host_allowed("", 8662));
    EXPECT(!host_allowed("[::1", 8662));
    EXPECT(!host_allowed("localhost:8662:1", 8662));
}

void test_origin() {
    EXPECT(origin_allowed("", 8662, {}));
    EXPECT(origin_allowed("http://127.0.0.1:8662", 8662, {}));
    EXPECT(origin_allowed("http://localhost:8662", 8662, {}));
    EXPECT(!origin_allowed("https://evil.example", 8662, {}));
    EXPECT(!origin_allowed("http://localhost:3000", 8662, {}));
    EXPECT(!origin_allowed("null", 8662, {}));
    EXPECT(!origin_allowed("http://127.0.0.1:8662/path", 8662, {}));
    EXPECT(origin_allowed("http://localhost:3000", 8662, {"http://localhost:3000"}));
    EXPECT(origin_allowed("vscode-webview://abc", 8662, {"VSCODE-WEBVIEW://abc"}));
}

void test_list() {
    using tona::server::parse_origin_list;
    EXPECT(parse_origin_list(nullptr).empty());
    EXPECT(parse_origin_list("").empty());
    const auto l = parse_origin_list(" http://a:1 ,, http://b:2");
    EXPECT(l.size() == 2 && l[0] == "http://a:1" && l[1] == "http://b:2");
}

void test_bind() {
    using tona::server::is_loopback_bind;
    EXPECT(is_loopback_bind("127.0.0.1") && is_loopback_bind("localhost") && is_loopback_bind("::1"));
    EXPECT(!is_loopback_bind("0.0.0.0") && !is_loopback_bind("192.168.1.5"));
}

}  // namespace

int main() {
    return tona::test::run("test_request_guard", [] {
        test_host();
        test_origin();
        test_list();
        test_bind();
    });
}
