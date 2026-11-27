// Download limits of the updater against a local server: size, truncation, redirects, schemes, certificates, timeouts.
#include <string>

#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "tests/update_testtls.h"
#include "update/fetch.h"

using namespace tona::test;
namespace {

up::FetchConfig loopback() {
    up::FetchConfig cfg;
    cfg.allow_http_loopback = true;
    cfg.connect_seconds = 3;
    cfg.read_seconds = 5;
    return cfg;
}

std::string url(const TestServer& s, const std::string& path) { return s.base() + path; }

std::string sha256_hex(const std::string& data) {
    up::Sha256 h;
    h.update(data.data(), data.size());
    const auto d = h.finish();
    return up::to_hex(d.data(), d.size());
}

up::FetchResult get(const up::FetchConfig& cfg, const std::string& address, uint64_t max, uint64_t exact = 0) {
    return up::fetch(cfg, {address, max, exact, 20}, [](const char*, size_t) { return true; });
}

void test_plain_download_and_hash() {
    TestServer s;
    const std::string body(300000, 'q');
    s.set("/releases/f", Route(body));
    std::string got;
    uint64_t last = 0;
    int calls = 0;
    const auto r = up::fetch(
        loopback(), {url(s, "/f"), body.size(), body.size(), 20},
        [&](const char* d, size_t n) {
            got.append(d, n);
            return true;
        },
        [&](uint64_t so_far) {
            EXPECT(so_far >= last);
            last = so_far;
            calls++;
        });
    EXPECT(r.ok);
    EXPECT(got == body);
    EXPECT(r.bytes == body.size());
    EXPECT(r.sha256 == sha256_hex(body));
    EXPECT(calls > 0 && last == body.size());
}

void test_resume() {
    TestServer s;
    std::string body(5000, 'a');
    for (size_t i = 0; i < body.size(); i++) body[i] = static_cast<char>('a' + i % 23);
    s.set("/releases/f", Route(body));
    std::string got = body.substr(0, 1200);
    up::FetchRequest req{url(s, "/f"), 0, body.size() - 1200, 20};
    req.offset = 1200;
    const auto r = up::fetch(loopback(), req, [&](const char* d, size_t n) {
        got.append(d, n);
        return true;
    });
    EXPECT(r.ok);
    EXPECT(got == body);
    EXPECT(r.bytes == body.size() - 1200);
    EXPECT(s.last_headers()["Range"] == "bytes=1200-");
    // A server that ignores Range and sends the whole body from the start is an error, not a silent restart
    Route whole(body);
    whole.chunked = true;
    s.set("/releases/g", whole);
    req.url = url(s, "/g");
    const auto bad = up::fetch(loopback(), req, [](const char*, size_t) { return true; });
    EXPECT(!bad.ok);
}

void test_no_identifying_headers() {
    TestServer s;
    s.set("/releases/f", Route("x"));
    EXPECT(get(loopback(), url(s, "/f"), 10).ok);
    const auto h = s.last_headers();
    EXPECT(h.at("User-Agent") == "tona-update");
    for (const char* name : {"Cookie", "Authorization", "Referer", "X-Request-Id"}) EXPECT(!h.contains(name));
}

void test_size_limits() {
    TestServer s;
    s.set("/releases/big", Route(std::string(1000, 'a')));
    EXPECT(get(loopback(), url(s, "/big"), 999).error.contains("size"));
    EXPECT(get(loopback(), url(s, "/big"), 1000).ok);
    EXPECT(!get(loopback(), url(s, "/big"), 2000, 999).ok);   // longer than promised
    EXPECT(!get(loopback(), url(s, "/big"), 2000, 1001).ok);  // shorter than promised
    EXPECT(get(loopback(), url(s, "/big"), 2000, 1000).ok);
    Route chunked(std::string(5000, 'b'));
    chunked.chunked = true;  // no Content-Length: only counting can stop it
    s.set("/releases/chunk", chunked);
    EXPECT(get(loopback(), url(s, "/chunk"), 4000).error.contains("more data"));
    EXPECT(get(loopback(), url(s, "/chunk"), 9000, 6000).error.contains("early"));
    EXPECT(get(loopback(), url(s, "/chunk"), 5000, 5000).ok);
}

void test_truncated_and_status() {
    TestServer s;
    Route cut(std::string(100000, 'c'));
    cut.cut_after = 40000;
    s.set("/releases/cut", cut);
    const auto r = get(loopback(), url(s, "/cut"), 200000, 100000);
    EXPECT(!r.ok);
    EXPECT(r.sha256.empty());
    EXPECT(get(loopback(), url(s, "/missing"), 10).error.contains("404"));
    Route denied;
    denied.status = 500;
    s.set("/releases/err", denied);
    EXPECT(get(loopback(), url(s, "/err"), 10).error.contains("500"));
}

void redirect(TestServer& s, const std::string& from, const std::string& to) {
    Route r;
    r.status = 302;
    r.location = to;
    s.set("/releases/" + from, r);
}

void test_redirects() {
    TestServer s;
    s.set("/releases/final", Route("done"));
    redirect(s, "r3", "/releases/final");
    redirect(s, "r2", s.base() + "/r3");
    redirect(s, "r1", "/releases/r2");
    redirect(s, "r0", "/releases/r1");
    EXPECT(get(loopback(), url(s, "/r1"), 10).ok);      // 3 redirects
    const auto r = get(loopback(), url(s, "/r0"), 10);  // 4 redirects
    EXPECT(!r.ok);
    EXPECT(r.error.contains("too many"));
    redirect(s, "loop", "/releases/loop");
    EXPECT(!get(loopback(), url(s, "/loop"), 10).ok);
}

void test_schemes_and_targets() {
    TestServer s;
    s.set("/releases/f", Route("x"));
    up::FetchConfig strict;  // production: plain http is never allowed
    EXPECT(get(strict, url(s, "/f"), 10).error.contains("non-https"));
    redirect(s, "to-localhost", "http://localhost:" + std::to_string(s.port()) + "/releases/f");
    redirect(s, "to-other", "http://192.0.2.1/f");
    redirect(s, "to-ftp", "ftp://127.0.0.1/f");
    redirect(s, "to-userinfo", "http://user:pw@127.0.0.1:" + std::to_string(s.port()) + "/releases/f");
    redirect(s, "to-empty", "");
    for (const char* r : {"to-localhost", "to-other", "to-ftp", "to-userinfo", "to-empty"}) {
        const auto res = get(loopback(), url(s, std::string("/") + r), 10);
        EXPECT(!res.ok);
    }
    EXPECT(!get(loopback(), "file:///etc/passwd", 10).ok);
    EXPECT(!get(loopback(), "https://user@host/x", 10).ok);
    EXPECT(!get(loopback(), "", 10).ok);
}

void test_cancel_and_sink_failure() {
    TestServer s;
    s.set("/releases/f", Route(std::string(1000, 'x')));
    std::atomic<bool> stop{true};
    up::FetchConfig cfg = loopback();
    cfg.cancel = &stop;
    EXPECT(get(cfg, url(s, "/f"), 5000).error.contains("cancelled"));
    const auto r = up::fetch(loopback(), {url(s, "/f"), 5000, 0, 20}, [](const char*, size_t) { return false; });
    EXPECT(!r.ok);
    EXPECT(r.error.contains("store"));
}

void test_timeout() {
    TestServer s;
    Route slow(std::string("late"));
    slow.delay_ms = 3000;
    s.set("/releases/slow", slow);
    up::FetchConfig cfg = loopback();
    cfg.read_seconds = 1;
    const auto begin = std::chrono::steady_clock::now();
    EXPECT(!get(cfg, url(s, "/slow"), 100).ok);
    EXPECT(std::chrono::steady_clock::now() - begin < std::chrono::milliseconds(2800));
    // A body that never finishes within the whole-transfer limit
    const auto r = up::fetch(loopback(), {url(s, "/slow"), 100, 0, 1}, [](const char*, size_t) { return true; });
    EXPECT(!r.ok);
}

void test_text_helper() {
    TestServer s;
    s.set("/releases/t", Route(std::string(2000, 't')));
    std::string body;
    EXPECT(up::fetch_text(loopback(), {url(s, "/t"), 2000, 0, 20}, body).ok);
    EXPECT(body.size() == 2000);
    EXPECT(!up::fetch_text(loopback(), {url(s, "/t"), 1999, 0, 20}, body).ok);
    EXPECT(body.empty());
}

void test_tls(const std::string& dir) {
    const SelfSigned good = make_self_signed(dir);
    TestServer s(good.cert_path, good.key_path);
    s.set("/releases/f", Route("secret-free"));
    up::FetchConfig cfg;  // https needs no flag
    cfg.connect_seconds = 3;
    cfg.read_seconds = 5;
    // Not in the system store: refused
    const auto untrusted = get(cfg, url(s, "/f"), 100);
    EXPECT(!untrusted.ok);
    EXPECT(untrusted.error.contains("request failed"));
    cfg.ca_file = good.cert_path;
    EXPECT(get(cfg, url(s, "/f"), 100).ok);
    // Trusted certificate for another address: the host name check fails
    const SelfSigned wrong = make_self_signed(dir + "/wrong", "127.0.0.2");
    TestServer s2(wrong.cert_path, wrong.key_path);
    s2.set("/releases/f", Route("x"));
    cfg.ca_file = wrong.cert_path;
    EXPECT(!get(cfg, url(s2, "/f"), 100).ok);
}

}  // namespace

int main() {
    return run("update_fetch", [] {
        TempDir dir("update-fetch");
        std::filesystem::create_directories(dir.path() / "wrong");
        test_plain_download_and_hash();
        test_resume();
        test_no_identifying_headers();
        test_size_limits();
        test_truncated_and_status();
        test_redirects();
        test_schemes_and_targets();
        test_cancel_and_sink_failure();
        test_timeout();
        test_text_helper();
        test_tls(dir.path().string());
    });
}
