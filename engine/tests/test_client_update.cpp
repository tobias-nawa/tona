// The update keys of tona (c, u and r with their confirmations). The terminal itself is not involved; the function
// takes the state and returns the message. What the pages show about an update is tested in test_client_views.cpp.
#include <string>

#include "client/term.h"
#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "update/updater.h"

using namespace tona::test;
using tona::update::UpdateView;
namespace top = tona::top;
namespace {

bool has(const std::string& text, const std::string& part) { return text.contains(part); }

struct Rig {
    TempDir dir;
    TestServer server;
    std::unique_ptr<tona::update::Updater> updater;
    top::KeyConfirm confirm;

    Rig(const std::string& name, bool release) : dir(name) {
        up::SwitchConfig sw;
        sw.restart = [](std::string&) { return true; };
        updater = std::make_unique<tona::update::Updater>(make_env(server, dir.path()), sw, release);
    }
    std::string key(char c) { return top::update_key(updater.get(), c, confirm); }
};

void test_keys_in_a_source_checkout() {
    top::KeyConfirm confirm;
    EXPECT(has(top::update_key(nullptr, 'u', confirm), "git pull"));
    Rig rig("top-source", false);
    EXPECT(has(rig.key('c'), "git pull"));
    EXPECT(!rig.updater->view().checks_on);
}

void test_confirmation_keys() {
    Rig rig("top-keys", true);
    EXPECT(has(rig.key('u'), "Update checks are off"));
    EXPECT(has(rig.key('c'), "Press c again to turn update checks on"));
    EXPECT(!rig.updater->view().checks_on);
    // Another key in between disarms the first
    EXPECT(has(rig.key('r'), "Press r again"));
    EXPECT(has(rig.key('c'), "Press c again"));
    EXPECT(rig.key('c').empty());
    EXPECT(rig.updater->view().checks_on);
    EXPECT(has(rig.key('u'), "No update known"));
    EXPECT(has(rig.key('c'), "turn update checks off"));
    EXPECT(rig.key('c').empty());
    EXPECT(!rig.updater->view().checks_on);
    EXPECT(has(rig.key('r'), "Press r again"));
}

void test_rollback_key_acts_after_confirmation() {
    Rig rig("top-rollback", true);
    EXPECT(has(rig.key('r'), "Press r again"));
    EXPECT(rig.key('r').empty());  // second press starts the roll back; there is no previous version to go to
    for (int i = 0; i < 300 && rig.updater->view().phase != UpdateView::Phase::Idle; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    for (int i = 0; i < 300 && !rig.updater->view().failed; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT(rig.updater->view().failed);
}

}  // namespace

int main() {
    return run("client_update", [] {
        test_keys_in_a_source_checkout();
        test_confirmation_keys();
        test_rollback_key_acts_after_confirmation();
    });
}
