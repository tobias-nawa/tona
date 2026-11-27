// The generic drafter check (server/drafter_options.h): it compares the options with what a model declares and knows
// no architecture.
#include <string>

#include "model/arch.h"
#include "server/drafter_options.h"
#include "tests/test_support.h"

using tona::DrafterSupport;
using tona::server::drafter_error;

namespace {

// Two invented models: one that needs either drafter file, one that takes none
const DrafterSupport NEEDS_ONE{true, true, true};
const DrafterSupport NONE{};

void test_a_model_that_needs_a_drafter() {
    EXPECT(drafter_error(NEEDS_ONE, "big", false, true).empty());
    EXPECT(drafter_error(NEEDS_ONE, "big", true, false).empty());
    EXPECT(drafter_error(NEEDS_ONE, "big", false, false).contains("big"));
    EXPECT(!drafter_error(NEEDS_ONE, "big", true, true).empty());
}

void test_a_model_without_drafters() {
    EXPECT(drafter_error(NONE, "small", false, false).empty());
    const std::string e = drafter_error(NONE, "small", false, true);
    EXPECT(e.contains("small") && e.contains("no drafter"));
    EXPECT(!drafter_error(NONE, "small", true, false).empty());
}

void test_a_model_with_one_optional_drafter() {
    const DrafterSupport only_mtp{false, true, false};
    EXPECT(drafter_error(only_mtp, "mid", false, false).empty());
    EXPECT(drafter_error(only_mtp, "mid", false, true).empty());
    EXPECT(drafter_error(only_mtp, "mid", true, false).contains("--mtp"));
}

}  // namespace

int main() {
    return tona::test::run("drafter_options", [] {
        test_a_model_that_needs_a_drafter();
        test_a_model_without_drafters();
        test_a_model_with_one_optional_drafter();
    });
}
