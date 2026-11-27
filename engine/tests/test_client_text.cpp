// Texts and number formats of the client: no empty text, no duplicate keys, number format
#include <set>

#include "client/text.h"
#include "tests/test_support.h"

using namespace tona::client;
using namespace tona::test;

namespace {

void test_every_key_has_a_text() {
    std::set<std::string> seen;
    for (size_t i = 0; i < text_count(); i++) {
        EXPECT(text_complete(i));
        EXPECT(seen.insert(text_key(i)).second);  // a key twice would silently hide the second text
    }
    EXPECT(text_count() > 200);
}

void test_lookup_and_fill() {
    EXPECT(tr("tab_models") == "Models");
    EXPECT(tr("no_such_key_x") == "no_such_key_x");  // a missing key shows itself, so it is found
    EXPECT(fill("{} of {} GB", "14", "16") == "14 of 16 GB");
    EXPECT(fill("{} left", "1", "2") == "1 left");  // surplus arguments are ignored
}

void test_numbers() {
    EXPECT(fnum(2900) == "2,900" && fnum(-1234567) == "-1,234,567" && fnum(0) == "0");
    EXPECT(fdec(14.1) == "14.1" && fdec(14.0, 0) == "14");
}

}  // namespace

int main() {
    return run("client_text", [] {
        test_every_key_has_a_text();
        test_lookup_and_fill();
        test_numbers();
    });
}
