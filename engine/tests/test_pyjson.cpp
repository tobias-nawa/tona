// to_pyjson must give what Jinja's tojson (Python json.dumps) gives: the chat template embeds tool schemas with it.
#include "server/chat.h"
#include "tests/test_support.h"

using tona::server::json;
using tona::server::to_pyjson;

void tests() {
    EXPECT(to_pyjson(json::parse(R"({"a": 1, "b": [1, 2, {"c": null}], "d": true})")) ==
           R"({"a": 1, "b": [1, 2, {"c": null}], "d": true})");  // ", " and ": " separators
    EXPECT(to_pyjson(json::parse("[]")) == "[]" && to_pyjson(json::parse("{}")) == "{}");
    EXPECT(to_pyjson(json("q\"b\\n\n\r\t\b\f\x01")) == R"("q\"b\\n\n\r\t\b\f\u0001")");
    EXPECT(to_pyjson(json("\xC3\xA9\xE2\x9C\x93")) == "\"\xC3\xA9\xE2\x9C\x93\"");  // non-ASCII stays as it is
    EXPECT(to_pyjson(json(1.0)) == "1.0" && to_pyjson(json(2.5)) == "2.5" && to_pyjson(json(1e25)) == "1e+25");
    EXPECT(to_pyjson(json(-7)) == "-7" && to_pyjson(json(18446744073709551615ULL)) == "18446744073709551615");
    EXPECT(to_pyjson(json(false)) == "false" && to_pyjson(json(nullptr)) == "null");
    // Key order is the order of insertion (ordered_json), as in a Python dict
    json o = json::object();
    o["z"] = 1, o["a"] = 2;
    EXPECT(to_pyjson(o) == R"({"z": 1, "a": 2})");
}

int main() { return tona::test::run("test_pyjson", tests); }
