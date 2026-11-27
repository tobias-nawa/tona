#include "server/chat_tokens.h"
#include "server/server_context.h"

namespace tona::server {

std::vector<int32_t> prompt_of(const ServerContext& ctx, const Request& q) {
    return encode_chat(*ctx.tok, request_pieces(*ctx.chat, q));
}

json warm_json(const Job& j) {
    return {{"done", j.warm_done},
            {"total", j.prompt.size()},
            {"checkpoint", j.info.ckpt},
            {"interrupted", j.info.aborted}};
}

void send_error(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

}  // namespace tona::server
