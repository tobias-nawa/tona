// Per-request options derived from the parsed request and the environment
#pragma once

#include <string>

#include "server/api.h"
#include "server/job.h"

namespace tona::server {

// TONA_ROLE_EFFORT="subagent=low,auxiliary=low" replaces the request's effort level for these roles (main, subagent,
// auxiliary); without an entry the request's level applies
std::string role_effort(const std::string& role, const std::string& effort);

// Copies max tokens, sampling, thinking, effort, thinking budget and stop sequences of the request into the job
void configure_job(Job& job, const Request& q);

}  // namespace tona::server
