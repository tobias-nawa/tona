// GPU and RAM numbers for /status
#pragma once

#include "server/chat.h"

namespace tona::server {

// The card with the most VRAM (the iGPU has 512 MiB); values from sysfs, because hipMemGetInfo does not see the
// desktop
json system_status();

}  // namespace tona::server
