// Keeps the GPU clock up between requests (TONA_CLOCK_HOLD): after 0.5 s without work the firmware drops the clock to
// 1.8 GHz and the next prefill pays about 190 ms before the clock is back at 3.1 GHz. A one-block kernel that keeps the
// engine busy for about 1 s before the request removes 91 % of that, at +27 W while it runs
// (research/60-gpu-clock-ramp.md, research/measurements/2026-11-16-clock-ramp.md). Nothing here runs while a request
// does: the server (server/hold_controller.h) switches the hold off before it admits one.
#pragma once

namespace tona::gpu {

// Once per process, after the other streams exist. stream: an idle stream to run the spin on (a spare queue of the
// prefill, so the number of HW queues stays as it was, 2026-11-12-prefill-bimodal.md result 3), or nullptr for a stream
// of its own. Does nothing on a second call; a HIP error switches the hold off for good.
void clock_hold_init(void* stream);

// on: starts the spin (returns at once). off: stops it and returns when no kernel of the hold runs any more (a
// millisecond or two: the kernel polls a flag), without a blocking HIP call. No-op before clock_hold_init.
void clock_hold_set(bool on);

// Pure rule of one chunk: the spin kernel's length in wall-clock ticks, 250 ms: far below the 2 s of the driver, so a
// stuck kernel can never be the one that makes the desktop wait.
long long clock_hold_chunk_ticks(int wall_khz);

}  // namespace tona::gpu
