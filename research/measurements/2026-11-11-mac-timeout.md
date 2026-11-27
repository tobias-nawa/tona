# 2026-11-11: event-wait timeout of a command buffer and GPU state after the abort, M1 Max (#108)

Step M0 of the Mac backend proposal ([../40-mac-backend-proposal.md](../40-mac-backend-proposal.md), D4). The earlier
run ([2026-11-11-mac-event.md](2026-11-11-mac-event.md), E3) saw a 2 s wait complete and an 8 s wait aborted with
`kIOGPUCommandBufferCallbackErrorTimeout`; unchecked were the exact limit and whether the GPU accepts work after the
abort. Apple documents `accessRevoked` after "too many timeouts or hangs" ([../39-mac-engines-2026-11.md](../39-mac-engines-2026-11.md) §4),
so this test provokes as few timeouts as possible. Released by the maintainer (via the coordinator, #108).

## Plan (written before the run)

**Scope.** Tool `bench/mac/timeout.swift`, a throwaway process. A probe command buffer (one tiny dispatch) first;
then command buffers that wait on an `MTLSharedEvent` the CPU signals after 2.5, 3, 3.5, 4, 4.5, 5, 6, 7 s, one
after the other; at the **first abort** the tool stops raising the wait, sends a probe on the same queue and one on a
new queue, records status, error domain and code, whether the probe wrote its value, and exits. At most one provoked
timeout. **One run only**; a repeat needs a new release. Total ≤ 36 s.

Before the run: GPU utilization from `ioreg` (`Device Utilization %`) must be ~0, i.e. no other app on the GPU. After
the run: the kernel log of the last minutes (`log show`, GPU and `IOGPU` messages) for a GPU restart.

Stop rule: `accessRevoked`, a failed probe after the abort, or a GPU restart in the log → stop, no further GPU test,
report to the coordinator.

**Expectation.**

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| limit of the wait | ~5 s (first abort at the 5 s step) | between 2 and 8 s | community reports "~5 s, waiting counts" (39 §4); our E3 |
| time from commit to the abort | 5 s | ±0.5 s of the limit | the watchdog fires at its limit, not at the signal |
| probe on the same queue after the abort | completes, writes its value | - | only that command buffer was reported in E3 (knowledge, unchecked) |
| probe on a new queue | completes | - | same |
| error code | timeout (`kIOGPUCommandBufferCallbackErrorTimeout`), not `accessRevoked` | - | E3; a single timeout |

**Active tooling and conditions.** `swiftc -O`, no validation, no capture. AC power. Before the run (13:07): GPU
utilization 0 % in three samples; load average 3.5 / 3.2 / 4.8.

## Results

Raw data: [2026-11-11-mac-timeout.jsonl](2026-11-11-mac-timeout.jsonl). Tool: `bench/mac/timeout.swift`. One run,
13:07:45 to 13:08:07; GPU utilization 0 % right before; load average 2.5.

| Step | Signal after | Completed after | Status | Error |
|---|---|---|---|---|
| probe before | - | 0.6 ms | completed, value written | - |
| wait | 2.5 / 3 / 3.5 / 4 / 4.5 s | 2.512 / 3.001 / 3.502 / 4.007 / 4.511 s | completed | - |
| **wait** | **5 s** | **5.002 s** | **error** | `MTLCommandBufferErrorDomain` code 2, "Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)"; `accessRevoked`: no |
| probe, same queue, after the abort | - | 1.0 ms | completed, value written | - |
| probe, new queue, after the abort | - | 0.4 ms | completed, value written | - |

- **The limit of an event wait is ~5 s:** 4.5 s completes; the 5 s command buffer ends with a timeout at 5.002 s, at
  the moment of the signal, so the watchdog fired between 4.51 and 5.00 s after the commit. It counts waiting as
  running (the command buffer did no work), as the community reports said. Inside the expectation.
- **One timeout does not break the process's GPU access:** the same queue and a new queue run work right after the
  abort, and no `accessRevoked` came. What happens after several timeouts was not tested on purpose.
- Kernel log: `log show` returned no lines at all for the kernel in that window, not only no GPU lines; unchecked
  whether it can see kernel messages without admin rights, so the log check says nothing. Nobody watched the screen;
  the terminal session ran on without a stall.
- **For D4:** a wait cap of 100 ms is 50× below the limit; a command buffer must stay well below ~4.5 s in total
  including all waits, so the proposal's split at 1 s of encoded waits keeps a margin of 4.5×. Recovery after a
  single timeout can reuse the queue.
