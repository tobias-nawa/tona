"""Safety rules of the VRAM pressure tools (research/62-compositor-survival-test-plan.md).

The display runs on the same GPU as Tona. On 2026-11-15 a second program that took VRAM ended the desktop session, so
everything that takes VRAM next to a live desktop goes through the functions here. They are pure (numbers in,
verdict out) so that the rules are unit-tested without a GPU: tools/tests/test_vram_guard.py.

The hard rule: free VRAM is read from sysfs before EVERY allocation, and an allocation that would leave less than
FLOOR_BYTES free is not made. A reading that fails or makes no sense counts as "not allowed" (fail closed).

Standard library only, Python 3.9+, Linux.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MIB = 1 << 20
GIB = 1 << 30
# 1.5 GiB: above the 1 GiB the compositor needed on 2026-11-15 (research/62 section 3, level 0)
FLOOR_BYTES = 3 * GIB // 2
STEP_BYTES = 64 * MIB
MAX_STEPS = 20  # research/62 section 3, stop rule 3
TIME_CAP_S = 15 * 60  # research/62 section 3, stop rule 4
HEARTBEAT_LIMIT_S = 5.0  # research/62 section 3, stop rule 6
SYSFS = Path("/sys/class/drm")

# Kernel lines that end the run at once: a failed command submission is the 2026-11-15 failure; a reset or a ring
# timeout would be a different, worse one (research/62 stop rule 5).
KERNEL_ABORT = re.compile(
    r"GPU reset|ring .* timeout|Not enough memory for command submission|amdgpu_vm_validate\(\) failed",
    re.IGNORECASE,
)


# "VM memory stats for proc NAME(PID) task NAME:cs0(PID)": a program name is never written down (tools/vram_watch.py)
NAME_PATTERN = re.compile(r"\b(proc|process|task|comm)\b[ =:]+\S+", re.IGNORECASE)


def scrub(line: str) -> str:
    """The kernel line without the names of programs."""
    return NAME_PATTERN.sub(lambda m: m.group(1) + " <name>", line.strip())


def kernel_abort(line: str) -> bool:
    """True for a kernel log line that stops the run."""
    return KERNEL_ABORT.search(line) is not None


def free_bytes(total: int | None, used: int | None) -> int | None:
    """Free VRAM, or None when a counter is missing or impossible."""
    if total is None or used is None or total <= 0 or used < 0 or used > total:
        return None
    return total - used


def may_allocate(free: int | None, step: int = STEP_BYTES, floor: int = FLOOR_BYTES) -> bool:
    """The hard rule: the allocation is allowed only if `free - step >= floor`; no reading, no allocation."""
    if free is None or step <= 0:
        return False
    return free - step >= floor


def steps_available(free: int | None, step: int = STEP_BYTES, floor: int = FLOOR_BYTES, cap: int = MAX_STEPS) -> int:
    """How many steps fit above the floor right now (an upper bound for the plan; every step is re-checked)."""
    if free is None or step <= 0 or free - floor < step:
        return 0
    return min(cap, (free - floor) // step)


def pick_card(root: Path = SYSFS) -> Path | None:
    """The device directory of the discrete GPU: the card with the most VRAM, over 4 GiB (the iGPU has 2)."""
    best, size = None, 4 * GIB
    for dev in sorted(root.glob("card[0-9]*/device")):
        try:
            total = int((dev / "mem_info_vram_total").read_text())
        except (OSError, ValueError):
            continue
        if total > size:
            best, size = dev, total
    return best


def read_counter(dev: Path, name: str) -> int | None:
    try:
        return int((dev / name).read_text())
    except (OSError, ValueError):
        return None


def read_free(dev: Path) -> int | None:
    """Free VRAM of the card, read now: `mem_info_vram_total` minus `mem_info_vram_used`."""
    return free_bytes(read_counter(dev, "mem_info_vram_total"), read_counter(dev, "mem_info_vram_used"))


def bus_id_of(dev: Path) -> str:
    """PCI address of the card ('0000:03:00.0'), the name of its device directory target."""
    return dev.resolve().name


def same_card(hip_bus_id: str, dev: Path) -> bool:
    """True only when the HIP device the balloon allocates on is the card whose free memory was read."""
    return hip_bus_id.strip().lower() == bus_id_of(dev).lower()


@dataclass(frozen=True)
class StopState:
    """What the stop rules look at, one reading."""

    elapsed_s: float
    stop_file: bool
    abort_line: str | None  # first kernel line that matched KERNEL_ABORT, else None
    canary_failed: bool
    heartbeat_age_s: float | None  # None: no heartbeat is expected (the process writes it itself)
    free: int | None


def stop_reason(
    state: StopState, time_cap_s: float = TIME_CAP_S, floor: int = FLOOR_BYTES, hb_limit_s: float = HEARTBEAT_LIMIT_S
) -> str | None:
    """The first rule that fires, or None. Order: what is most dangerous first."""
    if state.abort_line is not None:
        return "kernel log: " + state.abort_line
    if state.stop_file:
        return "STOP file"
    if state.canary_failed:
        return "canary failed"
    if state.free is None:
        return "free VRAM unreadable"
    if state.free < floor:
        return f"free VRAM {state.free // MIB} MiB is below the floor {floor // MIB} MiB"
    if state.heartbeat_age_s is not None and state.heartbeat_age_s > hb_limit_s:
        return "heartbeat stopped"
    if state.elapsed_s > time_cap_s:
        return "time cap"
    return None


def watchdog_kills(heartbeat_age_s: float | None, parent_alive: bool, limit_s: float = HEARTBEAT_LIMIT_S) -> bool:
    """The watchdog kills the balloon and the canary when the heartbeat is older than the limit (or missing) or the
    process that started the job is gone."""
    if not parent_alive:
        return True
    return heartbeat_age_s is None or heartbeat_age_s > limit_s
