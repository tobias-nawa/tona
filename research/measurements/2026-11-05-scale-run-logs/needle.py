#!/usr/bin/env python3
"""Needle driver of the scale run 2026-11-05 (as /var/tmp/claude/w83/needle1m.py of 2026-11-02). Usage: needle.py LABEL LENGTH CTX [ENV=V ...]"""
import json, os, subprocess, sys, threading, time, urllib.request
ROOT = "<worktree>"; M = "<repo>"
label, length, ctx = sys.argv[1], sys.argv[2], sys.argv[3]
extra = dict(a.split("=", 1) for a in sys.argv[4:])
out = f"/var/tmp/claude/scale1105/needle-{label}"; os.makedirs(out, exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith("TONA_")}
env.update(OMP_NUM_THREADS="6", ROCR_VISIBLE_DEVICES="0", TONA_DIAG="1", TONA_EXPERT_PROFILE=f"{M}/work/profile/experts-coder.bin")
env.update(extra)
cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=54G", "-p", "MemorySwapMax=0", "bash", "-c", 'ulimit -c unlimited; exec "$@"', "_",
       f"{M}/work/build/merged/tona-server", f"{M}/work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
       "--mtp", f"{M}/work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "--port", "8097", "--ctx", ctx]
p = subprocess.Popen(cmd, stdout=open(out + "/server.log", "wb"), stderr=subprocess.STDOUT, env=env)
peaks = {"rss": 0, "gtt": 0, "vram": 0}; stop = [False]
def sampler():
    dev = "/sys/class/drm/card1/device"
    while not stop[0]:
        try:
            for ln in open(f"/proc/{p.pid}/status"):
                if ln.startswith("VmRSS:"): peaks["rss"] = max(peaks["rss"], int(ln.split()[1]) / 1048576)
            peaks["gtt"] = max(peaks["gtt"], int(open(dev + "/mem_info_gtt_used").read()) / 2**30)
            peaks["vram"] = max(peaks["vram"], int(open(dev + "/mem_info_vram_used").read()) / 2**30)
        except OSError: pass
        time.sleep(1)
threading.Thread(target=sampler, daemon=True).start()
t0 = time.time()
try:
    for _ in range(600):
        if p.poll() is not None: raise SystemExit("server exited while loading, see " + out + "/server.log")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8097/status", timeout=2) as r:
                if json.load(r).get("ready"): break
        except Exception: pass
        time.sleep(2)
    r = subprocess.run([sys.executable, "-I", f"{ROOT}/bench/needle/run.py", "--url", "http://127.0.0.1:8097", "--pool", "/var/tmp/claude/w77b/pool.txt",
                        "--length", length, "--texts", os.environ.get("NEEDLE_TEXTS", "0"), "--label", label, "--out", out, "--hard", "--require-cache"], cwd=ROOT)
    print("needle exit", r.returncode, "wall", round(time.time() - t0), "s peaks(GB)", {k: round(v, 1) for k, v in peaks.items()}, flush=True)
finally:
    stop[0] = True; p.terminate(); p.wait()
