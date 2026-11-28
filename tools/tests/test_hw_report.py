import json

import hw_report as hw

REPORT = {
    "cpu": {"model": "Test CPU 9000", "cores": 6, "threads": 12, "avx512": True},
    "gpus": [
        {"vendor": "AMD", "name": "Test GPU", "arch": "gfx1200", "vram_gib": 16.0, "vram_used_gib": 1.5},
        {"vendor": "AMD", "name": "unknown", "pci_id": "1002:164e", "vram_gib": 2.0, "integrated": True},
    ],
    "ram": {"total_gib": 64.0, "available_gib": 40.5},
    "os": {"name": "Test Linux 1", "kernel": "7.0.0", "machine": "x86_64"},
    "toolkit": {"rocm": "10.1.0"},
    "limits": {"env": {"TONA_RAM_GB": "58"}, "cgroup": {"MemoryMax": "58.0G"}},
    "foreign_load": {"loadavg": [0.5, 0.4, 0.3], "cpu_busy_percent": 12.0, "swap_used_gib": 0.5},
    "build": {"commit": "abc1234", "dirty": False},
}


def test_scrub_replaces_home_and_user():
    text = "model /home/alice/work/m.gguf and /srv/alice/x, alice"
    out = hw.scrub(text, home="/home/alice", user="alice")
    assert out == "model ~/work/m.gguf and /srv/<user>/x, alice"


def test_scrub_ignores_root_home_and_short_user():
    assert hw.scrub("/root/a /tmp/ab/c", home="/", user="ab") == "/root/a /tmp/ab/c"


def test_markdown_has_all_sections():
    md = hw.render_markdown(REPORT, home="/nowhere", user="")
    assert md.startswith("### Test conditions")
    assert "- CPU: Test CPU 9000, 6 cores / 12 threads" in md
    assert "- GPU: Test GPU, gfx1200, 16.0 GiB VRAM, 1.5 GiB held by others before the run" in md
    assert "AMD 1002:164e, 2.0 GiB VRAM, integrated" in md
    assert "- RAM: 64.0 GiB, 40.5 GiB available before the run" in md
    assert "`TONA_RAM_GB=58`" in md
    assert "MemoryMax=58.0G" in md
    assert "CPU use by other programs over 0.5 s: 12.0 % of all threads, swap in use 0.5 GiB" in md
    assert "`abc1234` (clean tree)" in md


def test_markdown_unknowns_and_model_line():
    rep = json.loads(json.dumps(REPORT))
    rep["cpu"]["cores"] = None
    rep["gpus"] = []
    rep["limits"] = {"env": {}, "cgroup": {}}
    rep["build"] = {"commit": "unknown", "dirty": None}
    rep["model"] = {"file": "m.gguf", "size_gib": 27.6, "sha256_short": "e11083ba855e", "matches_pinned": False}
    md = hw.render_markdown(rep, home="/nowhere", user="")
    assert "unknown cores / 12 threads" in md
    assert "- GPU: none detected" in md
    assert "none set" in md
    assert "(unknown tree)" in md
    assert "sha256 `e11083ba855e` (DIFFERS from toolchain/models.toml)" in md


def test_markdown_never_contains_home(monkeypatch):
    rep = json.loads(json.dumps(REPORT))
    rep["limits"]["env"] = {"TONA_GPU_LOCK": "/home/bob/p/work/gpu.lock"}
    assert "/home/bob" not in hw.render_markdown(rep, home="/home/bob", user="bob")
    assert "~/p/work/gpu.lock" in hw.render_markdown(rep, home="/home/bob", user="bob")


def test_json_roundtrip_and_scrub():
    rep = json.loads(json.dumps(REPORT))
    rep["limits"]["env"] = {"TONA_X": "/home/bob/x"}
    out = json.loads(hw.render_json(rep, home="/home/bob", user="bob"))
    assert out["limits"]["env"]["TONA_X"] == "~/x"
    assert out["cpu"]["model"] == "Test CPU 9000"


def test_tona_env_only_tona(monkeypatch):
    monkeypatch.setenv("TONA_RAM_GB", "8")
    monkeypatch.setenv("OTHER_VAR", "x")
    assert hw.tona_env().get("TONA_RAM_GB") == "8"
    assert "OTHER_VAR" not in hw.tona_env()


def test_model_info_hash_and_missing(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"abc")
    info = hw.model_info(str(f), True)
    assert info["sha256_short"] == "ba7816bf8f01"
    assert hw.model_info(str(f), False)["sha256_short"] is None  # no hash unless asked
    assert "error" in hw.model_info(str(tmp_path / "none"), True)


def test_pinned_sha_lookup():
    name = "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
    assert hw.pinned_entry(name)[1].startswith("bfa3644d")
    assert hw.pinned_entry("nothing.gguf") == (0, "")


def test_collect_runs_here():
    rep = hw.collect()
    assert rep["cpu"]["model"]
    assert set(rep) >= {"cpu", "gpus", "ram", "os", "limits", "foreign_load", "build"}


def test_report_contains_no_process_names(monkeypatch):
    """Which programs run is private: neither the Markdown nor the JSON may carry a process name."""
    monkeypatch.setattr(hw, "cpu_ticks", lambda: {1: ("secretprogram", 10), 2: ("otherapp", 5)})
    monkeypatch.setattr(hw, "run", lambda cmd, timeout=10: "")
    rep = hw.collect()
    for text in (hw.render_markdown(rep), hw.render_json(rep)):
        assert "secretprogram" not in text and "otherapp" not in text
    assert "top_processes" not in rep["foreign_load"]


def test_model_recognized_by_name_and_size(tmp_path):
    name = "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
    size, _ = hw.pinned_entry(name)
    f = tmp_path / name
    with f.open("wb") as out:
        out.truncate(size)  # sparse file, never hashed
    info = hw.model_info(str(f), False)
    assert info["recognized"] and info["sha256_short"] is None
    rep = {**REPORT, "model": info}
    assert "recognized by name and size" in hw.render_markdown(rep, home="/nowhere", user="")


def test_unknown_model_asks_for_hash(tmp_path):
    f = tmp_path / "mine.gguf"
    f.write_bytes(b"x")
    rep = {**REPORT, "model": hw.model_info(str(f), False)}
    assert "unknown model, run with --hash" in hw.render_markdown(rep, home="/nowhere", user="")


def test_sandbox_reasons():
    assert hw.sandbox_info(env={}, exists=lambda p: False) == {"sandbox": False, "reasons": []}
    assert hw.sandbox_info(env={"CI": "true"}, exists=lambda p: False)["reasons"] == ["CI set"]
    assert hw.sandbox_info(env={}, exists=lambda p: p == "/.dockerenv")["reasons"] == ["/.dockerenv exists"]


def test_no_gpu_is_not_a_sandbox_reason():
    rep = {**REPORT, "gpus": [], "sandbox": hw.sandbox_info(env={}, exists=lambda p: False)}
    md = hw.render_markdown(rep, home="/nowhere", user="")
    assert "- GPU: none detected" in md and "- sandbox: no" in md


def test_sandbox_line_in_markdown_and_json():
    rep = {**REPORT, "sandbox": {"sandbox": True, "reasons": ["CI set", "/.dockerenv exists"]}}
    assert "- sandbox: yes (CI set, /.dockerenv exists)" in hw.render_markdown(rep, home="/nowhere", user="")
    assert json.loads(hw.render_json(rep))["sandbox"]["sandbox"] is True
    rep["sandbox"] = {"sandbox": False, "reasons": []}
    assert "- sandbox: no" in hw.render_markdown(rep, home="/nowhere", user="")


def test_run_uses_c_locale(monkeypatch):
    # A German locale prints "0,7" in ps and sysctl; float() then fails (seen on macOS)
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
    out = hw.run(["sh", "-c", 'printf "%s %s" "$LC_ALL" "$LANG"'])
    assert out == "C C"


def test_scrub_windows_paths_and_json():
    text = r"model C:\Users\alice\work\m.gguf and D:\data\alice\x, C:/Users/alice/y"
    out = hw.scrub(text, home=r"C:\Users\alice", user="alice")
    assert out == r"model ~\work\m.gguf and D:\data\<user>\x, ~/y"
    rep = json.loads(json.dumps(REPORT))
    rep["limits"]["env"] = {"TONA_MODEL": r"C:\Users\alice\models\m.gguf"}
    rendered = hw.render_json(rep, home=r"C:\Users\alice", user="alice")
    assert "alice" not in rendered
    assert json.loads(rendered)["limits"]["env"]["TONA_MODEL"] == r"~\models\m.gguf"


def test_win32_parsers():
    import hw_win32

    smi = "NVIDIA GeForce RTX 5090, 32607, 1024, 3, 572.60\nNVIDIA L4, 23034, 512, [N/A], 572.60\n"
    gpus = hw_win32.parse_nvidia_smi(smi)
    assert len(gpus) == 2
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 5090" and gpus[0]["vram_gib"] == 31.8
    assert gpus[0]["vram_used_gib"] == 1.0 and gpus[0]["busy_percent"] == 3 and gpus[0]["driver"] == "572.60"
    assert gpus[1]["busy_percent"] is None

    single = json.dumps(
        {
            "Name": "Microsoft Basic Display Adapter",
            "AdapterRAM": 0,
            "DriverVersion": "10.0.22621.1",
            "PNPDeviceID": "PCI\\VEN_1234&DEV_1111",
        }
    )
    fallback = hw_win32.parse_video_controllers(single)
    assert len(fallback) == 1 and fallback[0]["name"] == "Microsoft Basic Display Adapter"
    assert fallback[0]["vram_gib"] is None and "driver" not in fallback[0]

    nvcc_out = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.8, V12.8.61\n"
    assert hw_win32.parse_nvcc_version(nvcc_out) == "12.8"
    assert hw_win32.parse_nvcc_version("") == ""
    assert hw_win32.parse_nvidia_smi("bad,line\n") == []
    assert hw_win32.parse_video_controllers("{not-json") == []
    nv_ctrl = json.dumps({"Name": "NVIDIA RTX 5090", "AdapterRAM": 4 * hw_win32.GIB, "DriverVersion": "572.60"})
    parsed_nv = hw_win32.parse_video_controllers(nv_ctrl)
    assert parsed_nv[0]["vendor"] == "NVIDIA" and parsed_nv[0]["driver"] == "572.60"
    assert hw_win32._vendor_from_pnp("", "AMD Radeon RX 9060 XT") == "AMD"
    assert hw_win32._vendor_from_pnp("", "Unknown GPU") == hw_win32.UNKNOWN


class _FakeKernel32:
    def __init__(self, hw_win32):
        self.hw = hw_win32
        self.times = iter([(100, 200, 200), (250, 400, 200)])
        self.power_vals = (1, 1)
        self.fail = False

    def IsProcessorFeaturePresent(self, flag):
        if self.fail:
            raise OSError("boom")
        return flag == self.hw.PF_AVX512F

    def GlobalMemoryStatusEx(self, ptr):
        import ctypes

        if self.fail:
            return 0
        st = ctypes.cast(ptr, ctypes.POINTER(self.hw._MemoryStatusEx)).contents
        st.ullTotalPhys, st.ullAvailPhys = 64 * self.hw.GIB, 48 * self.hw.GIB
        return 1

    def GetSystemTimes(self, idle_p, kern_p, user_p):
        import ctypes

        if self.fail:
            return 0
        for p, v in zip((idle_p, kern_p, user_p), next(self.times), strict=True):
            ctypes.cast(p, ctypes.POINTER(self.hw._FileTime)).contents.dwLowDateTime = v
        return 1

    def GetSystemPowerStatus(self, ptr):
        import ctypes

        if self.fail:
            return 0
        st = ctypes.cast(ptr, ctypes.POINTER(self.hw._SystemPowerStatus)).contents
        st.ACLineStatus, st.BatteryFlag = self.power_vals
        return 1


def test_win32_kernel32_probes(monkeypatch):
    import ctypes
    import types

    import hw_win32

    k32 = _FakeKernel32(hw_win32)
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=k32), raising=False)
    monkeypatch.setattr(hw_win32.time, "sleep", lambda _: None)

    assert hw_win32.memory() == {"total_gib": 64.0, "available_gib": 48.0}
    assert hw_win32.cpu_busy_percent(0.01) == 25.0
    assert hw_win32.power() == {"on_battery": False}
    k32.power_vals = (0, 1)
    assert hw_win32.power() == {"on_battery": True}
    k32.power_vals = (1, 128)
    assert hw_win32.power() == {"on_battery": None}
    assert "Windows" in hw_win32.os_info()["name"]

    k32.fail = True
    assert hw_win32._has_avx512() is False
    assert hw_win32.memory() == {"total_gib": None, "available_gib": None}
    assert hw_win32.cpu_busy_percent(0.01) is None
    assert hw_win32.power() == {"on_battery": None}


def test_win32_cpu_and_gpu_probes(monkeypatch):
    import ctypes
    import sys
    import types

    import hw_win32

    k32 = _FakeKernel32(hw_win32)
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=k32), raising=False)

    def fake_run(cmd, timeout=10):
        if cmd[0] == "nvidia-smi":
            return "NVIDIA GeForce RTX 5090, 32607, 1024, 2, 572.60\n"
        if cmd[0] == "nvcc":
            return "release 12.8"
        if "NumberOfCores" in cmd[-1]:
            return "16\n"
        return "AMD Ryzen 9 9950X\n"

    assert hw_win32.nvcc_toolkit(fake_run) == "12.8"
    assert hw_win32.gpus(fake_run)[0]["name"] == "NVIDIA GeForce RTX 5090"
    fallback = hw_win32.gpus(lambda cmd, timeout=10: "" if cmd[0] == "nvidia-smi" else '[{"Name":"Intel GPU"}]')
    assert fallback[0]["vendor"] == "Intel"
    cpu_info = hw_win32.cpu(fake_run)
    assert cpu_info["model"] == "AMD Ryzen 9 9950X" and cpu_info["cores"] == 16 and cpu_info["avx512"] is True

    class FakeKey:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_winreg = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=1,
        OpenKey=lambda h, p: FakeKey(),
        QueryValueEx=lambda k, n: ("Registry CPU", 1),
    )
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    assert hw_win32._cpu_brand_registry() == "Registry CPU"
