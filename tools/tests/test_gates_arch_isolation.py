import sys

import arch_isolation
import pytest
from conftest import write

KERNEL = "engine/gpu/mega_op_qsa.h"


def run(monkeypatch, *files):
    monkeypatch.setattr(sys, "argv", ["arch_isolation.py", *files])
    return arch_isolation.main()


def violation(repo, monkeypatch, capsys, path, text):
    write(repo, path, text)
    assert run(monkeypatch, path) == 1
    return capsys.readouterr().err


def test_kernel_header_naming_arch_id_is_reported_with_line(repo, monkeypatch, capsys):
    err = violation(repo, monkeypatch, capsys, KERNEL, "#pragma once\nconstexpr auto a = ArchId::Qwen4Exp;\n")
    assert f"{KERNEL}:2: kernel header names ArchId" in err


@pytest.mark.parametrize("header", ["model/arch.h", "model/qwen35moe.h", "gpu/arch/qwen35moe/kernel_set.h"])
def test_kernel_header_including_an_architecture_header_is_reported(repo, monkeypatch, capsys, header):
    err = violation(repo, monkeypatch, capsys, "engine/gpu/matvec_phase.h", f'#include "{header}"\n')
    assert f"kernel header includes {header}" in err


def test_kernel_header_with_shared_includes_passes(repo, monkeypatch):
    write(repo, KERNEL, '#include "gpu/mega_op_common.h"\n#include "model/quant_tables.h"\n')
    assert run(monkeypatch, KERNEL) == 0


def test_comment_mentioning_the_rule_is_ignored(repo, monkeypatch):
    write(repo, KERNEL, "// never ArchId here\n/* switch (arch) { case ArchId::A: } */\nint x;\n")
    assert run(monkeypatch, KERNEL) == 0


def test_arch_directory_including_another_architecture_is_reported(repo, monkeypatch, capsys):
    err = violation(
        repo, monkeypatch, capsys, "engine/gpu/arch/qwen4exp/kernel.hip", '#include "gpu/arch/nemotron_h/mamba.h"\n'
    )
    assert "of architecture nemotron_h (own: qwen4exp)" in err


def test_arch_directory_including_its_own_headers_passes(repo, monkeypatch):
    write(repo, "engine/gpu/arch/qwen4exp/kernel.hip", '#include "gpu/arch/qwen4exp/kernel_set.h"\n')
    assert run(monkeypatch, "engine/gpu/arch/qwen4exp/kernel.hip") == 0


@pytest.mark.parametrize("head", ["switch (model.arch())", "switch (arch)", "switch (ArchId(x))"])
def test_switch_over_arch_outside_the_mapping_files_is_reported(repo, monkeypatch, capsys, head):
    err = violation(repo, monkeypatch, capsys, "engine/server/engine.cpp", f"void f() {{\n    {head} {{}}\n}}\n")
    assert "engine/server/engine.cpp:2: switch over the architecture" in err


def test_case_arch_id_is_reported(repo, monkeypatch, capsys):
    err = violation(repo, monkeypatch, capsys, "engine/server/engine.cpp", "case ArchId::Qwen35Moe: break;\n")
    assert "case ArchId::" in err


@pytest.mark.parametrize(
    "path",
    [
        "engine/model/model_registry.cpp",
        "engine/model/arch.cpp",
        "engine/backend/registry.cpp",
        "engine/gpu/arch/qwen35moe/program_factory.cpp",
        "engine/gpu/mega_build.cpp",
        "engine/gpu/prefill_run.cpp",
        "engine/cpu/ref_factory.cpp",
    ],
)
def test_mapping_files_may_switch_over_the_architecture(repo, monkeypatch, path):
    write(repo, path, "switch (id) { case ArchId::Qwen4Exp: break; }\nswitch (m.arch()) {}\n")
    assert run(monkeypatch, path) == 0


def test_switch_over_something_else_passes(repo, monkeypatch):
    write(repo, "engine/server/engine.cpp", "switch (kind) { case Kind::A: break; }\n")
    assert run(monkeypatch, "engine/server/engine.cpp") == 0


def test_files_outside_engine_are_ignored(repo, monkeypatch):
    write(repo, "tools/x.cpp", "switch (arch) {}\n")
    assert run(monkeypatch, "tools/x.cpp") == 0


def test_without_arguments_checks_tracked_files(repo, monkeypatch):
    write(repo, "engine/server/engine.cpp", "switch (arch) {}\n")
    assert run(monkeypatch) == 1


def test_without_arguments_and_no_sources_is_an_error(repo, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        run(monkeypatch)
    assert stop.value.code == 2
