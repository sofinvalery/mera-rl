from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_sft_local import (
    _build_launch_command,
    _ensure_min_free_space,
    _resolve_gpu_peak_tflops,
    _resolve_wandb_offline,
    _infer_torchrun_bin,
)


def _write_executable(path: Path) -> None:
    path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def test_infer_torchrun_bin_prefers_explicit_override() -> None:
    resolved = _infer_torchrun_bin("/custom/torchrun", "/tmp/sft")
    assert resolved == "/custom/torchrun"


def test_infer_torchrun_bin_from_sft_sibling(tmp_path: Path) -> None:
    bin_dir = tmp_path / "prime" / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    sft_path = bin_dir / "sft"
    torchrun_path = bin_dir / "torchrun"
    _write_executable(sft_path)
    _write_executable(torchrun_path)

    resolved = _infer_torchrun_bin(None, str(sft_path))
    assert resolved == str(torchrun_path)


def test_infer_torchrun_bin_falls_back_to_path(tmp_path: Path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    torchrun_path = bin_dir / "torchrun"
    _write_executable(torchrun_path)
    monkeypatch.setenv("PATH", str(bin_dir))

    resolved = _infer_torchrun_bin(None, str(tmp_path / "missing" / "sft"))
    assert Path(resolved).resolve() == torchrun_path.resolve()
    assert os.access(resolved, os.X_OK)


def test_build_launch_command_keeps_single_sft_entry_for_multi_gpu() -> None:
    cmd = _build_launch_command(
        torchrun_bin="/tmp/torchrun",
        resolved_sft_entry="/tmp/sft",
        config_path=Path("/tmp/config.toml"),
        nproc_per_node=2,
        master_port=23456,
        override_args=["--foo", "bar"],
        extra_args=["--baz"],
    )

    assert cmd[:3] == ["/tmp/sft", "@", "/tmp/config.toml"]
    assert "torchrun" not in cmd[0]


def test_ensure_min_free_space_allows_when_above_threshold(tmp_path: Path, monkeypatch) -> None:
    disk_usage_type = type(shutil.disk_usage("/"))
    monkeypatch.setattr(
        "scripts.run_sft_local.shutil.disk_usage",
        lambda _: disk_usage_type(total=100, used=40, free=60),
    )
    _ensure_min_free_space(tmp_path, min_free_gb=0.0, stage="test")


def test_ensure_min_free_space_raises_when_below_threshold(tmp_path: Path, monkeypatch) -> None:
    gib = 1024**3
    disk_usage_type = type(shutil.disk_usage("/"))
    monkeypatch.setattr(
        "scripts.run_sft_local.shutil.disk_usage",
        lambda _: disk_usage_type(total=100 * gib, used=95 * gib, free=5 * gib),
    )
    try:
        _ensure_min_free_space(tmp_path, min_free_gb=10.0, stage="test")
    except RuntimeError as exc:
        assert "Insufficient free disk" in str(exc)
        return
    raise AssertionError("Expected _ensure_min_free_space to raise RuntimeError")


def test_resolve_gpu_peak_tflops_prefers_explicit(monkeypatch) -> None:
    monkeypatch.setenv("MERA_GPU_PEAK_TFLOPS", "123.0")
    assert _resolve_gpu_peak_tflops(234.0) == 234.0


def test_resolve_gpu_peak_tflops_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("MERA_GPU_PEAK_TFLOPS", "456.5")
    assert _resolve_gpu_peak_tflops(None) == 456.5


def test_resolve_gpu_peak_tflops_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("MERA_GPU_PEAK_TFLOPS", raising=False)
    assert _resolve_gpu_peak_tflops(None) is None


def test_resolve_wandb_offline_forced_offline(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    offline, reason = _resolve_wandb_offline(
        wandb_mode="offline",
        allow_offline_fallback=True,
        probe_timeout=1.0,
        dry_run=False,
    )
    assert offline is True
    assert reason == "forced_offline"


def test_resolve_wandb_offline_forced_online_probe_ok(monkeypatch) -> None:
    monkeypatch.setattr("scripts.run_sft_local._probe_wandb_online", lambda timeout_seconds: (True, "ok"))
    offline, reason = _resolve_wandb_offline(
        wandb_mode="online",
        allow_offline_fallback=True,
        probe_timeout=1.0,
        dry_run=False,
    )
    assert offline is False
    assert reason == "forced_online_probe_ok"


def test_resolve_wandb_offline_forced_online_probe_fail_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.run_sft_local._probe_wandb_online",
        lambda timeout_seconds: (False, "AuthenticationError: blocked"),
    )
    offline, reason = _resolve_wandb_offline(
        wandb_mode="online",
        allow_offline_fallback=True,
        probe_timeout=1.0,
        dry_run=False,
    )
    assert offline is True
    assert "forced_online_fallback_offline" in reason


def test_resolve_wandb_offline_forced_online_probe_fail_no_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.run_sft_local._probe_wandb_online",
        lambda timeout_seconds: (False, "AuthenticationError: blocked"),
    )
    try:
        _resolve_wandb_offline(
            wandb_mode="online",
            allow_offline_fallback=False,
            probe_timeout=1.0,
            dry_run=False,
        )
    except RuntimeError as exc:
        assert "probe failed and fallback disabled" in str(exc)
        return
    raise AssertionError("Expected _resolve_wandb_offline to raise RuntimeError")


def test_resolve_wandb_offline_auto_respects_env_offline(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(
        "scripts.run_sft_local._probe_wandb_online",
        lambda timeout_seconds: (_ for _ in ()).throw(AssertionError("probe should not run")),
    )
    offline, reason = _resolve_wandb_offline(
        wandb_mode="auto",
        allow_offline_fallback=True,
        probe_timeout=1.0,
        dry_run=False,
    )
    assert offline is True
    assert reason == "auto_env_offline"


def test_resolve_wandb_offline_auto_probe_fail_fallback(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(
        "scripts.run_sft_local._probe_wandb_online",
        lambda timeout_seconds: (False, "AuthenticationError: blocked"),
    )
    offline, reason = _resolve_wandb_offline(
        wandb_mode="auto",
        allow_offline_fallback=True,
        probe_timeout=1.0,
        dry_run=False,
    )
    assert offline is True
    assert "auto_fallback_offline" in reason
