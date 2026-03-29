#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Literal, assert_never
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.manifest import ensure_manifest, update_manifest


Mode = Literal["smoke", "train"]
WandbMode = Literal["auto", "online", "offline", "disabled"]
SFT_MODEL = "sofinvalery/mera-qwen3-4b-sft"
DEFAULT_MODEL_DIR = "/workspace/models/mera-qwen3-4b-sft"
HEADER_RE = re.compile(r"^\s*\[\[?([^\]]+)\]?\]\s*$")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _default_model_dir() -> str | None:
    model_dir = os.getenv("MERA_RLVR_MODEL_DIR")
    if model_dir:
        return model_dir
    if Path("/workspace").exists():
        return DEFAULT_MODEL_DIR
    return None


def _toml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _section_name(line: str) -> str | None:
    match = HEADER_RE.match(line)
    if match is None:
        return None
    return match.group(1).strip()


def _remove_sections(toml_text: str, prefixes: tuple[str, ...]) -> str:
    lines = toml_text.splitlines()
    out: list[str] = []
    skipping = False

    for line in lines:
        section = _section_name(line)
        if section is not None:
            skipping = any(section == prefix or section.startswith(f"{prefix}.") for prefix in prefixes)
            if skipping:
                continue
        if not skipping:
            out.append(line)

    return ("\n".join(out).rstrip() + "\n") if out else ""


def _upsert_section_key(toml_text: str, *, section: str, key: str, value_literal: str) -> str:
    lines = toml_text.splitlines()
    out: list[str] = []
    in_target = False
    found_section = False
    wrote_key = False

    for line in lines:
        current_section = _section_name(line)
        if current_section is not None:
            if in_target and not wrote_key:
                out.append(f"{key} = {value_literal}")
                wrote_key = True
            in_target = current_section == section
            if in_target:
                found_section = True
                wrote_key = False
            out.append(line)
            continue

        if in_target and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            if not wrote_key:
                out.append(f"{key} = {value_literal}")
                wrote_key = True
            continue

        out.append(line)

    if in_target and not wrote_key:
        out.append(f"{key} = {value_literal}")
        wrote_key = True

    if not found_section:
        if out and out[-1].strip():
            out.append("")
        out.append(f"[{section}]")
        out.append(f"{key} = {value_literal}")

    return "\n".join(out).rstrip() + "\n"


def _apply_wandb_mode(toml_text: str, *, mode: WandbMode) -> str:
    if mode == "disabled":
        return _remove_sections(toml_text, ("wandb", "trainer.wandb", "orchestrator.wandb"))
    if mode == "offline":
        updated = _upsert_section_key(toml_text, section="wandb", key="offline", value_literal="true")
        return _upsert_section_key(updated, section="wandb", key="shared", value_literal="false")
    if mode == "online":
        return _upsert_section_key(toml_text, section="wandb", key="offline", value_literal="false")
    return toml_text


def _probe_wandb_online(timeout_seconds: float) -> tuple[bool, str]:
    timeout = max(1.0, float(timeout_seconds))
    request = urllib.request.Request("https://api.wandb.ai/graphql", method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, f"http_{int(response.getcode())}"
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        if status in (200, 401, 404, 405):
            return True, f"http_{status}"
        return False, f"http_{status}"
    except Exception as exc:  # pragma: no cover - defensive branch
        return False, f"{type(exc).__name__}: {exc}"


def _resolve_wandb_mode(
    *,
    mode: Mode,
    requested_mode: WandbMode,
    allow_offline_fallback: bool,
    probe_timeout: float,
    dry_run: bool,
) -> tuple[WandbMode, str]:
    if mode == "smoke":
        return "disabled", "smoke_forced_disabled"

    if requested_mode == "disabled":
        return "disabled", "forced_disabled"
    if requested_mode == "offline":
        return "offline", "forced_offline"
    if requested_mode == "online":
        if dry_run:
            return "online", "forced_online_dry_run"
        ok, detail = _probe_wandb_online(timeout_seconds=probe_timeout)
        if ok:
            return "online", "forced_online_probe_ok"
        if allow_offline_fallback:
            return "offline", f"forced_online_fallback_offline: {detail}"
        raise RuntimeError(f"W&B online probe failed and fallback disabled: {detail}")

    env_wandb_mode = os.getenv("WANDB_MODE", "").strip().lower()
    if env_wandb_mode == "offline":
        return "offline", "auto_env_offline"
    if dry_run:
        return "online", "auto_dry_run"
    ok, detail = _probe_wandb_online(timeout_seconds=probe_timeout)
    if ok:
        return "online", "auto_probe_ok"
    if allow_offline_fallback:
        return "offline", f"auto_fallback_offline: {detail}"
    raise RuntimeError(f"W&B auto mode probe failed and fallback disabled: {detail}")


def _ensure_local_model(
    *,
    model_repo: str,
    model_dir: Path,
    dry_run: bool,
    prepull: bool,
) -> tuple[str, str]:
    local_dir = model_dir.expanduser().resolve()
    has_model = (local_dir / "config.json").exists()
    if has_model:
        return str(local_dir), "cached"
    if dry_run:
        return str(local_dir), "dry_run_not_downloaded"
    if not prepull:
        return model_repo, "remote_repo_no_prepull"

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_repo, local_dir=str(local_dir), resume_download=True)
    return str(local_dir), "downloaded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local MERA RLVR with prime-rl.")
    parser.add_argument("mode", choices=["smoke", "train"])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default=os.getenv("MERA_RLVR_WANDB_MODE", "offline"),
        help="W&B mode policy for train runs. Smoke always disables W&B.",
    )
    parser.add_argument(
        "--wandb-online-probe-timeout",
        type=float,
        default=float(os.getenv("MERA_RLVR_WANDB_ONLINE_PROBE_TIMEOUT", "3")),
        help="Timeout in seconds for W&B online reachability probe.",
    )
    parser.add_argument("--wandb-offline-fallback", action="store_true", default=_env_flag("MERA_RLVR_WANDB_OFFLINE_FALLBACK", True))
    parser.add_argument("--no-wandb-offline-fallback", action="store_false", dest="wandb_offline_fallback")
    parser.add_argument(
        "--model-repo",
        default=os.getenv("MERA_RLVR_MODEL_REPO", SFT_MODEL),
        help="HF model repo id used for RLVR model weights.",
    )
    parser.add_argument(
        "--model-dir",
        default=_default_model_dir(),
        help="Optional local path for pre-pulled model weights. Defaults to /workspace/models/... when available.",
    )
    parser.add_argument("--prepull-model", action="store_true", default=_env_flag("MERA_RLVR_PREPULL_MODEL", True))
    parser.add_argument("--no-prepull-model", action="store_false", dest="prepull_model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode: Mode = args.mode
    requested_wandb_mode: WandbMode = args.wandb_mode
    run_root = REPO_ROOT / "outputs" / "runs" / args.experiment
    output_dir = run_root / ("rlvr_smoke" if mode == "smoke" else "rlvr")
    manifest_path = run_root / "manifest.json"
    config_copy_path = run_root / "configs" / f"rlvr_{mode}.toml"
    config_copy_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "smoke":
        template_path = REPO_ROOT / "configs" / "rl" / "local" / "mera-rlvr-smoke.toml"
    elif mode == "train":
        template_path = REPO_ROOT / "configs" / "rl" / "local" / "mera-rlvr-train.toml"
    else:
        assert_never(mode)
    assert template_path.exists(), template_path

    rl_entry = os.getenv("PRIME_RL_ENTRY") or shutil.which("rl")
    if args.dry_run and rl_entry is None:
        rl_entry = "rl"
    assert rl_entry is not None, "PRIME_RL_ENTRY is not set and 'rl' is not on PATH."
    if not args.dry_run:
        assert Path(rl_entry).exists() or rl_entry == "rl", rl_entry

    effective_wandb_mode, wandb_reason = _resolve_wandb_mode(
        mode=mode,
        requested_mode=requested_wandb_mode,
        allow_offline_fallback=args.wandb_offline_fallback,
        probe_timeout=args.wandb_online_probe_timeout,
        dry_run=args.dry_run,
    )

    config_text = template_path.read_text(encoding="utf-8")
    model_name = args.model_repo
    model_source = "remote_repo"
    if args.model_dir:
        model_name, model_source = _ensure_local_model(
            model_repo=args.model_repo,
            model_dir=Path(args.model_dir),
            dry_run=args.dry_run,
            prepull=args.prepull_model,
        )
    config_text = _upsert_section_key(config_text, section="model", key="name", value_literal=_toml_quote(model_name))
    config_text = _apply_wandb_mode(config_text, mode=effective_wandb_mode)
    config_copy_path.write_text(config_text, encoding="utf-8")

    ensure_manifest(manifest_path, experiment=args.experiment, base_model=SFT_MODEL)

    cmd = [
        rl_entry,
        "@",
        str(config_copy_path),
        "--output-dir",
        str(output_dir),
    ]

    print(f"mode={mode}")
    print(f"experiment={args.experiment}")
    print(f"config={config_copy_path}")
    print(f"output_dir={output_dir}")
    print(f"model_name={model_name}")
    print(f"model_source={model_source}")
    print(f"wandb_mode_requested={requested_wandb_mode}")
    print(f"wandb_mode_effective={effective_wandb_mode}")
    print(f"wandb_mode_reason={wandb_reason}")
    print(f"command={shlex.join(cmd)}")

    if args.dry_run:
        return

    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    if "UV_PROJECT" not in env:
        resolved_rl_entry = rl_entry if rl_entry != "rl" else (shutil.which("rl") or rl_entry)
        rl_path = Path(resolved_rl_entry)
        if rl_path.is_absolute():
            # prime-rl launcher uses nested `uv run ...`; pin uv project to the rl installation root.
            uv_project = rl_path.resolve().parents[2]
            if (uv_project / "pyproject.toml").exists():
                env["UV_PROJECT"] = str(uv_project)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    if mode == "smoke":
        update_manifest(
            manifest_path,
            {
                "notes": {
                    "rlvr_smoke": {
                        "config_path": str(config_copy_path),
                        "output_dir": str(output_dir),
                        "status": "completed",
                    }
                }
            },
        )
        return

    weight_dir = output_dir / "weights"
    stable_steps = [p for p in weight_dir.glob("step_*") if (p / "STABLE").exists()]
    assert stable_steps, f"No stable checkpoint in {weight_dir}"
    stable_steps.sort(key=lambda p: int(p.name.split("_")[-1]))
    final_step_path = stable_steps[-1]

    hf_repo_id = os.getenv("HF_RL_REPO_ID")
    assert hf_repo_id, "HF_RL_REPO_ID is required for train mode."
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    assert hf_token, "Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN before train mode."
    env["HUGGINGFACE_HUB_TOKEN"] = hf_token

    publish_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "publish_hf_artifact.py"),
        "--kind",
        "rl",
        "--source-path",
        str(final_step_path),
        "--repo-id",
        hf_repo_id,
        "--artifact-type",
        "checkpoint",
        "--task",
        "fair",
        "--manifest",
        str(manifest_path),
        "--experiment",
        args.experiment,
        "--public",
    ]
    print(f"publish_command={shlex.join(publish_cmd)}")
    subprocess.run(publish_cmd, check=True, cwd=REPO_ROOT, env=env)

    update_manifest(
        manifest_path,
        {
            "rl_runs": {
                "fair_local": {
                    "config_path": str(config_copy_path),
                    "output_dir": str(output_dir),
                    "final_checkpoint_path": str(final_step_path),
                    "hf_repo_id": hf_repo_id,
                    "status": "completed",
                }
            }
        },
    )


if __name__ == "__main__":
    main()
