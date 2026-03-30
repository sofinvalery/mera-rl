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
ID_RE = re.compile(r'^\s*id\s*=\s*"([^"]+)"\s*$')


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


def _upsert_top_level_key(toml_text: str, *, key: str, value_literal: str) -> str:
    lines = toml_text.splitlines()
    first_section_idx = next((idx for idx, line in enumerate(lines) if _section_name(line) is not None), len(lines))

    out = lines[:]
    replaced = False
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for idx in range(first_section_idx):
        if pattern.match(out[idx]):
            out[idx] = f"{key} = {value_literal}"
            replaced = True
            break

    if not replaced:
        out.insert(first_section_idx, f"{key} = {value_literal}")

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
    except Exception as exc:  # pragma: no cover
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


def _extract_orchestrator_env_blocks(toml_text: str) -> list[tuple[str, str]]:
    lines = toml_text.splitlines()
    blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        section = _section_name(lines[i])
        if section != "orchestrator.env":
            i += 1
            continue

        j = i + 1
        while j < len(lines) and _section_name(lines[j]) is None:
            j += 1

        block_lines = lines[i:j]
        task_id: str | None = None
        for line in block_lines:
            match = ID_RE.match(line)
            if match is not None:
                task_id = match.group(1).strip()
                break
        if not task_id:
            raise ValueError(f"Failed to extract task id from block:\n{chr(10).join(block_lines)}")
        blocks.append((task_id, "\n".join(block_lines).rstrip()))
        i = j
    return blocks


def _split_items(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        for part in value.split(","):
            item = part.strip()
            if item:
                out.append(item)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def _resolve_task_order(
    *,
    template_ids: list[str],
    selected_tasks: list[str],
    task_order_arg: str,
) -> list[str]:
    unknown_selected = [task for task in selected_tasks if task not in template_ids]
    if unknown_selected:
        raise ValueError(f"Unknown task(s): {unknown_selected}. Available: {template_ids}")

    selected_set = set(selected_tasks)
    if task_order_arg.strip() == "template":
        return [task for task in template_ids if task in selected_set]

    ordered = _split_items([task_order_arg])
    if not ordered:
        raise ValueError("--task-order is empty. Use 'template' or a comma-separated list.")
    unknown_ordered = [task for task in ordered if task not in template_ids]
    if unknown_ordered:
        raise ValueError(f"Unknown task(s) in --task-order: {unknown_ordered}. Available: {template_ids}")

    missing = [task for task in selected_tasks if task not in ordered]
    extra = [task for task in ordered if task not in selected_set]
    if missing or extra:
        raise ValueError(
            f"--tasks and --task-order mismatch. missing_in_order={missing}, extra_in_order={extra}"
        )
    return ordered


def _task_config_path(run_root: Path, *, mode: Mode, index: int, task: str) -> Path:
    return run_root / "configs" / "sequential" / f"rlvr_{mode}_{index:02d}_{task}.toml"


def _task_output_dir(run_root: Path, *, mode: Mode, index: int, task: str) -> Path:
    root_name = "rlvr_sequential_smoke" if mode == "smoke" else "rlvr_sequential"
    return run_root / root_name / f"{index:02d}_{task}"


def _expected_checkpoint_path(run_root: Path, *, mode: Mode, index: int, task: str, steps_per_task: int) -> Path:
    return _task_output_dir(run_root, mode=mode, index=index, task=task) / "weights" / f"step_{steps_per_task}"


def _latest_stable_checkpoint(output_dir: Path) -> Path | None:
    weight_dir = output_dir / "weights"
    if not weight_dir.exists():
        return None
    stable_steps = [p for p in weight_dir.glob("step_*") if (p / "STABLE").exists()]
    if not stable_steps:
        return None
    stable_steps.sort(key=lambda p: int(p.name.split("_")[-1]))
    return stable_steps[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local MERA RLVR sequentially (task-by-task) with prime-rl.")
    parser.add_argument("mode", choices=["smoke", "train"])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps-per-task", type=int, default=None, help="Per-task max_steps. Train defaults to 100.")
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional task subset. Accepts space- or comma-separated values.")
    parser.add_argument(
        "--task-order",
        default="template",
        help="Task order: 'template' or comma-separated list matching selected tasks.",
    )
    parser.add_argument("--resume-from-task", default=None, help="Resume from this task id in the resolved task order.")
    parser.add_argument("--continue-on-error", action="store_true", default=False)
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
        help="HF model repo id used for initial RLVR model weights.",
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
    manifest_path = run_root / "manifest.json"

    if mode == "smoke":
        template_path = REPO_ROOT / "configs" / "rl" / "local" / "mera-rlvr-smoke.toml"
    elif mode == "train":
        template_path = REPO_ROOT / "configs" / "rl" / "local" / "mera-rlvr-train.toml"
    else:
        assert_never(mode)
    assert template_path.exists(), template_path

    template_text = template_path.read_text(encoding="utf-8")
    env_blocks = _extract_orchestrator_env_blocks(template_text)
    assert env_blocks, f"No [[orchestrator.env]] blocks in {template_path}"
    template_task_ids = [task_id for task_id, _ in env_blocks]
    env_block_by_id = dict(env_blocks)

    selected_tasks = _split_items(args.tasks) if args.tasks else template_task_ids[:]
    task_order = _resolve_task_order(
        template_ids=template_task_ids,
        selected_tasks=selected_tasks,
        task_order_arg=args.task_order,
    )
    assert task_order, "Resolved task order is empty."

    resume_index = 0
    if args.resume_from_task is not None:
        if args.resume_from_task not in task_order:
            raise ValueError(f"--resume-from-task '{args.resume_from_task}' is not in resolved task order: {task_order}")
        resume_index = task_order.index(args.resume_from_task)

    if args.steps_per_task is None:
        default_steps = 100 if mode == "train" else 28
        steps_per_task = default_steps
    else:
        steps_per_task = int(args.steps_per_task)
    if steps_per_task < 1:
        raise ValueError("--steps-per-task must be >= 1")

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

    model_name = args.model_repo
    model_source = "remote_repo"
    if args.model_dir:
        model_name, model_source = _ensure_local_model(
            model_repo=args.model_repo,
            model_dir=Path(args.model_dir),
            dry_run=args.dry_run,
            prepull=args.prepull_model,
        )

    ensure_manifest(manifest_path, experiment=args.experiment, base_model=SFT_MODEL)
    update_manifest(
        manifest_path,
        {
            "notes": {
                "rlvr_sequential": {
                    "mode": mode,
                    "template_path": str(template_path),
                    "steps_per_task": steps_per_task,
                    "task_order": task_order,
                    "resume_from_task": args.resume_from_task,
                    "continue_on_error": args.continue_on_error,
                    "model_name_initial": model_name,
                    "model_source_initial": model_source,
                    "wandb_mode_requested": requested_wandb_mode,
                    "wandb_mode_effective": effective_wandb_mode,
                    "wandb_mode_reason": wandb_reason,
                    "status": "running",
                }
            }
        },
    )

    env: dict[str, str] | None = None
    if not args.dry_run:
        env = os.environ.copy()
        env["PRIME_DISABLE_VERSION_CHECK"] = "1"
        if "UV_PROJECT" not in env:
            resolved_rl_entry = rl_entry if rl_entry != "rl" else (shutil.which("rl") or rl_entry)
            rl_path = Path(resolved_rl_entry)
            if rl_path.is_absolute() and len(rl_path.resolve().parents) > 2:
                uv_project = rl_path.resolve().parents[2]
                if (uv_project / "pyproject.toml").exists():
                    env["UV_PROJECT"] = str(uv_project)

    current_model_name = model_name
    if resume_index > 0:
        prev_task = task_order[resume_index - 1]
        prev_idx = resume_index
        if args.dry_run:
            resume_ckpt = _expected_checkpoint_path(
                run_root,
                mode=mode,
                index=prev_idx,
                task=prev_task,
                steps_per_task=steps_per_task,
            )
            current_model_name = str(resume_ckpt)
        else:
            prev_output_dir = _task_output_dir(run_root, mode=mode, index=prev_idx, task=prev_task)
            resume_ckpt = _latest_stable_checkpoint(prev_output_dir)
            if resume_ckpt is None:
                raise FileNotFoundError(
                    f"Resume requested from {args.resume_from_task}, but no stable checkpoint for previous task '{prev_task}' in {prev_output_dir}"
                )
            current_model_name = str(resume_ckpt)

    print(f"mode={mode}")
    print(f"experiment={args.experiment}")
    print(f"template={template_path}")
    print(f"steps_per_task={steps_per_task}")
    print(f"model_name_initial={model_name}")
    print(f"model_source_initial={model_source}")
    print(f"wandb_mode_requested={requested_wandb_mode}")
    print(f"wandb_mode_effective={effective_wandb_mode}")
    print(f"wandb_mode_reason={wandb_reason}")
    print(f"task_order={','.join(task_order)}")
    print(f"resume_from_task={args.resume_from_task or ''}")

    failures: list[str] = []
    last_checkpoint_path: Path | None = None
    executed_tasks: list[str] = []

    for index, task in enumerate(task_order, start=1):
        if index <= resume_index:
            continue

        task_config_path = _task_config_path(run_root, mode=mode, index=index, task=task)
        task_output_dir = _task_output_dir(run_root, mode=mode, index=index, task=task)
        task_config_path.parent.mkdir(parents=True, exist_ok=True)

        task_text = template_text
        task_text = _upsert_top_level_key(task_text, key="max_steps", value_literal=str(steps_per_task))
        task_text = _upsert_section_key(task_text, section="model", key="name", value_literal=_toml_quote(current_model_name))
        task_text = _apply_wandb_mode(task_text, mode=effective_wandb_mode)
        task_text = _remove_sections(task_text, ("orchestrator.env",))
        task_text = task_text.rstrip() + "\n\n" + env_block_by_id[task].rstrip() + "\n"
        task_config_path.write_text(task_text, encoding="utf-8")

        cmd = [
            rl_entry,
            "@",
            str(task_config_path),
            "--output-dir",
            str(task_output_dir),
        ]

        print(f"task={task}")
        print(f"task_index={index}/{len(task_order)}")
        print(f"task_config={task_config_path}")
        print(f"task_output_dir={task_output_dir}")
        print(f"task_model_name={current_model_name}")
        print(f"task_command={shlex.join(cmd)}")

        if args.dry_run:
            executed_tasks.append(task)
            current_model_name = str(
                _expected_checkpoint_path(
                    run_root,
                    mode=mode,
                    index=index,
                    task=task,
                    steps_per_task=steps_per_task,
                )
            )
            update_manifest(
                manifest_path,
                {
                    "notes": {
                        "rlvr_sequential": {
                            "tasks": {
                                task: {
                                    "index": index,
                                    "status": "dry_run",
                                    "steps": steps_per_task,
                                    "config_path": str(task_config_path),
                                    "output_dir": str(task_output_dir),
                                    "model_name": current_model_name,
                                }
                            }
                        }
                    }
                },
            )
            continue

        try:
            assert env is not None
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
            final_step_path = _latest_stable_checkpoint(task_output_dir)
            if final_step_path is None:
                raise RuntimeError(f"No stable checkpoint in {task_output_dir / 'weights'}")
            last_checkpoint_path = final_step_path
            current_model_name = str(final_step_path)
            executed_tasks.append(task)
            update_manifest(
                manifest_path,
                {
                    "notes": {
                        "rlvr_sequential": {
                            "tasks": {
                                task: {
                                    "index": index,
                                    "status": "completed",
                                    "steps": steps_per_task,
                                    "config_path": str(task_config_path),
                                    "output_dir": str(task_output_dir),
                                    "final_checkpoint_path": str(final_step_path),
                                }
                            }
                        }
                    }
                },
            )
        except Exception as exc:
            failures.append(task)
            update_manifest(
                manifest_path,
                {
                    "notes": {
                        "rlvr_sequential": {
                            "tasks": {
                                task: {
                                    "index": index,
                                    "status": "failed",
                                    "steps": steps_per_task,
                                    "config_path": str(task_config_path),
                                    "output_dir": str(task_output_dir),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            }
                        }
                    }
                },
            )
            if not args.continue_on_error:
                raise
            print(f"task_failed={task}")

    final_status = "completed"
    if failures:
        final_status = "completed_with_failures"
    if args.dry_run:
        final_status = "dry_run"

    update_manifest(
        manifest_path,
        {
            "notes": {
                "rlvr_sequential": {
                    "status": final_status,
                    "executed_tasks": executed_tasks,
                    "failed_tasks": failures,
                    "final_chained_checkpoint": str(last_checkpoint_path) if last_checkpoint_path else current_model_name,
                }
            }
        },
    )

    if mode != "train" or args.dry_run:
        return

    if failures:
        print("publish_skipped=failures_present")
        return

    assert last_checkpoint_path is not None, "No final chained checkpoint available for publish."

    hf_repo_id = os.getenv("HF_RL_REPO_ID")
    assert hf_repo_id, "HF_RL_REPO_ID is required for train mode."
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    assert hf_token, "Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN before train mode."
    assert env is not None
    env["HUGGINGFACE_HUB_TOKEN"] = hf_token

    publish_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "publish_hf_artifact.py"),
        "--kind",
        "rl",
        "--source-path",
        str(last_checkpoint_path),
        "--repo-id",
        hf_repo_id,
        "--artifact-type",
        "checkpoint",
        "--task",
        "fair_sequential",
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
                "fair_local_sequential": {
                    "status": "completed",
                    "steps_per_task": steps_per_task,
                    "task_order": task_order,
                    "final_checkpoint_path": str(last_checkpoint_path),
                    "hf_repo_id": hf_repo_id,
                }
            }
        },
    )


if __name__ == "__main__":
    main()
