#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_RL_CHECKPOINT_INTERVAL,
    DEFAULT_RL_KEEP_CLOUD,
    DEFAULT_RL_LORA_ALPHA,
    DEFAULT_RL_LEARNING_RATE,
    FAIR_EVAL_TASKS,
    OUTPUTS_ROOT,
    RUNS_ROOT,
)
from prime_lab_rl.manifest import ensure_manifest, load_manifest, update_manifest


TERMINAL_RUN_STATUSES = {"COMPLETED", "FAILED", "STOPPED", "CANCELLED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MERA SFT -> hosted RL pipeline.")
    parser.add_argument("--stage", choices=["pre-rl", "submit-rl", "wait-rl", "finalize", "all"], required=True)
    parser.add_argument("--experiment", default=time.strftime("mera_%Y%m%d_%H%M%S"))
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--owner", default=os.getenv("PRIME_ENV_OWNER"))
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "mera"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))

    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--skip-intermediate-eval", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")

    parser.add_argument("--baseline-endpoint-id", default=None)
    parser.add_argument("--baseline-endpoint-url", default=None)
    parser.add_argument("--baseline-model-name", default=None)
    parser.add_argument("--baseline-api-key-env", default=None)

    parser.add_argument("--intermediate-endpoint-id", default=None)
    parser.add_argument("--intermediate-endpoint-url", default=None)
    parser.add_argument("--intermediate-model-name", default=None)
    parser.add_argument("--intermediate-api-key-env", default=None)

    parser.add_argument("--final-endpoint-id", default=None)
    parser.add_argument("--final-endpoint-url", default=None)
    parser.add_argument("--final-model-name", default=None)
    parser.add_argument("--final-api-key-env", default=None)

    parser.add_argument("--sft-output-dir", type=Path, default=None)
    parser.add_argument("--sft-adapter-repo-id", default=None)
    parser.add_argument("--sft-handoff-repo-id", default=None)
    parser.add_argument("--sft-private", action="store_true", default=True)
    parser.add_argument("--sft-public", action="store_false", dest="sft_private")

    parser.add_argument("--rl-tasks", nargs="+", default=FAIR_EVAL_TASKS)
    parser.add_argument("--rl-learning-rate", type=float, default=DEFAULT_RL_LEARNING_RATE)
    parser.add_argument("--rl-lora-alpha", type=int, default=DEFAULT_RL_LORA_ALPHA)
    parser.add_argument("--rl-checkpoint-interval", type=int, default=DEFAULT_RL_CHECKPOINT_INTERVAL)
    parser.add_argument("--rl-keep-cloud", type=int, default=DEFAULT_RL_KEEP_CLOUD)
    parser.add_argument("--rl-env-file", action="append", default=[])
    parser.add_argument("--rl-env-var", action="append", default=[])
    parser.add_argument("--wait-poll-seconds", type=int, default=60)

    parser.add_argument("--final-rl-repo-prefix", default=None, help="Prefix like owner/model-name; task is appended.")
    parser.add_argument("--final-rl-private", action="store_true", default=True)
    parser.add_argument("--final-rl-public", action="store_false", dest="final_rl_private")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_json(cmd: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    proc = subprocess.run(cmd, check=True, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    return json.loads(proc.stdout)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        return
    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _endpoint_args(prefix: str, args: argparse.Namespace) -> list[str]:
    endpoint_id = getattr(args, f"{prefix}_endpoint_id")
    endpoint_url = getattr(args, f"{prefix}_endpoint_url")
    model_name = getattr(args, f"{prefix}_model_name")
    api_key_env = getattr(args, f"{prefix}_api_key_env")
    rendered_args: list[str] = []
    if endpoint_id:
        rendered_args.extend(["--endpoint-id", endpoint_id])
    if endpoint_url:
        rendered_args.extend(["--endpoint-url", endpoint_url])
    if model_name:
        rendered_args.extend(["--model-name", model_name])
    if api_key_env:
        rendered_args.extend(["--api-key-env", api_key_env])
    return rendered_args


def _require_endpoint(prefix: str, args: argparse.Namespace) -> None:
    endpoint_id = getattr(args, f"{prefix}_endpoint_id")
    endpoint_url = getattr(args, f"{prefix}_endpoint_url")
    if not endpoint_id and not endpoint_url:
        raise ValueError(f"{prefix} eval requires either --{prefix}-endpoint-id or --{prefix}-endpoint-url")


def _latest_sft_handoff_model(args: argparse.Namespace) -> str:
    if args.sft_handoff_repo_id:
        return args.sft_handoff_repo_id
    if args.sft_adapter_repo_id:
        return args.sft_adapter_repo_id
    return args.base_model


def _submit_rl_run(
    *,
    task: str,
    args: argparse.Namespace,
    run_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    config_path = run_root / "configs" / "rl" / f"{task}.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    render_cmd = [
        "python3",
        str(REPO_ROOT / "scripts" / "render_hosted_config.py"),
        "--task",
        task,
        "--owner",
        args.owner,
        "--output",
        str(config_path),
        "--model",
        _latest_sft_handoff_model(args),
        "--learning-rate",
        str(args.rl_learning_rate),
        "--lora-alpha",
        str(args.rl_lora_alpha),
        "--checkpoint-interval",
        str(args.rl_checkpoint_interval),
        "--checkpoint-keep-cloud",
        str(args.rl_keep_cloud),
        "--run-name",
        f"{args.experiment}-{task}",
        "--wandb-project",
        args.wandb_project,
    ]
    if args.wandb_entity:
        render_cmd.extend(["--wandb-entity", args.wandb_entity])
    _run(render_cmd, dry_run=args.dry_run)

    rl_cmd = ["prime", "rl", "run", str(config_path), "--output", "json"]
    for env_file in args.rl_env_file:
        rl_cmd.extend(["--env-file", env_file])
    for env_var in args.rl_env_var:
        rl_cmd.extend(["--env-var", env_var])

    print(f"$ {' '.join(rl_cmd)}")
    if args.dry_run:
        run_info = {"id": f"dry-run-{task}", "status": "DRY_RUN", "name": f"{args.experiment}-{task}"}
    else:
        payload = _run_json(rl_cmd)
        run_info = payload["run"]

    update_manifest(
        manifest_path,
        {
            "rl_runs": {
                task: {
                    "config_path": str(config_path),
                    "run": run_info,
                }
            }
        },
    )
    return run_info


def _wait_for_rl_runs(*, manifest_path: Path, poll_seconds: int, dry_run: bool) -> None:
    if dry_run:
        return
    while True:
        manifest = load_manifest(manifest_path)
        statuses: dict[str, str] = {}
        for task, payload in manifest.get("rl_runs", {}).items():
            run_id = payload.get("run", {}).get("id")
            if not run_id:
                continue
            info = _run_json(["prime", "rl", "get", run_id, "--output", "json"])["run"]
            statuses[task] = info["status"]
            update_manifest(manifest_path, {"rl_runs": {task: {"run": info}}})

        if statuses and all(status in TERMINAL_RUN_STATUSES for status in statuses.values()):
            return
        if not statuses:
            raise ValueError("No RL runs found in manifest.")
        time.sleep(max(5, poll_seconds))


def _download_and_publish_rl_artifacts(
    *,
    args: argparse.Namespace,
    run_root: Path,
    manifest_path: Path,
) -> None:
    manifest = load_manifest(manifest_path)
    for task, payload in manifest.get("rl_runs", {}).items():
        run_id = payload.get("run", {}).get("id")
        if not run_id:
            continue
        checkpoint_dir = run_root / "rl_checkpoints" / task
        download_cmd = [
            "python3",
            str(REPO_ROOT / "scripts" / "download_hosted_checkpoint.py"),
            "--run-id",
            run_id,
            "--output-dir",
            str(checkpoint_dir),
            "--task",
            task,
            "--manifest",
            str(manifest_path),
            "--experiment",
            args.experiment,
            "--extract",
        ]
        _run(download_cmd, dry_run=args.dry_run)

        if args.final_rl_repo_prefix:
            repo_id = f"{args.final_rl_repo_prefix.rstrip('/')}-{task}"
            latest_manifest = load_manifest(manifest_path)
            extracted_path = latest_manifest.get("rl_runs", {}).get(task, {}).get("checkpoint", {}).get("extracted_path")
            source_path = Path(extracted_path) if extracted_path else checkpoint_dir
            publish_cmd = [
                "python3",
                str(REPO_ROOT / "scripts" / "publish_hf_artifact.py"),
                "--kind",
                "rl",
                "--source-path",
                str(source_path),
                "--repo-id",
                repo_id,
                "--artifact-type",
                "checkpoint",
                "--task",
                task,
                "--manifest",
                str(manifest_path),
                "--experiment",
                args.experiment,
            ]
            if args.final_rl_private:
                publish_cmd.append("--private")
            else:
                publish_cmd.append("--public")
            _run(publish_cmd, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    if args.stage in {"pre-rl", "submit-rl", "all"} and not args.owner:
        raise ValueError("--owner or PRIME_ENV_OWNER is required for hosted RL submission")

    run_root = (args.run_root or (RUNS_ROOT / args.experiment)).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest or (run_root / "manifest.json")).expanduser().resolve()
    ensure_manifest(manifest_path, experiment=args.experiment, base_model=args.base_model)

    sft_output_dir = (args.sft_output_dir or (run_root / "sft")).expanduser().resolve()

    if args.stage in {"pre-rl", "all"}:
        if not args.skip_baseline_eval:
            _require_endpoint("baseline", args)
            baseline_cmd = [
                "python3",
                str(REPO_ROOT / "scripts" / "run_eval_stage.py"),
                "--stage",
                "baseline",
                "--experiment",
                args.experiment,
                "--manifest",
                str(manifest_path),
                "--run-root",
                str(run_root),
                "--task-set",
                "fair",
                *(_endpoint_args("baseline", args)),
            ]
            _run(baseline_cmd, dry_run=args.dry_run)

        if not args.skip_sft:
            sft_cmd = [
                "python3",
                str(REPO_ROOT / "scripts" / "run_sft_local.py"),
                "--model",
                args.base_model,
                "--output-dir",
                str(sft_output_dir),
                "--experiment",
                args.experiment,
                "--manifest",
                str(manifest_path),
                "--wandb-project",
                args.wandb_project,
            ]
            if args.wandb_entity:
                sft_cmd.extend(["--wandb-entity", args.wandb_entity])
            _run(sft_cmd, dry_run=args.dry_run)

        if args.sft_adapter_repo_id:
            adapter_source = sft_output_dir / "latest"
            publish_adapter_cmd = [
                "python3",
                str(REPO_ROOT / "scripts" / "publish_hf_artifact.py"),
                "--kind",
                "sft",
                "--source-path",
                str(adapter_source),
                "--repo-id",
                args.sft_adapter_repo_id,
                "--artifact-type",
                "adapter",
                "--manifest",
                str(manifest_path),
                "--experiment",
                args.experiment,
            ]
            publish_adapter_cmd.append("--private" if args.sft_private else "--public")
            _run(publish_adapter_cmd, dry_run=args.dry_run)

        if args.sft_handoff_repo_id:
            handoff_source = sft_output_dir / "latest"
            publish_handoff_cmd = [
                "python3",
                str(REPO_ROOT / "scripts" / "publish_hf_artifact.py"),
                "--kind",
                "sft",
                "--source-path",
                str(handoff_source),
                "--repo-id",
                args.sft_handoff_repo_id,
                "--artifact-type",
                "merged",
                "--manifest",
                str(manifest_path),
                "--experiment",
                args.experiment,
                "--merge-lora",
                "--base-model",
                args.base_model,
            ]
            publish_handoff_cmd.append("--private")
            _run(publish_handoff_cmd, dry_run=args.dry_run)

        if not args.skip_intermediate_eval:
            _require_endpoint("intermediate", args)
            intermediate_cmd = [
                "python3",
                str(REPO_ROOT / "scripts" / "run_eval_stage.py"),
                "--stage",
                "intermediate",
                "--experiment",
                args.experiment,
                "--manifest",
                str(manifest_path),
                "--run-root",
                str(run_root),
                "--task-set",
                "fair",
                *(_endpoint_args("intermediate", args)),
            ]
            _run(intermediate_cmd, dry_run=args.dry_run)

    if args.stage in {"submit-rl", "all"}:
        for task in args.rl_tasks:
            _submit_rl_run(task=task, args=args, run_root=run_root, manifest_path=manifest_path)

    if args.stage in {"wait-rl", "all"}:
        _wait_for_rl_runs(manifest_path=manifest_path, poll_seconds=args.wait_poll_seconds, dry_run=args.dry_run)

    if args.stage in {"finalize", "all"}:
        _download_and_publish_rl_artifacts(args=args, run_root=run_root, manifest_path=manifest_path)
        if not args.skip_final_eval:
            _require_endpoint("final", args)
            final_cmd = [
                "python3",
                str(REPO_ROOT / "scripts" / "run_eval_stage.py"),
                "--stage",
                "final",
                "--experiment",
                args.experiment,
                "--manifest",
                str(manifest_path),
                "--run-root",
                str(run_root),
                "--task-set",
                "fair",
                *(_endpoint_args("final", args)),
            ]
            _run(final_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
