#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_SFT_BATCH_SIZE,
    DEFAULT_SFT_EPOCHS,
    DEFAULT_SFT_GRAD_ACCUM,
    DEFAULT_SFT_LORA_ALPHA,
    DEFAULT_SFT_LORA_DROPOUT,
    DEFAULT_SFT_LORA_RANK,
    DEFAULT_SFT_LR,
    DEFAULT_SFT_MAX_SEQ_LEN,
    DEFAULT_SFT_MICRO_BATCH_SIZE,
    DEFAULT_SFT_SAVE_STEPS,
    DEFAULT_SFT_WARMUP_RATIO,
    OUTPUTS_ROOT,
)
from prime_lab_rl.manifest import ensure_manifest, update_manifest
from prime_lab_rl.sft_dataset import prepare_sft_dataset_artifacts, resolve_sft_tasks


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run local Prime-RL SFT on MERA fair tasks.")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "sft")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "sft" / "mera-fair.toml")
    parser.add_argument("--task-set", choices=["fair"], default="fair")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_SFT_MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SFT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_SFT_GRAD_ACCUM)
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_SFT_MICRO_BATCH_SIZE)
    parser.add_argument("--epochs", type=float, default=DEFAULT_SFT_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=DEFAULT_SFT_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_SFT_WARMUP_RATIO)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SFT_SAVE_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pack-function", choices=["cat", "stack"], default="cat")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")

    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_SFT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=DEFAULT_SFT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_SFT_LORA_DROPOUT)

    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--torchrun-bin", default=None, help="Optional explicit torchrun binary.")
    parser.add_argument("--sft-entry", default="sft", help="Prime-RL SFT entrypoint on PATH.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "mera"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME"))
    parser.add_argument("--hf-adapter-repo-id", default=None, help="Upload the final SFT adapter to this HF repo after training.")
    parser.add_argument("--hf-merged-repo-id", default=None, help="Upload a merged full-model handoff artifact to this HF repo after training.")
    parser.add_argument("--hf-private", action="store_true", default=True)
    parser.add_argument("--hf-public", action="store_false", dest="hf_private")

    parser.add_argument("--experiment", default="manual")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()

def _infer_nproc_per_node(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        num = len([token for token in visible.split(",") if token.strip()])
        if num > 0:
            return num
    return 1


def _build_launch_command(
    *,
    torchrun_bin: str | None,
    sft_entry: str,
    config_path: Path,
    nproc_per_node: int,
    master_port: int | None,
    override_args: list[str],
    extra_args: list[str],
) -> list[str]:
    resolved_sft_entry = shutil.which(sft_entry) or sft_entry
    if nproc_per_node == 1:
        cmd = [resolved_sft_entry, "@", str(config_path)]
    else:
        cmd = [
            torchrun_bin or "torchrun",
            "--standalone",
            "--local-ranks-filter",
            "0",
            "--nproc_per_node",
            str(nproc_per_node),
        ]
        if master_port is not None:
            cmd.extend(["--master_port", str(master_port)])
        cmd.extend([resolved_sft_entry, "@", str(config_path)])
    cmd.extend(override_args)
    cmd.extend(extra_args)
    return cmd


def _resolve_latest_stable_weight_dir(output_dir: Path) -> Path:
    weights_dir = output_dir / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    stable_steps: list[tuple[int, Path]] = []
    for step_dir in weights_dir.glob("step_*"):
        if not step_dir.is_dir():
            continue
        try:
            step = int(step_dir.name.split("_")[-1])
        except ValueError:
            continue
        if (step_dir / "STABLE").exists():
            stable_steps.append((step, step_dir))

    if not stable_steps:
        raise FileNotFoundError(f"No stable Prime-RL weight checkpoints found in {weights_dir}.")

    stable_steps.sort(key=lambda item: item[0])
    return stable_steps[-1][1]


def _write_latest_pointers(output_dir: Path, latest_weight_dir: Path) -> None:
    step = latest_weight_dir.name.split("_")[-1]
    (output_dir / "LATEST_WEIGHT_STEP").write_text(f"{step}\n", encoding="utf-8")

    latest_link = output_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(latest_weight_dir.resolve())


def _run_hf_publish(
    *,
    repo_id: str,
    source_path: Path,
    artifact_type: str,
    private: bool,
    manifest_path: Path | None,
    experiment: str,
    base_model: str | None,
    merge_lora: bool,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "publish_hf_artifact.py"),
        "--kind",
        "sft",
        "--source-path",
        str(source_path),
        "--repo-id",
        repo_id,
        "--artifact-type",
        artifact_type,
        "--experiment",
        experiment,
    ]
    if manifest_path is not None:
        cmd.extend(["--manifest", str(manifest_path)])
    cmd.append("--private" if private else "--public")
    if merge_lora:
        cmd.extend(["--merge-lora", "--base-model", str(base_model)])

    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    args, extra_args = parse_args()

    if args.batch_size < 1 or args.grad_accum < 1 or args.micro_batch_size < 1:
        raise ValueError("--batch-size, --grad-accum and --micro-batch-size must be >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0.0, 1.0)")
    if args.save_steps < 1:
        raise ValueError("--save-steps must be >= 1")
    if not args.use_lora:
        raise ValueError("This repo's SFT path is LoRA-first and expects --use-lora.")
    if not args.config.exists():
        raise FileNotFoundError(f"SFT config template not found: {args.config}")
    if not args.dry_run and shutil.which(args.sft_entry) is None:
        raise FileNotFoundError(
            f"SFT entrypoint '{args.sft_entry}' not found on PATH. Install prime-rl or pass --sft-entry."
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = resolve_sft_tasks(args.task_set, args.tasks)
    dataset_missing_on_dry_run = False
    try:
        artifacts = prepare_sft_dataset_artifacts(
            output_dir=output_dir,
            data_dir=args.data_dir,
            tasks=tasks,
            limit=args.limit,
            base_model=args.model,
        )
    except FileNotFoundError:
        if not args.dry_run:
            raise
        dataset_missing_on_dry_run = True
        placeholder_dir = output_dir / "dataset"
        estimated_rows = max(1, (args.limit or 1) * max(1, len(tasks)))
        artifacts = SimpleNamespace(
            train_path=placeholder_dir / "train.jsonl",
            manifest_path=placeholder_dir / "manifest.json",
            num_rows=estimated_rows,
        )

    nproc_per_node = _infer_nproc_per_node(args.nproc_per_node)
    global_batch_size = args.batch_size * args.grad_accum * nproc_per_node
    if global_batch_size % (nproc_per_node * args.micro_batch_size) != 0:
        raise ValueError(
            "Derived Prime-RL batch_size is incompatible with world size. Ensure "
            "(batch_size * grad_accum * nproc_per_node) is divisible by "
            "(nproc_per_node * micro_batch_size)."
        )

    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = max(1, math.ceil((artifacts.num_rows * args.epochs) / max(1, global_batch_size)))

    warmup_steps = 0
    if args.warmup_ratio > 0.0 and max_steps > 1:
        warmup_steps = max(1, int(round(max_steps * args.warmup_ratio)))

    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve()
    resolved_args_path = config_dir / "sft_overrides.txt"
    override_args = [
        "--model.name",
        args.model,
        "--model.seq-len",
        str(args.max_seq_len),
        "--model.optimization-dtype",
        "bfloat16" if args.precision == "bf16" else "float32",
        "--model.reduce-dtype",
        "bfloat16" if args.precision == "bf16" else "float32",
        "--model.lora.rank",
        str(args.lora_rank),
        "--model.lora.alpha",
        str(args.lora_alpha),
        "--model.lora.dropout",
        str(args.lora_dropout),
        "--data.name",
        str(artifacts.train_path.parent.resolve()),
        "--data.seq-len",
        str(args.max_seq_len),
        "--data.batch-size",
        str(global_batch_size),
        "--data.micro-batch-size",
        str(args.micro_batch_size),
        "--data.pack-function",
        args.pack_function,
        "--data.shuffle",
        "false" if args.no_shuffle else "true",
        "--data.seed",
        str(args.seed),
        "--optim.lr",
        str(args.lr),
        "--scheduler.warmup-steps",
        str(warmup_steps),
        "--ckpt.interval",
        str(args.save_steps),
        "--max-steps",
        str(max_steps),
        "--output-dir",
        str(output_dir),
        "--wandb.project",
        args.wandb_project,
    ]
    if args.wandb_entity:
        override_args.extend(["--wandb.entity", args.wandb_entity])
    if args.wandb_run_name:
        override_args.extend(["--wandb.name", args.wandb_run_name])
    resolved_args_path.write_text(
        "\n".join(override_args) + "\n",
        encoding="utf-8",
    )

    cmd = _build_launch_command(
        torchrun_bin=args.torchrun_bin,
        sft_entry=args.sft_entry,
        config_path=config_path,
        nproc_per_node=nproc_per_node,
        master_port=args.master_port,
        override_args=override_args,
        extra_args=extra_args,
    )

    print(f"output_dir={output_dir}")
    print(f"dataset_rows={artifacts.num_rows}")
    print(f"tasks={','.join(tasks)}")
    if dataset_missing_on_dry_run:
        print("dataset_warning=MERA dataset not found; using placeholder dataset path for dry-run")
    print(f"nproc_per_node={nproc_per_node}")
    print(f"global_batch_size={global_batch_size}")
    print(f"max_steps={max_steps}")
    print(f"config_template={config_path}")
    print(f"override_args_path={resolved_args_path}")
    print(f"command={shlex.join(cmd)}")
    if args.hf_adapter_repo_id:
        print(f"hf_adapter_repo_id={args.hf_adapter_repo_id}")
    if args.hf_merged_repo_id:
        print(f"hf_merged_repo_id={args.hf_merged_repo_id}")

    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        ensure_manifest(manifest_path, experiment=args.experiment, base_model=args.model)
        update_manifest(
            manifest_path,
            {
                "sft": {
                    "dataset": {
                        "output_dir": str((output_dir / "dataset").resolve()),
                        "train_path": str(artifacts.train_path.resolve()),
                        "manifest_path": str(artifacts.manifest_path.resolve()),
                        "num_rows": artifacts.num_rows,
                        "tasks": tasks,
                        "limit": args.limit,
                    },
                    "run": {
                        "output_dir": str(output_dir),
                        "config_template": str(config_path),
                        "override_args_path": str(resolved_args_path),
                        "max_steps": max_steps,
                        "global_batch_size": global_batch_size,
                        "nproc_per_node": nproc_per_node,
                        "use_lora": args.use_lora,
                        "hf_adapter_repo_id": args.hf_adapter_repo_id,
                        "hf_merged_repo_id": args.hf_merged_repo_id,
                        "wandb": {
                            "project": args.wandb_project,
                            "entity": args.wandb_entity,
                            "name": args.wandb_run_name,
                        },
                    },
                }
            },
        )

    if args.dry_run:
        return

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    latest_weight_dir = _resolve_latest_stable_weight_dir(output_dir)
    _write_latest_pointers(output_dir, latest_weight_dir)
    print(f"latest_weight_dir={latest_weight_dir}")

    manifest_path = args.manifest.expanduser().resolve() if args.manifest is not None else None
    if manifest_path is not None:
        update_manifest(
            manifest_path,
            {
                "sft": {
                    "run": {
                        "latest_weight_dir": str(latest_weight_dir.resolve()),
                        "latest_link": str((output_dir / "latest").resolve()),
                    }
                }
            },
        )

    if args.hf_adapter_repo_id:
        _run_hf_publish(
            repo_id=args.hf_adapter_repo_id,
            source_path=output_dir / "latest",
            artifact_type="adapter",
            private=args.hf_private,
            manifest_path=manifest_path,
            experiment=args.experiment,
            base_model=args.model,
            merge_lora=False,
        )

    if args.hf_merged_repo_id:
        _run_hf_publish(
            repo_id=args.hf_merged_repo_id,
            source_path=output_dir / "latest",
            artifact_type="merged",
            private=True,
            manifest_path=manifest_path,
            experiment=args.experiment,
            base_model=args.model,
            merge_lora=True,
        )


if __name__ == "__main__":
    main()
