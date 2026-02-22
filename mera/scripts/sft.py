from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
from typing import Iterable

from sft_dataset import prepare_sft_dataset_artifacts, resolve_sft_tasks

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Prime-RL native SFT on MERA tasks.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/sft")
    parser.add_argument("--task-set", choices=["fair", "all"], default="fair")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit task subset to train on (overrides --task-set).",
    )
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device micro-batch count equivalent to legacy script semantics.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Legacy-compatible grad accumulation factor.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Prime-RL micro batch size used by the SFT trainer.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pack-function", choices=["cat", "stack"], default="cat")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA block in Prime-RL config. Disabled by default for compatibility.",
    )

    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=None)

    parser.add_argument(
        "--prime-rl-dir",
        type=Path,
        default=None,
        help="Path to the prime-rl repo (defaults to <repo>/_deps/prime-rl).",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")

    # Legacy TRL flags. We keep parsing for actionable migration errors.
    parser.add_argument("--dataloader-workers", type=int, default=None)
    parser.add_argument("--best-checkpoint", action="store_true")
    parser.add_argument("--eval-split-ratio", type=float, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--save-merged", action="store_true")
    parser.add_argument("--merged-output-dir", default=None)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")

    return parser.parse_known_args()


def _raise_on_unsupported_legacy_flags(args: argparse.Namespace) -> None:
    unsupported: list[str] = []
    if args.dataloader_workers is not None:
        unsupported.append("--dataloader-workers")
    if args.best_checkpoint:
        unsupported.append("--best-checkpoint")
    if args.eval_split_ratio is not None:
        unsupported.append("--eval-split-ratio")
    if args.eval_steps is not None:
        unsupported.append("--eval-steps")
    if args.eval_seed is not None:
        unsupported.append("--eval-seed")
    if args.save_merged:
        unsupported.append("--save-merged")
    if args.merged_output_dir is not None:
        unsupported.append("--merged-output-dir")
    if args.no_gradient_checkpointing:
        unsupported.append("--no-gradient-checkpointing")

    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            "The Prime-RL SFT rewrite no longer supports legacy TRL flags: "
            f"{joined}. Use Prime-RL checkpoints at <output_dir>/weights/step_<N>."
        )


def _escape_toml(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_string_list(values: Iterable[str]) -> str:
    quoted = [f'"{_escape_toml(v)}"' for v in values]
    return "[\n" + "\n".join(f"  {item}," for item in quoted) + "\n]"


def _infer_nproc_per_node(explicit: int | None) -> int:
    if explicit is not None:
        return explicit

    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        num = len([x for x in visible.split(",") if x.strip()])
        if num > 0:
            return num
    return 1


def _resolve_prime_rl_dir(args: argparse.Namespace) -> Path:
    if args.prime_rl_dir is not None:
        return args.prime_rl_dir.expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "_deps" / "prime-rl").resolve()


def _build_prime_sft_toml(
    args: argparse.Namespace,
    output_dir: Path,
    dataset_name: str,
    global_batch_size: int,
    max_steps: int,
    warmup_steps: int,
) -> str:
    optimization_dtype = "bfloat16" if args.precision == "bf16" else "float32"
    output_dir_s = _escape_toml(str(output_dir))
    model_s = _escape_toml(args.model)
    dataset_name_s = _escape_toml(dataset_name)

    lines = [
        f"max_steps = {max_steps}",
        f'output_dir = "{output_dir_s}"',
        "",
        "[model]",
        f'name = "{model_s}"',
        f"seq_len = {args.max_seq_len}",
        "trust_remote_code = true",
        f'optimization_dtype = "{optimization_dtype}"',
        f'reduce_dtype = "{optimization_dtype}"',
        "",
    ]

    if args.use_lora:
        lines.extend(
            [
                "[model.lora]",
                f"rank = {args.lora_rank}",
                f"alpha = {args.lora_alpha}",
                f"dropout = {args.lora_dropout}",
                f"target_modules = {_format_string_list(DEFAULT_LORA_TARGET_MODULES)}",
                "",
            ]
        )

    lines.extend(
        [
            "[data]",
            'type = "sft"',
            f'name = "{dataset_name_s}"',
            f"seq_len = {args.max_seq_len}",
            f"batch_size = {global_batch_size}",
            f"micro_batch_size = {args.micro_batch_size}",
            f'pack_function = "{args.pack_function}"',
            f"shuffle = {str(not args.no_shuffle).lower()}",
            f"seed = {args.seed}",
            "",
            "[data.loss_mask]",
            "system = false",
            "user = false",
            "assistant = true",
            "tool = false",
            "",
            "[optim]",
            'type = "adamw"',
            f"lr = {args.lr}",
            "weight_decay = 0.0",
            "max_norm = 1.0",
            "",
            "[scheduler]",
            'type = "cosine"',
            f"warmup_steps = {warmup_steps}",
            "min_lr = 0.0",
            "",
            "[ckpt]",
            f"interval = {args.save_steps}",
            "",
            "[ckpt.weights]",
            "save_sharded = true",
            'save_format = "safetensors"',
            "save_adapter_separately = false",
            "",
            "[log]",
            'level = "info"',
            "file = true",
        ]
    )

    if args.wandb:
        project = args.wandb_project or os.getenv("WANDB_PROJECT") or "rl"
        run_name = args.wandb_run_name or os.getenv("WANDB_RUN_NAME")
        lines.extend(
            [
                "",
                "[wandb]",
                f'project = "{_escape_toml(project)}"',
            ]
        )
        if run_name:
            lines.append(f'name = "{_escape_toml(run_name)}"')

    return "\n".join(lines) + "\n"


def _build_launch_command(
    prime_rl_dir: Path,
    config_path: Path,
    nproc_per_node: int,
    master_port: int | None,
    extra_args: list[str],
) -> list[str]:
    train_entry = prime_rl_dir / "src" / "prime_rl" / "trainer" / "sft" / "train.py"
    if not train_entry.exists():
        raise FileNotFoundError(f"Prime-RL SFT trainer entrypoint not found: {train_entry}")

    torchrun_bin = prime_rl_dir / ".venv" / "bin" / "torchrun"
    if torchrun_bin.exists():
        cmd = [
            str(torchrun_bin),
            "--standalone",
            "--nproc_per_node",
            str(nproc_per_node),
        ]
        if master_port is not None:
            cmd.extend(["--master_port", str(master_port)])
        cmd.extend([str(train_entry), "@", str(config_path)])
    else:
        cmd = [
            "uv",
            "run",
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(nproc_per_node),
        ]
        if master_port is not None:
            cmd.extend(["--master_port", str(master_port)])
        cmd.extend([str(train_entry), "@", str(config_path)])

    return cmd + extra_args


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
        raise FileNotFoundError(
            f"No stable Prime-RL weight checkpoints found in {weights_dir}."
        )

    stable_steps.sort(key=lambda item: item[0])
    return stable_steps[-1][1]


def _write_latest_pointers(output_dir: Path, latest_weight_dir: Path) -> None:
    step = latest_weight_dir.name.split("_")[-1]

    latest_step_path = output_dir / "LATEST_WEIGHT_STEP"
    latest_step_path.write_text(f"{step}\n", encoding="utf-8")

    latest_link = output_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        if latest_link.is_dir() and not latest_link.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite existing directory at {latest_link}. Remove it and rerun."
            )
        latest_link.unlink()

    latest_link.symlink_to(latest_weight_dir.resolve())


def main() -> None:
    args, extra_args = parse_args()
    _raise_on_unsupported_legacy_flags(args)

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Do not launch mera/scripts/sft.py with torchrun. This wrapper launches Prime-RL torchrun internally."
        )

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

    prime_rl_dir = _resolve_prime_rl_dir(args)
    if not prime_rl_dir.exists():
        raise FileNotFoundError(f"prime-rl directory not found: {prime_rl_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = resolve_sft_tasks(args.task_set, args.tasks)
    artifacts = prepare_sft_dataset_artifacts(
        output_dir=output_dir,
        data_dir=args.data_dir,
        tasks=tasks,
        limit=args.limit,
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

    dataset_dir = artifacts.train_path.parent
    dataset_name = str(dataset_dir.resolve())

    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "prime_sft.toml"
    config_text = _build_prime_sft_toml(
        args=args,
        output_dir=output_dir,
        dataset_name=dataset_name,
        global_batch_size=global_batch_size,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
    )
    config_path.write_text(config_text, encoding="utf-8")

    cmd = _build_launch_command(
        prime_rl_dir=prime_rl_dir,
        config_path=config_path,
        nproc_per_node=nproc_per_node,
        master_port=args.master_port,
        extra_args=extra_args,
    )

    print(f"Prime-RL dir: {prime_rl_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Dataset rows: {artifacts.num_rows}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"nproc_per_node: {nproc_per_node}")
    print(f"Derived Prime-RL global batch_size: {global_batch_size}")
    print(f"max_steps: {max_steps}")
    print(f"dataset_name: {dataset_name}")
    print(f"config: {config_path}")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        return

    env = os.environ.copy()

    wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT")
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
    wandb_entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    wandb_run_name = args.wandb_run_name or os.getenv("WANDB_RUN_NAME")
    if wandb_run_name:
        env["WANDB_RUN_NAME"] = wandb_run_name

    subprocess.run(cmd, check=True, cwd=prime_rl_dir, env=env)

    latest_weight_dir = _resolve_latest_stable_weight_dir(output_dir)
    _write_latest_pointers(output_dir, latest_weight_dir)

    print(f"Latest stable weight checkpoint: {latest_weight_dir}")
    print(f"Pointer symlink: {output_dir / 'latest'}")
    print(f"Latest step file: {output_dir / 'LATEST_WEIGHT_STEP'}")


if __name__ == "__main__":
    main()
