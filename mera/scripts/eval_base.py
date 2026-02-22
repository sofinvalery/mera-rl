from __future__ import annotations

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/eval_base")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="eval-base")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--force-score", action="store_true")
    parser.add_argument(
        "--task-set",
        choices=["all", "benchmark", "validation"],
        default="all",
        help="Task preset to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit subset of tasks to evaluate (overrides --task-set).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_script = Path(__file__).parent / "eval.py"

    cmd = [
        sys.executable,
        str(eval_script),
        "--model",
        args.model,
        "--output-dir",
        args.output_dir,
        "--split",
        args.split,
        "--temperature",
        str(args.temperature),
        "--tensor-parallel",
        str(args.tensor_parallel),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--task-set",
        args.task_set,
    ]

    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.tasks:
        cmd.extend(["--tasks", *args.tasks])
    if args.skip_scoring or (args.limit is not None and not args.force_score):
        cmd.append("--skip-scoring")
    if args.wandb:
        cmd.append("--wandb")
        if args.wandb_project:
            cmd.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_run_name:
            cmd.extend(["--wandb-run-name", args.wandb_run_name])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
