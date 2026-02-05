from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


TASKS = [
    "bps",
    "chegeka",
    "lcs",
    "mamuramu",
    "mathlogicqa",
    "multiq",
    "parus",
    "rcb",
    "rudetox",
    "rummlu",
    "rumodar",
    "rumultiar",
    "ruopenbookqa",
    "rutie",
    "ruworldtree",
    "rwsd",
    "simplear",
    "use",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Prime-RL GRPO for a single MERA task.")
    parser.add_argument("task", choices=TASKS)
    parser.add_argument(
        "--prime-rl-dir",
        type=Path,
        default=None,
        help="Path to the prime-rl repository (defaults to <repo>/_deps/prime-rl).",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "prime_rl",
        help="Root directory containing per-task Prime-RL configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the run (defaults to outputs/grpo_prime/<task>).",
    )
    parser.add_argument(
        "--trainer-gpu-ids",
        default="[0]",
        help="Trainer GPU IDs as a JSON list, e.g. '[0]'.",
    )
    parser.add_argument(
        "--inference-gpu-ids",
        default="[0]",
        help="Inference GPU IDs as a JSON list, e.g. '[0]'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Prime-RL command without running it.",
    )
    return parser.parse_known_args()


def build_command(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    task_dir = args.config_root / args.task
    train_path = task_dir / "train.toml"
    orch_path = task_dir / "orch.toml"
    infer_path = task_dir / "infer.toml"

    missing = [path for path in [train_path, orch_path, infer_path] if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Prime-RL config(s): {missing_list}")

    output_dir = args.output_dir or Path("outputs") / "grpo_prime" / args.task

    cmd = [
        "uv",
        "run",
        "rl",
        "--trainer",
        "@",
        str(train_path),
        "--orchestrator",
        "@",
        str(orch_path),
        "--inference",
        "@",
        str(infer_path),
        "--trainer-gpu-ids",
        args.trainer_gpu_ids,
        "--inference-gpu-ids",
        args.inference_gpu_ids,
        "--output-dir",
        str(output_dir),
    ]
    return cmd + extra_args


def resolve_prime_rl_dir(args: argparse.Namespace) -> Path:
    if args.prime_rl_dir is not None:
        return args.prime_rl_dir
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "_deps" / "prime-rl"


def main() -> None:
    args, extra_args = parse_args()
    prime_rl_dir = resolve_prime_rl_dir(args)
    if not prime_rl_dir.exists():
        raise FileNotFoundError(f"prime-rl repo not found: {prime_rl_dir}")
    cmd = build_command(args, extra_args)
    if args.dry_run:
        print(f"(prime-rl cwd) {prime_rl_dir}")
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True, cwd=prime_rl_dir)


if __name__ == "__main__":
    main()
