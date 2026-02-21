from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path


TASKS = [
    "chegeka",
    "lcs",
    "mamuramu",
    "mathlogicqa",
    "multiq",
    "parus",
    "rcb",
    "rucodeeval",
    "rumodar",
    "rumultiar",
    "ruopenbookqa",
    "rutie",
    "ruworldtree",
    "rwsd",
    "use",
]

FAIR_TRAIN_SPLITS = {"train", "public_test"}


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
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help=(
            "Allow GRPO on non-fair splits (e.g. test/validation). "
            "By default only train/public_test splits are allowed."
        ),
    )
    return parser.parse_known_args()


def _collect_splits_from_toml(path: Path, seen: set[Path] | None = None) -> list[str]:
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen or not path.exists():
        return []
    seen.add(path)

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    splits: list[str] = []

    for env in data.get("env", []):
        if not isinstance(env, dict):
            continue
        args = env.get("args", {})
        if isinstance(args, dict):
            split = args.get("split")
            if isinstance(split, str):
                splits.append(split)

    for include in data.get("toml_files", []):
        if not isinstance(include, str):
            continue
        child = (path.parent / include).resolve()
        splits.extend(_collect_splits_from_toml(child, seen))

    return splits


def _validate_fair_training_splits(orch_path: Path, allow_test_split: bool) -> None:
    splits = _collect_splits_from_toml(orch_path)
    non_fair = sorted({split for split in splits if split not in FAIR_TRAIN_SPLITS})
    if not non_fair or allow_test_split:
        return

    joined = ", ".join(non_fair)
    raise ValueError(
        "Refusing to run GRPO on non-fair split(s): "
        f"{joined}. Allowed by default: {sorted(FAIR_TRAIN_SPLITS)}. "
        "If this is intentional exploratory training, pass --allow-test-split."
    )


def build_rl_args(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    task_dir = args.config_root / args.task
    train_path = task_dir / "train.toml"
    orch_path = task_dir / "orch.toml"
    infer_path = task_dir / "infer.toml"

    missing = [path for path in [train_path, orch_path, infer_path] if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Prime-RL config(s): {missing_list}")

    _validate_fair_training_splits(orch_path, allow_test_split=args.allow_test_split)

    output_dir = args.output_dir or Path("outputs") / "grpo_prime" / args.task

    cmd = [
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


def resolve_rl_entrypoint(prime_rl_dir: Path) -> list[str]:
    rl_bin = prime_rl_dir / ".venv" / "bin" / "rl"
    if rl_bin.exists():
        return [str(rl_bin)]
    return ["uv", "run", "rl"]


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
    entrypoint = resolve_rl_entrypoint(prime_rl_dir)
    cmd = entrypoint + build_rl_args(args, extra_args)
    if args.dry_run:
        print(f"(prime-rl cwd) {prime_rl_dir}")
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True, cwd=prime_rl_dir)


if __name__ == "__main__":
    main()
