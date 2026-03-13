#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.constants import DEFAULT_BASE_MODEL
from prime_lab_rl.manifest import ensure_manifest, update_manifest
from prime_lab_rl.sft_dataset import prepare_sft_dataset_artifacts, resolve_sft_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local Prime-RL SFT dataset from MERA task splits.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write dataset artifacts into.")
    parser.add_argument("--data-dir", default=None, help="Optional explicit MERA data directory.")
    parser.add_argument("--task-set", choices=["fair"], default="fair")
    parser.add_argument("--tasks", nargs="+", default=None, help="Explicit task subset.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task example cap.")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Base model identifier recorded in the dataset manifest.")
    parser.add_argument("--experiment", default="manual", help="Experiment name for manifest bookkeeping.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional pipeline manifest to update.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    tasks = resolve_sft_tasks(args.task_set, args.tasks)
    artifacts = prepare_sft_dataset_artifacts(
        output_dir=output_dir,
        data_dir=args.data_dir,
        tasks=tasks,
        limit=args.limit,
        base_model=args.model,
    )

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
                    }
                }
            },
        )

    print(f"dataset_dir={artifacts.train_path.parent}")
    print(f"train_path={artifacts.train_path}")
    print(f"manifest_path={artifacts.manifest_path}")
    print(f"num_rows={artifacts.num_rows}")


if __name__ == "__main__":
    main()
