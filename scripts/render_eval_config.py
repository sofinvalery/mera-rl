#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.constants import (
    DEFAULT_EVAL_NUM_EXAMPLES,
    DEFAULT_EVAL_ROLLOUTS_PER_EXAMPLE,
    FAIR_EVAL_TASKS,
    SMOKE_EVAL_TASKS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Prime eval config for a MERA stage.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--endpoints-path", type=Path, required=True)
    parser.add_argument("--task-set", choices=["fair", "smoke"], default="fair")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--num-examples", type=int, default=DEFAULT_EVAL_NUM_EXAMPLES)
    parser.add_argument("--rollouts-per-example", type=int, default=DEFAULT_EVAL_ROLLOUTS_PER_EXAMPLE)
    parser.add_argument("--save-results", action="store_true", default=True)
    parser.add_argument("--no-save-results", action="store_false", dest="save_results")
    return parser.parse_args()


def render_config(
    *,
    endpoint_id: str,
    endpoints_path: Path,
    tasks: list[str],
    num_examples: int,
    rollouts_per_example: int,
    save_results: bool,
) -> str:
    rel_endpoints_path = os.path.relpath(endpoints_path.resolve(), REPO_ROOT.resolve())
    lines = [
        f'endpoints_path = "{rel_endpoints_path}"',
        f'endpoint_id = "{endpoint_id}"',
        f"save_results = {str(save_results).lower()}",
        "",
    ]
    for task in tasks:
        lines.extend(
            [
                "[[eval]]",
                f'env_id = "{task}"',
                f"num_examples = {num_examples}",
                f"rollouts_per_example = {rollouts_per_example}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    tasks = args.tasks or (FAIR_EVAL_TASKS if args.task_set == "fair" else SMOKE_EVAL_TASKS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_config(
            endpoint_id=args.endpoint_id,
            endpoints_path=args.endpoints_path,
            tasks=tasks,
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts_per_example,
            save_results=args.save_results,
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
