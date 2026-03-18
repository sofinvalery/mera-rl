#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.constants import DEFAULT_BASE_MODEL, DEFAULT_SFT_MAX_SEQ_LEN
from prime_lab_rl.sft_dataset import count_trainable_tokens_in_window
from prime_lab_rl.sft_dataset import _load_tokenizer as load_chat_tokenizer


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SFT rows for trainability under Prime-RL assistant-only masking."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing train.jsonl")
    parser.add_argument("--train-file", default="train.jsonl", help="Train JSONL filename under dataset-dir")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_SFT_MAX_SEQ_LEN)
    parser.add_argument(
        "--overall-threshold",
        type=float,
        default=0.005,
        help="Fail if overall zero-trainable ratio exceeds this value (default: 0.005 == 0.5%%).",
    )
    parser.add_argument(
        "--per-task-threshold",
        type=float,
        default=0.05,
        help="Fail if any task zero-trainable ratio exceeds this value (default: 0.05 == 5%%).",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--fail-on-threshold", action="store_true", default=True)
    parser.add_argument("--no-fail-on-threshold", action="store_false", dest="fail_on_threshold")
    return parser.parse_args()


def _new_task_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "zero_trainable_rows": 0,
        "prompt_tokens": [],
    }


def main() -> None:
    args = parse_args()
    if args.max_seq_len < 1:
        raise ValueError("--max-seq-len must be >= 1")
    if not 0.0 <= args.overall_threshold <= 1.0:
        raise ValueError("--overall-threshold must be in [0, 1]")
    if not 0.0 <= args.per_task_threshold <= 1.0:
        raise ValueError("--per-task-threshold must be in [0, 1]")

    train_path = args.dataset_dir.expanduser().resolve() / args.train_file
    if not train_path.exists():
        raise FileNotFoundError(f"Train JSONL not found: {train_path}")

    tokenizer = load_chat_tokenizer(args.model)
    task_stats: dict[str, dict[str, Any]] = {}
    total_rows = 0
    total_zero = 0

    with train_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = str(row.get("task", "<unknown>"))
            stats = task_stats.setdefault(task, _new_task_stats())

            trainable_tokens, prompt_tokens = count_trainable_tokens_in_window(
                row=row,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
            )
            stats["rows"] += 1
            stats["prompt_tokens"].append(prompt_tokens)
            total_rows += 1
            if trainable_tokens == 0:
                stats["zero_trainable_rows"] += 1
                total_zero += 1

            if index % 5000 == 0:
                print(f"progress_rows={index}", file=sys.stderr)

    if total_rows == 0:
        raise ValueError("Dataset is empty.")

    per_task_summary: dict[str, dict[str, Any]] = {}
    for task in sorted(task_stats):
        stats = task_stats[task]
        rows = stats["rows"]
        zero = stats["zero_trainable_rows"]
        ratio = (zero / rows) if rows else 0.0
        prompts = [int(x) for x in stats["prompt_tokens"]]
        per_task_summary[task] = {
            "rows": rows,
            "zero_trainable_rows": zero,
            "zero_trainable_ratio": ratio,
            "prompt_tokens_p50": int(_percentile(prompts, 50)),
            "prompt_tokens_p90": int(_percentile(prompts, 90)),
            "prompt_tokens_p99": int(_percentile(prompts, 99)),
        }

    overall_ratio = total_zero / total_rows
    summary = {
        "train_path": str(train_path),
        "model": args.model,
        "max_seq_len": args.max_seq_len,
        "rows": total_rows,
        "zero_trainable_rows": total_zero,
        "zero_trainable_ratio": overall_ratio,
        "overall_threshold": args.overall_threshold,
        "per_task_threshold": args.per_task_threshold,
        "task_stats": per_task_summary,
    }

    print(f"rows={total_rows}")
    print(f"zero_trainable_rows={total_zero}")
    print(f"zero_trainable_ratio={overall_ratio:.6f}")
    print(f"overall_threshold={args.overall_threshold:.6f}")
    print(f"per_task_threshold={args.per_task_threshold:.6f}")
    for task, stats in per_task_summary.items():
        print(
            "task="
            f"{task} rows={stats['rows']} zero_trainable_rows={stats['zero_trainable_rows']} "
            f"zero_trainable_ratio={stats['zero_trainable_ratio']:.6f} "
            f"prompt_tokens_p50={stats['prompt_tokens_p50']} "
            f"prompt_tokens_p90={stats['prompt_tokens_p90']} "
            f"prompt_tokens_p99={stats['prompt_tokens_p99']}"
        )

    if args.json_output is not None:
        output_path = args.json_output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json_output={output_path}")

    if not args.fail_on_threshold:
        return

    exceeded_tasks = [
        task
        for task, stats in per_task_summary.items()
        if stats["zero_trainable_ratio"] > args.per_task_threshold
    ]
    if overall_ratio > args.overall_threshold or exceeded_tasks:
        raise SystemExit(
            "SFT dataset audit failed thresholds: "
            f"overall_ratio={overall_ratio:.6f} (limit={args.overall_threshold:.6f}) "
            f"exceeded_tasks={','.join(exceeded_tasks) if exceeded_tasks else 'none'}"
        )


if __name__ == "__main__":
    main()
