#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


SUCCESS_STEP_RE = re.compile(
    r"SUCCESS Step (?P<step>\d+)\s+\|\s+Time:\s+(?P<step_time>[0-9.]+)s\s+\|"
    r".*?\|\s+Throughput:\s+(?P<throughput>[0-9.]+)\s+tokens/s\s+\|\s+MFU:\s+(?P<mfu>[0-9.]+)%",
)


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
        description="Summarize SFT throughput and MFU from Prime-RL SUCCESS Step logs."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        action="append",
        default=[],
        help="Log file(s) to parse. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional SFT output dir; parses *.log under output-dir/logs recursively.",
    )
    parser.add_argument(
        "--gpu-peak-tflops",
        type=float,
        default=None,
        help="GPU BF16 peak TFLOPS for corrected MFU (or set MERA_GPU_PEAK_TFLOPS).",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def _resolve_log_files(args: argparse.Namespace) -> list[Path]:
    files = [path.expanduser().resolve() for path in args.log_file]
    if args.output_dir is not None:
        logs_root = args.output_dir.expanduser().resolve() / "logs"
        if logs_root.exists():
            files.extend(path.resolve() for path in logs_root.rglob("*.log"))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _resolve_gpu_peak_tflops(explicit: float | None) -> float | None:
    if explicit is not None:
        return explicit
    env_value = os.getenv("MERA_GPU_PEAK_TFLOPS")
    if env_value is None or not env_value.strip():
        return None
    return float(env_value)


def _parse_success_rows(paths: list[Path]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = SUCCESS_STEP_RE.search(line)
            if match is None:
                continue
            rows.append(
                {
                    "step": float(match.group("step")),
                    "step_time": float(match.group("step_time")),
                    "throughput": float(match.group("throughput")),
                    "mfu": float(match.group("mfu")),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    log_files = _resolve_log_files(args)
    if not log_files:
        raise FileNotFoundError(
            "No log files provided. Use --log-file or --output-dir with output-dir/logs/*.log files."
        )

    samples = _parse_success_rows(log_files)
    if not samples:
        raise RuntimeError("No SUCCESS Step rows found in provided logs.")

    step_times = [row["step_time"] for row in samples]
    throughputs = [row["throughput"] for row in samples]
    raw_mfus = [row["mfu"] for row in samples]
    steps = [int(row["step"]) for row in samples]

    gpu_peak_tflops = _resolve_gpu_peak_tflops(args.gpu_peak_tflops)
    corrected_mfus: list[float] | None = None
    if gpu_peak_tflops is not None:
        if gpu_peak_tflops <= 0:
            raise ValueError("--gpu-peak-tflops (or MERA_GPU_PEAK_TFLOPS) must be > 0")
        correction_factor = 312.0 / gpu_peak_tflops
        corrected_mfus = [value * correction_factor for value in raw_mfus]

    summary: dict[str, Any] = {
        "log_files": [str(path) for path in log_files],
        "num_samples": len(samples),
        "step_min": min(steps),
        "step_max": max(steps),
        "throughput_tokens_per_s_median": _percentile(throughputs, 50),
        "throughput_tokens_per_s_p90": _percentile(throughputs, 90),
        "step_time_s_median": _percentile(step_times, 50),
        "step_time_s_p90": _percentile(step_times, 90),
        "mfu_raw_pct_median": _percentile(raw_mfus, 50),
        "mfu_raw_pct_p90": _percentile(raw_mfus, 90),
        "gpu_peak_tflops": gpu_peak_tflops,
    }
    if corrected_mfus is not None:
        summary["mfu_corrected_pct_median"] = _percentile(corrected_mfus, 50)
        summary["mfu_corrected_pct_p90"] = _percentile(corrected_mfus, 90)

    print(f"log_files={len(log_files)}")
    print(f"samples={summary['num_samples']}")
    print(f"step_min={summary['step_min']}")
    print(f"step_max={summary['step_max']}")
    print(f"throughput_tokens_per_s_median={summary['throughput_tokens_per_s_median']:.2f}")
    print(f"throughput_tokens_per_s_p90={summary['throughput_tokens_per_s_p90']:.2f}")
    print(f"step_time_s_median={summary['step_time_s_median']:.2f}")
    print(f"step_time_s_p90={summary['step_time_s_p90']:.2f}")
    print(f"mfu_raw_pct_median={summary['mfu_raw_pct_median']:.3f}")
    print(f"mfu_raw_pct_p90={summary['mfu_raw_pct_p90']:.3f}")
    if corrected_mfus is not None:
        print(f"gpu_peak_tflops={gpu_peak_tflops}")
        print(f"mfu_corrected_pct_median={summary['mfu_corrected_pct_median']:.3f}")
        print(f"mfu_corrected_pct_p90={summary['mfu_corrected_pct_p90']:.3f}")
    else:
        print("gpu_peak_tflops=unset")
        print("mfu_corrected_pct_median=unset")
        print("mfu_corrected_pct_p90=unset")

    if args.json_output is not None:
        output_path = args.json_output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json_output={output_path}")


if __name__ == "__main__":
    main()
