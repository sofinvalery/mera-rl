#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.constants import (
    DEFAULT_EVAL_NUM_EXAMPLES,
    DEFAULT_EVAL_ROLLOUTS_PER_EXAMPLE,
    FAIR_EVAL_TASKS,
    RUNS_ROOT,
    SMOKE_EVAL_TASKS,
)
from prime_lab_rl.eval_utils import (
    build_stage_summary,
    compare_stage_summaries,
    find_latest_eval_run_dir,
    infer_stage_run_name,
    load_eval_summary,
    load_stage_summary,
    write_stage_summary,
)
from prime_lab_rl.manifest import ensure_manifest, update_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MERA eval stage with prime eval.")
    parser.add_argument("--stage", required=True, choices=["baseline", "intermediate", "final", "smoke"])
    parser.add_argument("--experiment", default="manual")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--task-set", choices=["fair", "smoke"], default="fair")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--endpoint-id", default=None, help="Existing endpoint alias in endpoints.toml.")
    parser.add_argument("--endpoints-path", type=Path, default=REPO_ROOT / "configs" / "endpoints.toml")
    parser.add_argument("--model-name", default=None, help="Required when rendering a temporary endpoint.")
    parser.add_argument("--endpoint-url", default=None, help="OpenAI-compatible or Anthropic endpoint URL.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable name containing the API key.")
    parser.add_argument(
        "--endpoint-type",
        choices=["openai_chat_completions", "anthropic_messages"],
        default="openai_chat_completions",
    )
    parser.add_argument("--num-examples", type=int, default=DEFAULT_EVAL_NUM_EXAMPLES)
    parser.add_argument("--rollouts-per-example", type=int, default=DEFAULT_EVAL_ROLLOUTS_PER_EXAMPLE)
    parser.add_argument("--upload", action="store_true", help="Allow Prime to upload eval results.")
    parser.add_argument("--skip-install-envs", action="store_true", help="Skip prime env install before eval.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _render_temp_endpoints_file(
    *,
    path: Path,
    endpoint_id: str,
    model_name: str,
    endpoint_url: str,
    api_key_env: str,
    endpoint_type: str,
) -> None:
    text = (
        "[[endpoint]]\n"
        f'endpoint_id = "{endpoint_id}"\n'
        f'model = "{model_name}"\n'
        f'url = "{endpoint_url}"\n'
        f'key = "{api_key_env}"\n'
        f'type = "{endpoint_type}"\n'
    )
    path.write_text(text, encoding="utf-8")


def _resolve_model_name(endpoints_path: Path, endpoint_id: str) -> str:
    with endpoints_path.open("rb") as handle:
        data = tomllib.load(handle)
    for endpoint in data.get("endpoint", []):
        if endpoint.get("endpoint_id") == endpoint_id:
            model_name = endpoint.get("model")
            if not model_name:
                raise ValueError(f"Endpoint '{endpoint_id}' has no model field in {endpoints_path}")
            return str(model_name)
    raise ValueError(f"Endpoint '{endpoint_id}' not found in {endpoints_path}")


def _render_eval_config(
    *,
    path: Path,
    endpoint_id: str,
    endpoints_path: Path,
    tasks: list[str],
    num_examples: int,
    rollouts_per_example: int,
) -> None:
    rel_endpoints_path = os.path.relpath(endpoints_path.resolve(), path.parent.resolve())
    lines = [
        f'endpoints_path = "{rel_endpoints_path}"',
        f'endpoint_id = "{endpoint_id}"',
        "save_results = true",
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
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _install_envs(tasks: list[str]) -> None:
    cmd = ["prime", "env", "install", *tasks, "-p", str(REPO_ROOT / "environments")]
    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    args = parse_args()
    run_root = (args.run_root or (RUNS_ROOT / args.experiment)).expanduser().resolve()
    stage_dir = run_root / "evaluations" / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    endpoint_id = args.endpoint_id or f"mera-{args.stage}"
    endpoints_path = args.endpoints_path.expanduser().resolve()
    temp_endpoints_path: Path | None = None
    if args.endpoint_url:
        if not args.model_name or not args.api_key_env:
            raise ValueError("--model-name and --api-key-env are required with --endpoint-url")
        temp_endpoints_path = stage_dir / "endpoints.toml"
        _render_temp_endpoints_file(
            path=temp_endpoints_path,
            endpoint_id=endpoint_id,
            model_name=args.model_name,
            endpoint_url=args.endpoint_url,
            api_key_env=args.api_key_env,
            endpoint_type=args.endpoint_type,
        )
        endpoints_path = temp_endpoints_path

    if not endpoints_path.exists():
        raise FileNotFoundError(f"Endpoints file not found: {endpoints_path}")

    tasks = args.tasks or (FAIR_EVAL_TASKS if args.task_set == "fair" else SMOKE_EVAL_TASKS)
    config_path = stage_dir / "eval.toml"
    _render_eval_config(
        path=config_path,
        endpoint_id=endpoint_id,
        endpoints_path=endpoints_path,
        tasks=tasks,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
    )

    model_name = args.model_name or _resolve_model_name(endpoints_path, endpoint_id)
    run_name = infer_stage_run_name(args.stage, args.experiment)
    manifest_path = args.manifest.expanduser().resolve() if args.manifest is not None else None
    if manifest_path is not None:
        ensure_manifest(manifest_path, experiment=args.experiment, base_model=model_name)

    cmd = ["prime", "eval", "run", str(config_path)]
    if args.upload:
        pass
    else:
        cmd.append("--skip-upload")

    print(f"stage={args.stage}")
    print(f"config={config_path}")
    print(f"endpoint_id={endpoint_id}")
    print(f"model_name={model_name}")
    print(f"tasks={','.join(tasks)}")

    if args.dry_run:
        return

    if not args.skip_install_envs:
        _install_envs(tasks)

    started_at = time.time() - 1.0
    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    summaries = []
    run_dirs: dict[str, str] = {}
    for task in tasks:
        run_dir = find_latest_eval_run_dir(task, model_name, started_after=started_at)
        if run_dir is None:
            raise FileNotFoundError(f"Could not find saved eval outputs for task '{task}' and model '{model_name}'")
        run_dirs[task] = str(run_dir.resolve())
        summaries.append(load_eval_summary(run_dir))

    summary = build_stage_summary(args.stage, summaries)
    summary_path = stage_dir / "summary.json"
    write_stage_summary(summary_path, summary)

    comparison_path = stage_dir / "comparison.json"
    baseline_summary = None
    previous_summary = None
    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        baseline_path = manifest.get("evaluations", {}).get("baseline", {}).get("summary_path")
        previous_key = "baseline" if args.stage == "intermediate" else "intermediate" if args.stage == "final" else None
        previous_path = manifest.get("evaluations", {}).get(previous_key or "", {}).get("summary_path")
        if baseline_path and Path(baseline_path).exists():
            baseline_summary = load_stage_summary(Path(baseline_path))
        if previous_path and Path(previous_path).exists():
            previous_summary = load_stage_summary(Path(previous_path))
        if args.stage == "baseline":
            previous_summary = None
            baseline_summary = None

    comparison = compare_stage_summaries(
        baseline=baseline_summary,
        current=summary,
        previous=previous_summary,
    )
    write_stage_summary(comparison_path, comparison)

    if manifest_path is not None:
        update_manifest(
            manifest_path,
            {
                "evaluations": {
                    args.stage: {
                        "run_name": run_name,
                        "config_path": str(config_path),
                        "endpoints_path": str(endpoints_path),
                        "endpoint_id": endpoint_id,
                        "model_name": model_name,
                        "tasks": tasks,
                        "summary_path": str(summary_path),
                        "comparison_path": str(comparison_path),
                        "run_dirs": run_dirs,
                    }
                }
            },
        )

    print(f"summary_path={summary_path}")
    print(f"comparison_path={comparison_path}")


if __name__ == "__main__":
    main()
