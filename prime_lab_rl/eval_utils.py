from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import OUTPUTS_ROOT
from .manifest import utc_now_iso


def slugify_model_for_results(model_name: str) -> str:
    return model_name.replace("/", "--")


def results_root(env_id: str, model_name: str) -> Path:
    env_module = env_id.split("/")[-1].replace("-", "_")
    env_local_dir = Path("environments") / env_module
    env_model_slug = f"{env_module}--{slugify_model_for_results(model_name)}"
    if env_local_dir.exists():
        return env_local_dir / "outputs" / "evals" / env_model_slug
    return OUTPUTS_ROOT / "evals" / env_model_slug


def find_latest_eval_run_dir(env_id: str, model_name: str, *, started_after: float | None = None) -> Path | None:
    root = results_root(env_id, model_name)
    if not root.exists():
        return None

    candidates = [path for path in root.iterdir() if path.is_dir()]
    if started_after is not None:
        candidates = [path for path in candidates if path.stat().st_mtime >= started_after]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


@dataclass
class EvalRunSummary:
    env_id: str
    model: str
    run_dir: Path
    avg_reward: float | None
    avg_error: float | None
    num_examples: int | None
    rollouts_per_example: int | None
    date: str | None
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "model": self.model,
            "run_dir": str(self.run_dir),
            "avg_reward": self.avg_reward,
            "avg_error": self.avg_error,
            "num_examples": self.num_examples,
            "rollouts_per_example": self.rollouts_per_example,
            "date": self.date,
            "metrics": self.metrics,
        }


def load_eval_summary(run_dir: Path) -> EvalRunSummary:
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    metrics = {
        key: value
        for key, value in metadata.items()
        if key.startswith("avg_") and key not in {"avg_reward", "avg_error"}
    }

    return EvalRunSummary(
        env_id=metadata["env_id"],
        model=metadata["model"],
        run_dir=run_dir,
        avg_reward=metadata.get("avg_reward"),
        avg_error=metadata.get("avg_error"),
        num_examples=metadata.get("num_examples"),
        rollouts_per_example=metadata.get("rollouts_per_example"),
        date=metadata.get("date"),
        metrics=metrics,
    )


def build_stage_summary(stage: str, summaries: list[EvalRunSummary]) -> dict[str, Any]:
    rewards = [summary.avg_reward for summary in summaries if summary.avg_reward is not None]
    avg_reward = (sum(rewards) / len(rewards)) if rewards else None
    return {
        "version": 1,
        "stage": stage,
        "created_at_utc": utc_now_iso(),
        "num_envs": len(summaries),
        "macro_avg_reward": avg_reward,
        "envs": [summary.to_dict() for summary in summaries],
    }


def write_stage_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_stage_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_stage_summaries(
    *,
    baseline: dict[str, Any] | None,
    current: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def index_envs(stage_summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        if not stage_summary:
            return {}
        return {entry["env_id"]: entry for entry in stage_summary.get("envs", [])}

    baseline_envs = index_envs(baseline)
    previous_envs = index_envs(previous)
    current_envs = index_envs(current)

    env_deltas: list[dict[str, Any]] = []
    for env_id, current_entry in current_envs.items():
        current_reward = current_entry.get("avg_reward")
        baseline_reward = baseline_envs.get(env_id, {}).get("avg_reward")
        previous_reward = previous_envs.get(env_id, {}).get("avg_reward")
        env_deltas.append(
            {
                "env_id": env_id,
                "current_avg_reward": current_reward,
                "delta_vs_baseline": _safe_delta(current_reward, baseline_reward),
                "delta_vs_previous": _safe_delta(current_reward, previous_reward),
            }
        )

    return {
        "version": 1,
        "created_at_utc": utc_now_iso(),
        "current_stage": current.get("stage"),
        "baseline_stage": baseline.get("stage") if baseline else None,
        "previous_stage": previous.get("stage") if previous else None,
        "macro_avg_reward": current.get("macro_avg_reward"),
        "macro_delta_vs_baseline": _safe_delta(
            current.get("macro_avg_reward"),
            baseline.get("macro_avg_reward") if baseline else None,
        ),
        "macro_delta_vs_previous": _safe_delta(
            current.get("macro_avg_reward"),
            previous.get("macro_avg_reward") if previous else None,
        ),
        "envs": env_deltas,
    }


def _safe_delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def infer_stage_run_name(stage: str, experiment: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment).strip("-")
    return f"{stage}-{normalized}" if normalized else stage
