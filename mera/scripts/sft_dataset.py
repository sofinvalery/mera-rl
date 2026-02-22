from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import string
from typing import Any

MERA_DATA_ENV = "MERA_DATA_DIR"
MERA_CACHE_GLOBS = [
    "datasets--MERA-evaluation--MERA/snapshots/*/data",
    ".hf/hub/datasets--MERA-evaluation--MERA/snapshots/*/data",
]

ALL_SFT_TASKS = [
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

FAIR_SFT_TASKS = [
    "chegeka",
    "lcs",
    "mamuramu",
    "mathlogicqa",
    "multiq",
    "parus",
    "rcb",
    "rumodar",
    "rumultiar",
    "ruopenbookqa",
    "rutie",
    "ruworldtree",
    "rwsd",
    "use",
]

SFT_SPLITS = {
    "bps": "train",
    "chegeka": "train",
    "lcs": "public_test",
    "mamuramu": "train",
    "mathlogicqa": "train",
    "multiq": "train",
    "parus": "train",
    "rcb": "train",
    "rudetox": "train",
    "rummlu": "train",
    "rumodar": "public_test",
    "rumultiar": "train",
    "ruopenbookqa": "train",
    "rutie": "train",
    "ruworldtree": "train",
    "rwsd": "train",
    "simplear": "train",
    "use": "train",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_root(data_dir: str | None = None) -> Path:
    if data_dir:
        path = Path(data_dir).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"MERA data dir not found: {path}")

    env_dir = os.getenv(MERA_DATA_ENV)
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"MERA data dir not found: {path}")

    candidates: list[Path] = []
    bases = [_repo_root(), Path.cwd()]

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        bases.append(Path(hf_home).expanduser())
    bases.append(Path.home() / ".cache" / "huggingface")

    for base in bases:
        for pattern in MERA_CACHE_GLOBS:
            candidates.extend(base.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            "MERA dataset not found. Set MERA_DATA_DIR or place the HF cache under the repo."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_task_split(task: str, split: str, data_dir: str | None = None) -> list[dict[str, Any]]:
    root = resolve_data_root(data_dir)
    task_dir = root / task
    path = task_dir / f"{split}.jsonl"
    if not path.exists() and split == "validation":
        path = task_dir / "dev.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split '{split}' not found for task '{task}' at {task_dir}")
    return _load_jsonl(path)


_FORMATTER = string.Formatter()


def format_prompt(instruction: str, inputs: Any, context: str = "") -> str:
    fields = [field for _, field, _, _ in _FORMATTER.parse(instruction) if field is not None]
    named_fields = [field for field in fields if field and not field.isdigit()]
    positional_fields = [field for field in fields if field == "" or (field and field.isdigit())]

    mapping: dict[str, Any] = {}
    if isinstance(inputs, dict):
        mapping.update(inputs)
    else:
        for name in named_fields:
            mapping[name] = inputs

    if "context" in fields and "context" not in mapping:
        mapping["context"] = context

    for name in named_fields:
        mapping.setdefault(name, "")

    if positional_fields:
        pos_val = "" if isinstance(inputs, dict) else inputs
        pos_args = [pos_val for _ in positional_fields]
        return instruction.format(*pos_args, **mapping)

    return instruction.format(**mapping)


@dataclass
class DatasetArtifacts:
    train_path: Path
    manifest_path: Path
    num_rows: int


def resolve_sft_tasks(task_set: str, tasks: list[str] | None) -> list[str]:
    if tasks:
        selected = tasks
    elif task_set == "all":
        selected = ALL_SFT_TASKS
    else:
        selected = FAIR_SFT_TASKS

    unknown = sorted(set(selected) - set(SFT_SPLITS))
    if unknown:
        raise ValueError(f"Unknown SFT task(s): {', '.join(unknown)}")
    return selected


def _normalize_text(text: str) -> str:
    # Keep deterministic formatting and avoid leading/trailing whitespace-only samples.
    return text.strip()


def normalize_answer(answer: Any) -> str:
    if isinstance(answer, str):
        return _normalize_text(answer)
    if answer is None:
        return ""
    if isinstance(answer, (dict, list)):
        return _normalize_text(json.dumps(answer, ensure_ascii=False, sort_keys=True))
    return _normalize_text(str(answer))


def _build_row(prompt: str, answer: Any, task: str) -> dict[str, Any] | None:
    prompt_text = _normalize_text(prompt)
    answer_text = normalize_answer(answer)
    if not prompt_text or not answer_text:
        return None
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "completion": [{"role": "assistant", "content": answer_text}],
        "task": task,
    }


def _build_rutie_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dialogs: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        dialog_id = int(rec["meta"]["dialog_id"])
        dialogs.setdefault(dialog_id, []).append(rec)

    rows: list[dict[str, Any]] = []
    for _dialog_id, items in sorted(dialogs.items(), key=lambda x: x[0]):
        items.sort(key=lambda x: int(x["meta"]["question_id"]))
        context = ""
        for rec in items:
            prompt = format_prompt(rec["instruction"], rec["inputs"], context=context)
            row = _build_row(prompt, rec.get("outputs"), task="rutie")
            if row is not None:
                rows.append(row)

            choice1 = str(rec["inputs"]["choice1"])
            choice2 = str(rec["inputs"]["choice2"])
            answer_text = choice1 if str(rec.get("outputs", "")).strip() == "1" else choice2
            context += (
                f"{rec['inputs']['question']}\n1. {choice1}\n2. {choice2}\n"
                f"Ответ: {answer_text}\n\n"
            )
    return rows


def build_sft_rows(data_dir: str | None, tasks: list[str], limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        split = SFT_SPLITS[task]
        records = load_task_split(task, split, data_dir=data_dir)
        if limit is not None:
            records = records[:limit]

        if task == "rutie":
            rows.extend(_build_rutie_rows(records))
            continue

        for rec in records:
            prompt = format_prompt(rec["instruction"], rec["inputs"], context="")
            row = _build_row(prompt, rec.get("outputs"), task=task)
            if row is not None:
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_sft_dataset_artifacts(
    output_dir: Path,
    data_dir: str | None,
    tasks: list[str],
    limit: int | None,
) -> DatasetArtifacts:
    dataset_dir = output_dir / "dataset"
    train_path = dataset_dir / "train.jsonl"
    manifest_path = dataset_dir / "manifest.json"

    rows = build_sft_rows(data_dir=data_dir, tasks=tasks, limit=limit)
    if not rows:
        raise ValueError("SFT dataset is empty after preprocessing. Check task splits and --limit.")

    _write_jsonl(train_path, rows)

    manifest = {
        "version": 1,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "train_file": str(train_path.resolve()),
        "task_splits": {task: SFT_SPLITS[task] for task in tasks},
        "tasks": tasks,
        "num_train_rows": len(rows),
        "data_dir": data_dir,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return DatasetArtifacts(train_path=train_path, manifest_path=manifest_path, num_rows=len(rows))
