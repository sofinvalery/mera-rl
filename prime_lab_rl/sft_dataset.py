from __future__ import annotations

import math
import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .constants import DEFAULT_SFT_MAX_SEQ_LEN, FAIR_SFT_SPLITS, FAIR_SFT_TASKS
from .manifest import utc_now_iso


MERA_DATA_ENV = "MERA_DATA_DIR"
MERA_HF_DATASET_ID = "MERA-evaluation/MERA"
MERA_CACHE_GLOBS = [
    "datasets--MERA-evaluation--MERA/snapshots/*/data",
    ".hf/hub/datasets--MERA-evaluation--MERA/snapshots/*/data",
]

_FORMATTER = string.Formatter()
RUTIE_CONTEXT_MODES = ("single_turn", "rolling")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_root(data_dir: str | None = None) -> Path:
    import os

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

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
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


def _load_hf_split(task: str, split: str, data_dir: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    split_candidates = [split]
    if split == "validation":
        split_candidates.append("dev")

    last_error: Exception | None = None
    for split_name in split_candidates:
        try:
            ds = load_dataset(MERA_HF_DATASET_ID, task, split=split_name, cache_dir=data_dir)
            return [ds[i] for i in range(len(ds))]
        except Exception as exc:
            last_error = exc

    assert last_error is not None
    raise FileNotFoundError(
        f"Split '{split}' not found for task '{task}' via local JSONL or Hugging Face dataset loader."
    ) from last_error


def load_task_split(task: str, split: str, data_dir: str | None = None) -> list[dict[str, Any]]:
    try:
        root = resolve_data_root(data_dir)
    except FileNotFoundError:
        return _load_hf_split(task, split, data_dir=data_dir)

    task_dir = root / task
    path = task_dir / f"{split}.jsonl"
    if not path.exists() and split == "validation":
        path = task_dir / "dev.jsonl"
    if path.exists():
        return _load_jsonl(path)

    return _load_hf_split(task, split, data_dir=data_dir)


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


def _normalize_text(text: str) -> str:
    return text.strip()


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return int(round(ordered[low] + (ordered[high] - ordered[low]) * frac))


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


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _build_loss_mask(
    *,
    prompt: list[dict[str, Any]],
    completion: list[dict[str, Any]],
    tokenizer: Any,
) -> list[bool]:
    messages = prompt + completion
    loss_mask: list[bool] = []
    prev_ids: list[int] = []
    prev_len = 0

    for i, message in enumerate(messages):
        role = message.get("role")
        add_generation_prompt = (
            role in {"user", "tool"}
            and i + 1 < len(messages)
            and messages[i + 1].get("role") == "assistant"
        )
        cur_ids = tokenizer.apply_chat_template(
            messages[: i + 1],
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
        if prev_ids != cur_ids[:prev_len]:
            raise ValueError(
                "Incremental tokenization mismatch while building SFT loss mask. "
                f"message_index={i}"
            )
        should_mask = role == "assistant"
        loss_mask.extend([should_mask] * (len(cur_ids) - prev_len))
        prev_ids = cur_ids
        prev_len = len(cur_ids)

    return loss_mask


def count_trainable_tokens_in_window(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
) -> tuple[int, int]:
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        raise ValueError("SFT row must include list-valued 'prompt' and 'completion'.")

    prompt_ids = tokenizer.apply_chat_template(prompt, return_dict=False)
    loss_mask = _build_loss_mask(prompt=prompt, completion=completion, tokenizer=tokenizer)
    full_ids = tokenizer.apply_chat_template(prompt + completion, return_dict=False)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in full_ids:
        loss_mask.append(True)

    aligned_loss_mask = loss_mask[1:] if loss_mask else []
    trainable_tokens = int(sum(aligned_loss_mask[:max_seq_len]))
    return trainable_tokens, len(prompt_ids)


def _build_rutie_rows(
    records: list[dict[str, Any]],
    *,
    context_mode: Literal["single_turn", "rolling"] = "single_turn",
) -> list[dict[str, Any] | None]:
    if context_mode not in RUTIE_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported rutie context mode: {context_mode}. "
            f"Expected one of: {', '.join(RUTIE_CONTEXT_MODES)}."
        )

    dialogs: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        dialog_id = int(rec["meta"]["dialog_id"])
        dialogs.setdefault(dialog_id, []).append(rec)

    rows: list[dict[str, Any] | None] = []
    for _dialog_id, items in sorted(dialogs.items(), key=lambda item: item[0]):
        items.sort(key=lambda item: int(item["meta"]["question_id"]))
        context = ""
        for rec in items:
            prompt_context = context if context_mode == "rolling" else ""
            prompt = format_prompt(rec["instruction"], rec["inputs"], context=prompt_context)
            row = _build_row(prompt, rec.get("outputs"), task="rutie")
            rows.append(row)

            if context_mode == "rolling":
                choice1 = str(rec["inputs"]["choice1"])
                choice2 = str(rec["inputs"]["choice2"])
                answer_text = choice1 if str(rec.get("outputs", "")).strip() == "1" else choice2
                context += (
                    f"{rec['inputs']['question']}\n1. {choice1}\n2. {choice2}\n"
                    f"Ответ: {answer_text}\n\n"
                )
    return rows


def resolve_sft_tasks(task_set: str, tasks: list[str] | None) -> list[str]:
    if tasks:
        selected = tasks
    elif task_set == "fair":
        selected = FAIR_SFT_TASKS
    else:
        raise ValueError(f"Unsupported task set: {task_set}")

    unknown = sorted(set(selected) - set(FAIR_SFT_SPLITS))
    if unknown:
        raise ValueError(f"Unknown SFT task(s): {', '.join(unknown)}")
    return selected


def _new_task_stats() -> dict[str, Any]:
    return {
        "raw_rows": 0,
        "kept_rows": 0,
        "dropped_empty": 0,
        "dropped_overlength": 0,
        "zero_trainable_before_filter": 0,
        "zero_trainable_after_filter": 0,
        "prompt_token_lengths": [],
    }


def _finalize_task_stats(task_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for task, stats in task_stats.items():
        prompt_token_lengths = [int(value) for value in stats.pop("prompt_token_lengths", [])]
        finalized[task] = {
            "raw_rows": int(stats.get("raw_rows", 0)),
            "kept_rows": int(stats.get("kept_rows", 0)),
            "dropped_empty": int(stats.get("dropped_empty", 0)),
            "dropped_overlength": int(stats.get("dropped_overlength", 0)),
            "zero_trainable_before_filter": int(stats.get("zero_trainable_before_filter", 0)),
            "zero_trainable_after_filter": int(stats.get("zero_trainable_after_filter", 0)),
            "prompt_tokens_p50": _percentile(prompt_token_lengths, 50),
            "prompt_tokens_p90": _percentile(prompt_token_lengths, 90),
            "prompt_tokens_p99": _percentile(prompt_token_lengths, 99),
        }
    return finalized


def build_sft_rows(
    data_dir: str | None,
    tasks: list[str],
    limit: int | None = None,
    *,
    rutie_context_mode: Literal["single_turn", "rolling"] = "single_turn",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    task_stats: dict[str, dict[str, Any]] = {task: _new_task_stats() for task in tasks}
    for task in tasks:
        split = FAIR_SFT_SPLITS[task]
        records = load_task_split(task, split, data_dir=data_dir)
        if limit is not None:
            records = records[:limit]

        if task == "rutie":
            for row in _build_rutie_rows(records, context_mode=rutie_context_mode):
                task_stats[task]["raw_rows"] += 1
                if row is None:
                    task_stats[task]["dropped_empty"] += 1
                    continue
                rows.append(row)
            continue

        for rec in records:
            task_stats[task]["raw_rows"] += 1
            prompt = format_prompt(rec["instruction"], rec["inputs"], context="")
            row = _build_row(prompt, rec.get("outputs"), task=task)
            if row is None:
                task_stats[task]["dropped_empty"] += 1
                continue
            rows.append(row)
    return rows, task_stats


@dataclass
class DatasetArtifacts:
    train_path: Path
    manifest_path: Path
    num_rows: int
    dataset_stats: dict[str, Any]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_sft_dataset_artifacts(
    output_dir: Path,
    *,
    data_dir: str | None,
    tasks: list[str],
    limit: int | None,
    base_model: str,
    max_seq_len: int = DEFAULT_SFT_MAX_SEQ_LEN,
    drop_overlength: bool = True,
    rutie_context_mode: Literal["single_turn", "rolling"] = "single_turn",
) -> DatasetArtifacts:
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be >= 1")

    dataset_dir = output_dir / "dataset"
    train_path = dataset_dir / "train.jsonl"
    manifest_path = dataset_dir / "manifest.json"

    rows, task_stats = build_sft_rows(
        data_dir=data_dir,
        tasks=tasks,
        limit=limit,
        rutie_context_mode=rutie_context_mode,
    )
    if not rows:
        raise ValueError("SFT dataset is empty after preprocessing. Check task splits and --limit.")

    tokenizer = _load_tokenizer(base_model)
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        task = str(row.get("task", "<unknown>"))
        task_entry = task_stats.setdefault(task, _new_task_stats())
        trainable_tokens, prompt_tokens = count_trainable_tokens_in_window(
            row=row,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        task_entry["prompt_token_lengths"].append(prompt_tokens)
        if trainable_tokens == 0:
            task_entry["zero_trainable_before_filter"] += 1
            if drop_overlength:
                task_entry["dropped_overlength"] += 1
                continue
            task_entry["zero_trainable_after_filter"] += 1
        task_entry["kept_rows"] += 1
        filtered_rows.append(row)

    if not filtered_rows:
        raise ValueError(
            "SFT dataset is empty after dropping overlength/untrainable rows. "
            "Try increasing --max-seq-len or disabling --drop-overlength."
        )

    _write_jsonl(train_path, filtered_rows)
    finalized_task_stats = _finalize_task_stats(task_stats)
    dataset_stats = {
        "max_seq_len": max_seq_len,
        "drop_overlength": drop_overlength,
        "rutie_context_mode": rutie_context_mode,
        "raw_rows": sum(entry["raw_rows"] for entry in finalized_task_stats.values()),
        "kept_rows": sum(entry["kept_rows"] for entry in finalized_task_stats.values()),
        "dropped_empty": sum(entry["dropped_empty"] for entry in finalized_task_stats.values()),
        "dropped_overlength": sum(entry["dropped_overlength"] for entry in finalized_task_stats.values()),
        "zero_trainable_before_filter": sum(
            entry["zero_trainable_before_filter"] for entry in finalized_task_stats.values()
        ),
        "zero_trainable_after_filter": sum(
            entry["zero_trainable_after_filter"] for entry in finalized_task_stats.values()
        ),
        "task_stats": finalized_task_stats,
    }

    manifest = {
        "version": 1,
        "created_at_utc": utc_now_iso(),
        "train_file": str(train_path.resolve()),
        "task_splits": {task: FAIR_SFT_SPLITS[task] for task in tasks},
        "tasks": tasks,
        "num_train_rows": len(filtered_rows),
        "data_dir": data_dir,
        "base_model": base_model,
        "dataset_stats": dataset_stats,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    return DatasetArtifacts(
        train_path=train_path,
        manifest_path=manifest_path,
        num_rows=len(filtered_rows),
        dataset_stats=dataset_stats,
    )
