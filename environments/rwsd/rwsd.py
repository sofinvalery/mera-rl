from __future__ import annotations

import re
import string
from typing import Any, Dict, Optional

import verifiers as vf
from datasets import Dataset, load_dataset

DATASET_ID = "MERA-evaluation/MERA"
TASK_ID = "rwsd"

_FORMATTER = string.Formatter()

_YES = {"YES", "Y", "TRUE", "1", "\u0414\u0410"}
_NO = {"NO", "N", "FALSE", "0", "\u041d\u0415\u0422"}


def format_prompt(instruction: str, inputs: Any, context: str = "") -> str:
    fields = [field for _, field, _, _ in _FORMATTER.parse(instruction) if field is not None]
    named_fields = [field for field in fields if field and not field.isdigit()]
    positional_fields = [field for field in fields if field == "" or (field and field.isdigit())]

    mapping: Dict[str, Any] = {}
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


def _build_dataset(split: str, cache_dir: Optional[str] = None) -> Dataset:
    raw = load_dataset(DATASET_ID, TASK_ID, split=split, cache_dir=cache_dir)

    def to_example(x: Dict[str, Any]) -> Dict[str, Any]:
        question = format_prompt(x["instruction"], x["inputs"], context="")
        return {
            "question": question,
            "answer": x.get("outputs", ""),
            "meta": x.get("meta", {}),
            "inputs": x.get("inputs", {}),
        }

    return raw.map(to_example, remove_columns=raw.column_names)


def _extract_yes_no(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z\u0400-\u04FF0-9]", " ", text, flags=re.UNICODE)
    tokens = [tok for tok in normalized.strip().upper().split() if tok]
    for token in tokens:
        if token in _YES:
            return "YES"
        if token in _NO:
            return "NO"
    return ""


def _normalize_yes_no(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"YES", "\u0414\u0410", "1"}:
        return "YES"
    if text in {"NO", "\u041d\u0415\u0422", "0"}:
        return "NO"
    return text


def load_environment(
    split: str = "train",
    system_prompt: str | None = None,
    cache_dir: str | None = None,
    max_eval_examples: int | None = None,
    **_kwargs,
) -> vf.Environment:
    train_ds = _build_dataset("train", cache_dir=cache_dir)
    eval_ds = _build_dataset(split, cache_dir=cache_dir)
    if max_eval_examples is not None:
        eval_ds = eval_ds.select(range(min(int(max_eval_examples), len(eval_ds))))

    parser = vf.MaybeThinkParser(_extract_yes_no)

    def reward_fn(parser_obj: vf.Parser, completion: vf.Messages, answer: Any, **_kw: Any) -> float:
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            return 0.0
        pred = parser_obj.parse_answer(completion) or ""
        if not pred:
            return 0.0
        return 1.0 if _normalize_yes_no(pred) == _normalize_yes_no(answer) else 0.0

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(reward_fn)

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
