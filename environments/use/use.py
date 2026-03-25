from __future__ import annotations

import json
import re
import string
from typing import Any, Dict, Optional

import verifiers as vf
from datasets import Dataset, load_dataset

DATASET_ID = "MERA-evaluation/MERA"
TASK_ID = "use"

_FORMATTER = string.Formatter()


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
            "prompt": [{"role": "user", "content": question}],
            "answer": str(x.get("outputs", "")),
            "info": json.dumps(
                {"inputs": x.get("inputs", {}), "meta": x.get("meta", {})},
                ensure_ascii=False,
            ),
        }

    return raw.map(to_example, remove_columns=raw.column_names)


def use_multiple_choice_score(answer: str, prediction: str, is_task16: bool = False) -> int:
    pred = prediction.split(",")
    ans = answer.split(",")
    if is_task16:
        while len(pred) < len(ans):
            pred.append("-1")
        return max(0, len(set(ans) & set(pred)) - len(pred) + len(ans))
    ans_set = set(ans)
    pred_set = set(pred)
    return int(len(ans_set & pred_set) == len(ans_set) == len(pred_set))


def use_matching_score(answer: str, prediction: str) -> int:
    pred = prediction.split(",")
    ans = answer.split(",")
    if len(ans) != len(pred):
        return 0
    score = 0
    for idx, num in enumerate(ans):
        if num == pred[idx]:
            score += 1
    return score


def use_text_score(answer: str, prediction: str) -> int:
    pred = re.sub(r"[\d+\W+]", "", prediction).lower()
    ans = answer.split(",")
    return 1 if pred in ans else 0


def use_example_score(answer: str, prediction: str, task_type: str, id_task: str) -> int:
    if task_type == "matching":
        return use_matching_score(answer, prediction)
    if task_type == "text":
        return use_text_score(answer, prediction)
    is_task16 = id_task == "16"
    return use_multiple_choice_score(answer, prediction, is_task16)


def parse_use_answer(completion: str) -> str:
    text = completion.strip()
    match = re.search(r"\d+(?:,\d+)*", text)
    if match:
        return match.group(0)
    match = re.search(r"[A-Za-z\u0400-\u04FF]+", text)
    if match:
        return match.group(0).lower()
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

    parser = vf.MaybeThinkParser(lambda x: x.strip())

    def reward_fn(
        parser: vf.Parser,
        completion: vf.Messages,
        answer: Any,
        info: Any,
        **_kw: Any,
    ) -> float:
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            return 0.0
        pred_raw = parser.parse_answer(completion) or ""
        pred = parse_use_answer(pred_raw)
        if not pred:
            return 0.0
        info_obj: Dict[str, Any]
        if isinstance(info, str):
            try:
                parsed_info = json.loads(info)
            except json.JSONDecodeError:
                parsed_info = {}
            info_obj = parsed_info if isinstance(parsed_info, dict) else {}
        elif isinstance(info, dict):
            info_obj = info
        else:
            info_obj = {}
        raw_meta = info_obj.get("meta", {})
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        score = use_example_score(
            str(answer),
            pred,
            str(meta.get("type", "")),
            str(meta.get("id_task", "")),
        )
        max_score = int(meta.get("score", 1)) or 1
        return float(score) / float(max_score)

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(reward_fn)

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
