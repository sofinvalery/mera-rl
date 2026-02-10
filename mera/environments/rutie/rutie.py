from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional, Tuple

import verifiers as vf
from datasets import Dataset, load_dataset

DATASET_ID = "MERA-evaluation/MERA"
TASK_ID = "rutie"

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


def _group_dialogs(records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    dialogs: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        dialog_id = int(rec["meta"]["dialog_id"])
        dialogs.setdefault(dialog_id, []).append(rec)

    grouped = []
    for dialog_id, items in sorted(dialogs.items(), key=lambda x: x[0]):
        items.sort(key=lambda x: int(x["meta"]["question_id"]))
        grouped.append(items)
    return grouped


def _extract_12(text: str) -> str:
    match = re.search(r"\b([12])\b", text.strip())
    return match.group(1) if match else ""


def _first_prompt_messages(prompt: str, system_prompt: Optional[str] = None) -> vf.Messages:
    messages: vf.Messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


class RuTiEEnv(vf.MultiTurnEnv):
    def __init__(self, dialogs: List[List[Dict[str, Any]]], system_prompt: str | None = None, **kwargs):
        ds = Dataset.from_list([{"prompt": [], "dialog": dialog} for dialog in dialogs])
        super().__init__(dataset=ds, eval_dataset=ds, system_prompt=system_prompt, **kwargs)
        self._parser = vf.MaybeThinkParser(_extract_12)

    async def setup_state(self, state: vf.State, **_kwargs: Any) -> vf.State:
        dialog = state["dialog"]
        state["turn_index"] = 0
        state["correct"] = 0
        state["total"] = len(dialog)
        state["context"] = ""

        first = dialog[0]
        prompt = format_prompt(first["instruction"], first["inputs"], context="")
        state["prompt"] = _first_prompt_messages(prompt, system_prompt=self.system_prompt)
        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **_kwargs: Any
    ) -> Tuple[vf.Messages, vf.State]:
        if not messages or messages[-1]["role"] != "assistant":
            return [], state

        dialog: List[Dict[str, Any]] = state["dialog"]
        idx = int(state["turn_index"])
        current = dialog[idx]

        parsed = self._parser.parse_answer(messages) or ""
        gold = str(current.get("outputs", "")).strip()
        if parsed and gold and parsed == gold:
            state["correct"] += 1

        answer_text = ""
        if parsed == "1":
            answer_text = str(current["inputs"].get("choice1", ""))
        elif parsed == "2":
            answer_text = str(current["inputs"].get("choice2", ""))

        question = str(current["inputs"].get("question", ""))
        choice1 = str(current["inputs"].get("choice1", ""))
        choice2 = str(current["inputs"].get("choice2", ""))

        context = str(state.get("context", ""))
        context += (
            f"{question}\n1. {choice1}\n2. {choice2}\n\u041e\u0442\u0432\u0435\u0442: {answer_text}\n\n"
        )
        state["context"] = context
        state["turn_index"] = idx + 1

        if state["turn_index"] >= state["total"]:
            return [], state

        nxt = dialog[state["turn_index"]]
        prompt = format_prompt(nxt["instruction"], nxt["inputs"], context=context)
        return [{"role": "user", "content": prompt}], state

    async def is_completed(self, state: vf.State, **_kwargs: Any) -> bool:
        return int(state.get("turn_index", 0)) >= int(state.get("total", 0))


def load_environment(
    split: str = "train",
    system_prompt: str | None = None,
    cache_dir: str | None = None,
    max_eval_examples: int | None = None,
    **_kwargs: Any,
) -> vf.Environment:
    records_ds = load_dataset(DATASET_ID, TASK_ID, split=split, cache_dir=cache_dir)
    records: List[Dict[str, Any]] = [records_ds[i] for i in range(len(records_ds))]
    dialogs = _group_dialogs(records)
    if max_eval_examples is not None:
        dialogs = dialogs[: min(int(max_eval_examples), len(dialogs))]

    def reward_fn(state: vf.State, **_kw: Any) -> float:
        total = int(state.get("total", 0)) or 1
        correct = int(state.get("correct", 0))
        return float(correct) / float(total)

    rubric = vf.Rubric()
    rubric.add_reward_func(reward_fn)

    return RuTiEEnv(dialogs=dialogs, system_prompt=system_prompt, rubric=rubric)
