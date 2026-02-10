from __future__ import annotations

import json
import os
import pickle
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import verifiers as vf
from datasets import Dataset
from verifiers.parsers.parser import Parser
from verifiers.types import Messages

MERA_DATA_ENV = "MERA_DATA_DIR"
MERA_REPO_ENV = "MERA_REPO_DIR"
MERA_CACHE_GLOBS = [
    "datasets--MERA-evaluation--MERA/snapshots/*/data",
    ".hf/hub/datasets--MERA-evaluation--MERA/snapshots/*/data",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _looks_like_mera_repo(path: Path) -> bool:
    if not path.exists():
        return False
    return (path / "modules" / "scoring").exists() or (path / "humanbenchmarks").exists()


def find_mera_repo_root(override: Optional[str] = None) -> Optional[Path]:
    if override:
        path = Path(override).expanduser().resolve()
        return path if _looks_like_mera_repo(path) else None

    env_dir = os.getenv(MERA_REPO_ENV)
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        return path if _looks_like_mera_repo(path) else None

    repo_root = _repo_root()
    candidates = [repo_root / "_deps" / "MERA"]
    for path in candidates:
        if _looks_like_mera_repo(path):
            return path.resolve()
    return None


def resolve_mera_repo_root(override: Optional[str] = None) -> Path:
    path = find_mera_repo_root(override)
    if path is None:
        repo_root = _repo_root()
        raise FileNotFoundError(
            "MERA repo not found. Expected one of: "
            f"{repo_root / '_deps' / 'MERA'}. "
            f"You can also set {MERA_REPO_ENV}."
        )
    return path


def resolve_data_root(data_dir: Optional[str] = None) -> Path:
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

    candidates = []
    bases = [_repo_root(), Path.cwd()]

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        bases.append(Path(hf_home).expanduser())
    bases.append(Path.home() / ".cache" / "huggingface")

    for base in bases:
        for pattern in MERA_CACHE_GLOBS:
            for match in base.glob(pattern):
                candidates.append(match)
    if not candidates:
        raise FileNotFoundError(
            "MERA dataset not found. Set MERA_DATA_DIR or place the HF cache under the repo."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_task_split(task: str, split: str, data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
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


def build_messages(prompt: str, system_prompt: Optional[str] = None) -> Messages:
    messages: Messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


class BaseParser(Parser):
    def parse_answer(self, completion: Messages) -> Optional[str]:
        if isinstance(completion, list):
            content = completion[-1].get("content", "") if completion else ""
        else:
            content = str(completion)
        return self.parse(content)


class Binary01Parser(BaseParser):
    def parse(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"\b([01])\b", text.strip())
        return match.group(1) if match else None


class Choice12Parser(BaseParser):
    def parse(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"\b([12])\b", text.strip())
        return match.group(1) if match else None


class Choice123Parser(BaseParser):
    def parse(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"\b([123])\b", text.strip())
        return match.group(1) if match else None


class ChoiceABCDParser(BaseParser):
    def parse(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"\b([ABCD])\b", text.strip().upper())
        return match.group(1) if match else None


class YesNoParser(BaseParser):
    _YES = {"YES", "Y", "TRUE", "1", "\u0414\u0410"}
    _NO = {"NO", "N", "FALSE", "0", "\u041d\u0415\u0422"}

    def parse(self, text: str) -> Optional[str]:
        if not text:
            return None
        normalized = re.sub(r"[^A-Za-z\u0400-\u04FF0-9]", " ", text, flags=re.UNICODE)
        tokens = [tok for tok in normalized.strip().upper().split() if tok]
        for token in tokens:
            if token in self._YES:
                return "YES"
            if token in self._NO:
                return "NO"
        return None


class NumericParser(BaseParser):
    def parse(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"[-+]?\d+(?:[\.,]\d+)?", text.replace(" ", ""))
        if not match:
            return None
        value = match.group(0).replace(",", ".")
        return value


class FreeTextParser(BaseParser):
    def parse(self, text: str) -> Optional[str]:
        if not text:
            return None
        return text.strip()


def _normalize_choice(value: Any) -> str:
    return str(value).strip().upper()


def _normalize_binary(value: Any) -> str:
    return str(value).strip()


def _normalize_numeric(value: Any) -> str:
    text = str(value).strip().replace(",", ".")
    try:
        num = float(text)
    except Exception:
        return text
    if num.is_integer():
        return str(int(num))
    return str(num).rstrip("0").rstrip(".")


def _normalize_yes_no(value: Any) -> str:
    value = str(value).strip().upper()
    if value in {"YES", "\u0414\u0410", "1"}:
        return "YES"
    if value in {"NO", "\u041d\u0415\u0422", "0"}:
        return "NO"
    return value


def _normalize_squad(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def squad_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_squad(prediction).split()
    gold_tokens = _normalize_squad(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    num_same = 0
    for tok in gold_tokens:
        if common.get(tok, 0):
            num_same += 1
            common[tok] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_em(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_squad(prediction) == _normalize_squad(ground_truth) else 0.0


def max_over_ground_truths(
    metric_fn: Callable[[str, str], float], prediction: str, ground_truths: Sequence[str]
) -> float:
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores) if scores else 0.0


def exact_match_reward(
    parser: BaseParser, completion: Messages, answer: Any, normalize: Callable[[Any], str]
) -> float:
    parsed = parser.parse_answer(completion)
    if parsed is None:
        return 0.0
    return 1.0 if normalize(parsed) == normalize(answer) else 0.0


def f1_em_reward(parser: BaseParser, completion: Messages, answer: str) -> float:
    parsed = parser.parse_answer(completion)
    if not parsed:
        return 0.0
    answers = [a.strip() for a in answer.split(";") if a.strip()]
    f1 = max_over_ground_truths(squad_f1, parsed, answers)
    em = max_over_ground_truths(squad_em, parsed, answers)
    return (f1 + em) / 2.0


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


class RuDetoxScorer:
    def __init__(self, fast: bool = False, device: Optional[str] = None):
        self.fast = fast
        self.device = device
        self._loaded = False

    def _load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import numpy as np
        import torch

        if self._loaded:
            return
        self._np = np
        self._torch = torch
        self._style_model = AutoModelForSequenceClassification.from_pretrained(
            "IlyaGusev/rubertconv_toxic_clf"
        )
        self._style_model.to(self._device())
        self._style_tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/rubertconv_toxic_clf")

        mera_repo = find_mera_repo_root()
        calibration_candidates = []
        if mera_repo is not None:
            calibration_candidates = [
                mera_repo
                / "humanbenchmarks"
                / "ruDetox"
                / "score_calibrations_ru.pkl",
                mera_repo
                / "lm-evaluation-harness"
                / "lm_eval"
                / "tasks"
                / "rudetox"
                / "score_calibrations_ru.pkl",
            ]
        calibration_path = next((path for path in calibration_candidates if path.exists()), None)
        if calibration_path:
            with calibration_path.open("rb") as handle:
                calibration = pickle.load(handle)
            self._style_calibration = lambda x: calibration.predict(x[:, self._np.newaxis])
        else:
            self._style_calibration = lambda x: x

        if self.fast:
            self._loaded = True
            return

        self._meaning_model = AutoModelForSequenceClassification.from_pretrained(
            "s-nlp/rubert-base-cased-conversational-paraphrase-v1"
        )
        self._meaning_model.to(self._device())
        self._meaning_tokenizer = AutoTokenizer.from_pretrained(
            "s-nlp/rubert-base-cased-conversational-paraphrase-v1"
        )

        self._cola_model = AutoModelForSequenceClassification.from_pretrained(
            "s-nlp/ruRoberta-large-RuCoLa-v1"
        )
        self._cola_model.to(self._device())
        self._cola_tokenizer = AutoTokenizer.from_pretrained("s-nlp/ruRoberta-large-RuCoLa-v1")

        if calibration_path:
            with calibration_path.open("rb") as handle:
                calibration = pickle.load(handle)
            self._meaning_calibration = lambda x: calibration.predict(x[:, self._np.newaxis])
            self._fluency_calibration = lambda x: calibration.predict(x[:, self._np.newaxis])
        else:
            self._meaning_calibration = lambda x: x
            self._fluency_calibration = lambda x: x

        self._loaded = True

    def _device(self) -> str:
        if self.device:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _classify(self, model, tokenizer, texts, second_texts=None, target_label=None):
        inputs = [texts]
        if second_texts is not None:
            inputs.append(second_texts)
        data = tokenizer(
            *inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with self._torch.no_grad():
            logits = model(**data).logits
            if logits.shape[-1] > 1:
                preds = self._torch.softmax(logits, -1)[:, target_label]
            else:
                preds = self._torch.sigmoid(logits)[:, 0]
        return preds.view(-1).cpu().numpy()

    def _prepare_label(self, model, target_label):
        if target_label in model.config.id2label:
            return target_label
        if target_label in model.config.label2id:
            return model.config.label2id.get(target_label)
        if isinstance(target_label, str) and target_label.isnumeric():
            target_idx = int(target_label)
            if target_idx in model.config.id2label:
                return target_idx
        raise ValueError(f"Invalid target label {target_label}")

    def score(self, source_text: str, rewritten_text: str) -> Dict[str, float]:
        self._load()
        style_label = self._prepare_label(self._style_model, 0)
        style_scores = self._classify(
            self._style_model, self._style_tokenizer, [rewritten_text], target_label=style_label
        )
        sta = float(self._style_calibration(style_scores))
        if self.fast:
            return {"sta": sta, "sim": 0.0, "fl": 0.0, "j": sta}

        meaning_label = self._prepare_label(self._meaning_model, "paraphrase")
        meaning_scores = self._classify(
            self._meaning_model,
            self._meaning_tokenizer,
            [source_text],
            [rewritten_text],
            target_label=meaning_label,
        )
        sim = float(self._meaning_calibration(meaning_scores))

        cola_label = self._prepare_label(self._cola_model, 1)
        cola_scores = self._classify(
            self._cola_model, self._cola_tokenizer, [rewritten_text], target_label=cola_label
        )
        fl = float(self._fluency_calibration(cola_scores))
        return {"sta": sta, "sim": sim, "fl": fl, "j": sta * sim * fl}


@dataclass
class TaskSpec:
    parser: BaseParser
    reward: Callable[..., float]
    normalize: Callable[[Any], str]


TASK_SPECS: Dict[str, TaskSpec] = {
    "bps": TaskSpec(
        Binary01Parser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_binary),
        _normalize_binary,
    ),
    "lcs": TaskSpec(
        NumericParser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_numeric),
        _normalize_numeric,
    ),
    "simplear": TaskSpec(
        NumericParser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_numeric),
        _normalize_numeric,
    ),
    "rumodar": TaskSpec(
        NumericParser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_numeric),
        _normalize_numeric,
    ),
    "rumultiar": TaskSpec(
        NumericParser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_numeric),
        _normalize_numeric,
    ),
    "mathlogicqa": TaskSpec(
        ChoiceABCDParser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_choice),
        _normalize_choice,
    ),
    "mamuramu": TaskSpec(
        ChoiceABCDParser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_choice),
        _normalize_choice,
    ),
    "rummlu": TaskSpec(
        ChoiceABCDParser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_choice),
        _normalize_choice,
    ),
    "ruopenbookqa": TaskSpec(
        ChoiceABCDParser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_choice),
        _normalize_choice,
    ),
    "ruworldtree": TaskSpec(
        ChoiceABCDParser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_choice),
        _normalize_choice,
    ),
    "parus": TaskSpec(
        Choice12Parser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_binary),
        _normalize_binary,
    ),
    "rcb": TaskSpec(
        Choice123Parser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_binary),
        _normalize_binary,
    ),
    "rwsd": TaskSpec(
        YesNoParser(),
        lambda p, c, a: exact_match_reward(p, c, _normalize_yes_no(a), _normalize_yes_no),
        _normalize_yes_no,
    ),
    "ruhhh": TaskSpec(
        Choice12Parser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_binary),
        _normalize_binary,
    ),
    "ruhatespeech": TaskSpec(
        Choice12Parser(),
        lambda p, c, a: exact_match_reward(p, c, a, _normalize_binary),
        _normalize_binary,
    ),
}


def _build_dataset(
    records: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
) -> Dataset:
    rows = []
    for rec in records:
        prompt = format_prompt(rec["instruction"], rec["inputs"], context=context or "")
        rows.append(
            {
                "prompt": build_messages(prompt, system_prompt=system_prompt),
                "answer": rec.get("outputs"),
                "meta": rec.get("meta", {}),
                "inputs": rec.get("inputs"),
                "instruction": rec.get("instruction"),
            }
        )
    return Dataset.from_list(rows)


def make_single_turn_env(
    task: str,
    split: str = "test",
    data_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_eval_examples: Optional[int] = None,
) -> vf.Environment:
    spec = TASK_SPECS[task]

    train_records = []
    try:
        train_records = load_task_split(task, "train", data_dir=data_dir)
    except FileNotFoundError:
        train_records = []
    eval_records = load_task_split(task, split, data_dir=data_dir)

    train_ds = _build_dataset(train_records, system_prompt=system_prompt) if train_records else None
    eval_ds = _build_dataset(eval_records, system_prompt=system_prompt)
    if max_eval_examples:
        eval_ds = eval_ds.select(range(min(max_eval_examples, len(eval_ds))))
    if train_ds is None:
        train_ds = eval_ds

    def reward_fn(parser: BaseParser, completion: Messages, answer: Any, **_kwargs) -> float:
        return spec.reward(parser, completion, answer)

    rubric = vf.Rubric(parser=spec.parser)
    rubric.add_reward_func(reward_fn)
    return vf.SingleTurnEnv(
        dataset=train_ds, eval_dataset=eval_ds, parser=spec.parser, rubric=rubric
    )


def make_f1_em_env(
    task: str,
    split: str = "test",
    data_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_eval_examples: Optional[int] = None,
) -> vf.Environment:
    parser = FreeTextParser()
    train_records = []
    try:
        train_records = load_task_split(task, "train", data_dir=data_dir)
    except FileNotFoundError:
        train_records = []
    eval_records = load_task_split(task, split, data_dir=data_dir)

    train_ds = _build_dataset(train_records, system_prompt=system_prompt) if train_records else None
    eval_ds = _build_dataset(eval_records, system_prompt=system_prompt)
    if max_eval_examples:
        eval_ds = eval_ds.select(range(min(max_eval_examples, len(eval_ds))))
    if train_ds is None:
        train_ds = eval_ds

    def reward_fn(parser_obj: BaseParser, completion: Messages, answer: Any, **_kwargs) -> float:
        parser_used = parser_obj or parser
        return f1_em_reward(parser_used, completion, str(answer))

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(reward_fn)
    return vf.SingleTurnEnv(dataset=train_ds, eval_dataset=eval_ds, parser=parser, rubric=rubric)


def make_use_env(
    split: str = "test",
    data_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_eval_examples: Optional[int] = None,
) -> vf.Environment:
    parser = FreeTextParser()
    train_records = load_task_split("use", "train", data_dir=data_dir)
    eval_records = load_task_split("use", split, data_dir=data_dir)
    train_ds = _build_dataset(train_records, system_prompt=system_prompt)
    eval_ds = _build_dataset(eval_records, system_prompt=system_prompt)
    if max_eval_examples:
        eval_ds = eval_ds.select(range(min(max_eval_examples, len(eval_ds))))

    def reward_fn(
        parser_obj: BaseParser, completion: Messages, answer: Any, meta: Dict[str, Any], **_kwargs
    ) -> float:
        parser_used = parser_obj or parser
        parsed = parser_used.parse_answer(completion)
        if parsed is None:
            return 0.0
        score = use_example_score(
            str(answer), parsed, meta.get("type", ""), str(meta.get("id_task", ""))
        )
        max_score = int(meta.get("score", 1)) or 1
        return float(score) / float(max_score)

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(reward_fn)
    return vf.SingleTurnEnv(dataset=train_ds, eval_dataset=eval_ds, parser=parser, rubric=rubric)


def make_ruethics_env(
    split: str = "test",
    data_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_eval_examples: Optional[int] = None,
) -> vf.Environment:
    parser = Binary01Parser()
    eval_records = load_task_split("ruethics", split, data_dir=data_dir)
    eval_ds = _build_dataset(eval_records, system_prompt=system_prompt)
    if max_eval_examples:
        eval_ds = eval_ds.select(range(min(max_eval_examples, len(eval_ds))))

    def reward_fn(parser_obj: BaseParser, completion: Messages, answer: Any, **_kwargs) -> float:
        parser_used = parser_obj or parser
        parsed = parser_used.parse_answer(completion)
        if parsed is None:
            return 0.0
        target = 1 if sum(int(v) for v in answer.values()) >= 3 else 0
        return 1.0 if int(parsed) == target else 0.0

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(reward_fn)
    return vf.SingleTurnEnv(dataset=eval_ds, eval_dataset=eval_ds, parser=parser, rubric=rubric)


def make_rudetox_env(
    split: str = "test",
    data_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    fast: bool = False,
    max_eval_examples: Optional[int] = None,
) -> vf.Environment:
    parser = FreeTextParser()
    train_records = []
    try:
        train_records = load_task_split("rudetox", "train", data_dir=data_dir)
    except FileNotFoundError:
        train_records = []
    eval_records = load_task_split("rudetox", split, data_dir=data_dir)

    train_ds = _build_dataset(train_records, system_prompt=system_prompt) if train_records else None
    eval_ds = _build_dataset(eval_records, system_prompt=system_prompt)
    if max_eval_examples:
        eval_ds = eval_ds.select(range(min(max_eval_examples, len(eval_ds))))
    if train_ds is None:
        train_ds = eval_ds

    scorer = RuDetoxScorer(fast=fast)

    def reward_fn(parser_obj: BaseParser, completion: Messages, inputs: Any, **_kwargs) -> float:
        parser_used = parser_obj or parser
        parsed = parser_used.parse_answer(completion)
        if not parsed:
            return 0.0
        scores = scorer.score(str(inputs), parsed)
        return float(scores.get("j", 0.0))

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(reward_fn)
    return vf.SingleTurnEnv(dataset=train_ds, eval_dataset=eval_ds, parser=parser, rubric=rubric)


def _group_rutie_dialogs(records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    dialogs: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        dialog_id = int(rec["meta"]["dialog_id"])
        dialogs.setdefault(dialog_id, []).append(rec)
    grouped = []
    for dialog_id, items in sorted(dialogs.items(), key=lambda x: x[0]):
        items.sort(key=lambda x: int(x["meta"]["question_id"]))
        grouped.append(items)
    return grouped


class RuTiEEnv(vf.MultiTurnEnv):
    def __init__(self, dialogs: List[List[Dict[str, Any]]], split: str, **kwargs):
        dataset = Dataset.from_list([{"prompt": [], "dialog": dialog} for dialog in dialogs])
        super().__init__(eval_dataset=dataset, **kwargs)
        self._parser = Choice12Parser()

    async def setup_state(self, state: vf.State, **_kwargs) -> vf.State:
        dialog = state["dialog"]
        state["dialog"] = dialog
        state["turn_index"] = 0
        state["correct"] = 0
        state["total"] = len(dialog)
        state["context"] = ""

        first = dialog[0]
        prompt = format_prompt(first["instruction"], first["inputs"], context="")
        state["prompt"] = [{"role": "user", "content": prompt}]
        return state

    async def env_response(
        self, messages: Messages, state: vf.State, **_kwargs
    ) -> Tuple[Messages, vf.State]:
        if not messages or messages[-1]["role"] != "assistant":
            return [], state

        dialog: List[Dict[str, Any]] = state["dialog"]
        idx = int(state["turn_index"])
        current = dialog[idx]

        parsed = self._parser.parse_answer(messages)
        gold = str(current["outputs"]).strip()
        if parsed is not None and parsed == gold:
            state["correct"] += 1

        answer_text = ""
        if parsed == "1":
            answer_text = str(current["inputs"]["choice1"])
        elif parsed == "2":
            answer_text = str(current["inputs"]["choice2"])

        question = str(current["inputs"]["question"])
        choice1 = str(current["inputs"]["choice1"])
        choice2 = str(current["inputs"]["choice2"])
        context = state["context"]
        context += f"{question}\n1. {choice1}\n2. {choice2}\n\u041e\u0442\u0432\u0435\u0442: {answer_text}\n\n"
        state["context"] = context
        state["turn_index"] = idx + 1

        if state["turn_index"] >= state["total"]:
            return [], state

        nxt = dialog[state["turn_index"]]
        prompt = format_prompt(nxt["instruction"], nxt["inputs"], context=context)
        return [{"role": "user", "content": prompt}], state

    async def is_completed(self, messages: Messages, state: vf.State, **_kwargs) -> bool:
        return int(state.get("turn_index", 0)) >= int(state.get("total", 0))


def make_rutie_env(
    split: str = "test",
    data_dir: Optional[str] = None,
    max_eval_examples: Optional[int] = None,
) -> vf.Environment:
    records = load_task_split("rutie", split, data_dir=data_dir)
    dialogs = _group_rutie_dialogs(records)
    if max_eval_examples:
        dialogs = dialogs[:max_eval_examples]

    def reward_fn(state: vf.State, **_kwargs) -> float:
        total = int(state.get("total", 0)) or 1
        correct = int(state.get("correct", 0))
        return float(correct) / float(total)

    rubric = vf.Rubric()
    rubric.add_reward_func(reward_fn)
    return RuTiEEnv(dialogs=dialogs, split=split, rubric=rubric)


def make_code_env(
    task: str,
    split: str = "test",
    data_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_eval_examples: Optional[int] = None,
) -> vf.Environment:
    parser = FreeTextParser()
    eval_records = load_task_split(task, split, data_dir=data_dir)
    eval_ds = _build_dataset(eval_records, system_prompt=system_prompt)
    if max_eval_examples:
        eval_ds = eval_ds.select(range(min(max_eval_examples, len(eval_ds))))

    def reward_fn(
        parser_obj: BaseParser,
        completion: Messages,
        inputs: Any,
        meta: Dict[str, Any],
        answer: Any,
        **_kwargs,
    ) -> float:
        parser_used = parser_obj or parser
        parsed = parser_used.parse_answer(completion)
        if not parsed:
            return 0.0
        from mera_common_code import run_code_tests

        return 1.0 if run_code_tests(parsed, inputs, meta, answer) else 0.0

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(reward_fn)
    return vf.SingleTurnEnv(dataset=eval_ds, eval_dataset=eval_ds, parser=parser, rubric=rubric)


def load_task_environment(task: str, split: str = "test", **kwargs) -> vf.Environment:
    if task in {"chegeka", "multiq"}:
        return make_f1_em_env(task, split=split, **kwargs)
    if task == "use":
        return make_use_env(split=split, **kwargs)
    if task == "ruethics":
        return make_ruethics_env(split=split, **kwargs)
    if task == "rudetox":
        return make_rudetox_env(split=split, **kwargs)
    if task == "rutie":
        return make_rutie_env(split=split, **kwargs)
    if task in {"ruhumaneval", "rucodeeval"}:
        return make_code_env(task, split=split, **kwargs)
    return make_single_turn_env(task, split=split, **kwargs)
