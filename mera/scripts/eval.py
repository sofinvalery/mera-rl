from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

env_root = Path(__file__).resolve().parents[1] / "environments"
if str(env_root) not in sys.path:
    sys.path.insert(0, str(env_root))

import mera_common
import mera_common_code


ALL_TASKS = [
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
    "simplear",
    "bps",
    "chegeka",
    "lcs",
    "ruhumaneval",
    "rucodeeval",
    "rummlu",
    "mamuramu",
    "use",
    "rudetox",
    "ruethics",
    "ruhatespeech",
    "ruhhh",
]

BENCHMARK_TASKS = [
    "chegeka",
    "lcs",
    "mamuramu",
    "mathlogicqa",
    "multiq",
    "parus",
    "rcb",
    "rucodeeval",
    "rumodar",
    "rumultiar",
    "ruopenbookqa",
    "rutie",
    "ruworldtree",
    "rwsd",
    "use",
]

VALIDATION_TASKS = ["parus", "rcb", "rwsd", "use"]


TASK_FILE_NAMES = {
    "bps": "BPS",
    "chegeka": "CheGeKa",
    "lcs": "LCS",
    "mathlogicqa": "MathLogicQA",
    "multiq": "MultiQ",
    "parus": "PARus",
    "rcb": "RCB",
    "rudetox": "ruDetox",
    "ruethics": "ruEthics",
    "ruhatespeech": "ruHateSpeech",
    "ruhhh": "ruHHH",
    "ruhumaneval": "ruHumanEval",
    "rucodeeval": "ruCodeEval",
    "rummlu": "ruMMLU",
    "mamuramu": "MaMuRAMu",
    "rumodar": "ruModAr",
    "rumultiar": "ruMultiAr",
    "ruopenbookqa": "ruOpenBookQA",
    "rutie": "ruTiE",
    "ruworldtree": "ruWorldTree",
    "rwsd": "RWSD",
    "simplear": "SimpleAr",
    "use": "USE",
}

TASK_MAX_TOKENS = {
    "chegeka": 64,
    "multiq": 64,
    "rudetox": 128,
    "ruhumaneval": 256,
    "rucodeeval": 256,
}

RUTIE_MAX_TOKENS = 4
RUTIE_TRUNCATION_SAFETY_TOKENS = 8


def parse_choice(task: str, completion: str, answer: Any) -> str:
    if task in mera_common.TASK_SPECS:
        spec = mera_common.TASK_SPECS[task]
        parsed = spec.parser.parse(completion)
        if parsed is None:
            return ""
        value = spec.normalize(parsed)
        if task == "rwsd":
            return "Да" if value == "YES" else "Нет"
        return value
    return completion.strip()


def parse_use_answer(completion: str) -> str:
    text = completion.strip()
    match = re.search(r"\d+(?:,\d+)*", text)
    if match:
        return match.group(0)
    match = re.search(r"[A-Za-z\u0400-\u04FF]+", text)
    if match:
        return match.group(0).lower()
    return text


def build_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch(llm: LLM, prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    completions = []
    for out in outputs:
        text = out.outputs[0].text if out.outputs else ""
        completions.append(text)
    return completions


def build_submission_entry(outputs: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {"outputs": outputs, "meta": meta}


def run_task(
    task: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    split: str,
    data_dir: str | None,
    limit: int | None,
    temperature: float,
    max_model_len: int | None,
) -> List[Dict[str, Any]]:
    records = mera_common.load_task_split(task, split, data_dir=data_dir)
    if limit:
        records = records[:limit]

    if task == "rutie":
        return run_rutie(llm, tokenizer, records, temperature, max_model_len)

    prompts = []
    for rec in records:
        prompt = mera_common.format_prompt(rec["instruction"], rec["inputs"], context="")
        prompts.append(build_prompt(tokenizer, prompt))

    max_tokens = TASK_MAX_TOKENS.get(task, 32)
    completions = generate_batch(llm, prompts, max_tokens, temperature)

    outputs = []
    for rec, completion in zip(records, completions):
        meta = dict(rec.get("meta", {}))
        if task in {"ruhumaneval", "rucodeeval"}:
            pred = mera_common_code.preprocess_generation(completion)
        elif task == "ruethics":
            parsed = mera_common.Binary01Parser().parse(completion)
            if parsed is None:
                parsed = "0"
            pred = {k: parsed for k in rec["outputs"].keys()}
        elif task == "use":
            pred = parse_use_answer(completion)
        else:
            pred = parse_choice(task, completion, rec.get("outputs"))
        outputs.append(build_submission_entry(pred, meta))
    return outputs



def _truncate_rutie_context(
    tokenizer,
    max_model_len,
    instruction,
    inputs,
    context_parts,
    output_token_reserve: int,
):
    if max_model_len is None:
        context = "\n\n".join(context_parts)
        prompt = mera_common.format_prompt(instruction, inputs, context=context)
        return prompt, build_prompt(tokenizer, prompt), context_parts

    prompt_token_limit = max(1, max_model_len - max(1, output_token_reserve))
    while True:
        context = "\n\n".join(context_parts)
        prompt = mera_common.format_prompt(instruction, inputs, context=context)
        formatted = build_prompt(tokenizer, prompt)
        token_count = len(tokenizer.encode(formatted, add_special_tokens=False))
        if token_count <= prompt_token_limit:
            return prompt, formatted, context_parts
        if not context_parts:
            return prompt, formatted, context_parts
        context_parts.pop(0)

def run_rutie(
    llm: LLM,
    tokenizer: AutoTokenizer,
    records: List[Dict[str, Any]],
    temperature: float,
    max_model_len: int | None,
) -> List[Dict[str, Any]]:
    rutie_output_token_reserve = RUTIE_MAX_TOKENS + RUTIE_TRUNCATION_SAFETY_TOKENS
    dialogs: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        dialog_id = int(rec["meta"]["dialog_id"])
        dialogs.setdefault(dialog_id, []).append(rec)
    outputs: List[Dict[str, Any]] = []
    for dialog_id, items in sorted(dialogs.items(), key=lambda x: x[0]):
        items.sort(key=lambda x: int(x["meta"]["question_id"]))
        context_parts = []
        for rec in items:
            prompt, formatted, context_parts = _truncate_rutie_context(
                tokenizer,
                max_model_len,
                rec["instruction"],
                rec["inputs"],
                context_parts,
                rutie_output_token_reserve,
            )
            while True:
                try:
                    completion = generate_batch(llm, [formatted], RUTIE_MAX_TOKENS, temperature)[0]
                    break
                except ValueError as exc:
                    if "maximum model length" not in str(exc) or not context_parts:
                        raise
                    context_parts.pop(0)
                    prompt, formatted, context_parts = _truncate_rutie_context(
                        tokenizer,
                        max_model_len,
                        rec["instruction"],
                        rec["inputs"],
                        context_parts,
                        rutie_output_token_reserve,
                    )
            parsed = mera_common.Choice12Parser().parse(completion) or ""
            outputs.append(build_submission_entry(parsed, rec["meta"]))
            choice1 = str(rec["inputs"]["choice1"])
            choice2 = str(rec["inputs"]["choice2"])
            answer_text = ""
            if parsed == "1":
                answer_text = choice1
            elif parsed == "2":
                answer_text = choice2
            context_parts.append(
                f"{rec['inputs']['question']}\n1. {choice1}\n2. {choice2}\nОтвет: {answer_text}"
            )
    return outputs

def build_submission_files(
    output_dir: Path, split: str, task_outputs: Dict[str, List[Dict[str, Any]]]
) -> Path:
    submission_dir = output_dir / "submission_files"
    submission_dir.mkdir(parents=True, exist_ok=True)
    for task, outputs in task_outputs.items():
        file_name = TASK_FILE_NAMES.get(task, task)
        path = submission_dir / f"{file_name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"data": {split: outputs}}, handle, ensure_ascii=False)
    zip_path = shutil.make_archive(str(output_dir / "submission"), "zip", submission_dir)
    return Path(zip_path)


def run_scoring(zip_path: Path, results_path: Path) -> Dict[str, Any]:
    scoring_root = mera_common.resolve_mera_repo_root() / "modules" / "scoring"
    zip_path = Path(zip_path).resolve()
    results_path = Path(results_path).resolve()
    active_python = Path(sys.executable)
    missing_dep_help = (
        f"Missing scoring dependency 'omegaconf' in interpreter {active_python}. "
        "Install scoring deps with: "
        f"uv pip install --python {active_python} omegaconf boto3 scikit-learn"
    )
    try:
        import omegaconf  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(missing_dep_help) from exc

    sys.path.insert(0, str(scoring_root))
    from src.worker import Worker
    from src.utils import save_json

    config_path = scoring_root / "configs" / "main.yaml"
    cwd = Path.cwd()
    try:
        os.chdir(scoring_root)
        worker = Worker(conf=str(config_path), no_load_models=False)
        worker.load()
        results = worker.evaluate(local_path=str(zip_path), remove_local_file=False)
        save_json(results, str(results_path))
        return results
    finally:
        os.chdir(cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument(
        "--task-set",
        choices=["all", "benchmark", "validation"],
        default="all",
        help="Task preset to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit subset of tasks to evaluate (overrides --task-set).",
    )
    return parser.parse_args()


def resolve_eval_tasks(task_set: str, tasks: list[str] | None) -> list[str]:
    if tasks:
        selected = tasks
    elif task_set == "benchmark":
        selected = BENCHMARK_TASKS
    elif task_set == "validation":
        selected = VALIDATION_TASKS
    else:
        selected = ALL_TASKS

    unknown = sorted(set(selected) - set(ALL_TASKS))
    if unknown:
        raise ValueError(f"Unknown eval task(s): {', '.join(unknown)}")
    return selected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    eval_tasks = resolve_eval_tasks(args.task_set, args.tasks)

    wandb_run = None
    if args.wandb:
        import wandb

        project = args.wandb_project or os.getenv("WANDB_PROJECT")
        entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
        run_name = args.wandb_run_name or os.getenv("WANDB_RUN_NAME")
        wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=vars(args),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    task_outputs = {}
    for task in tqdm(eval_tasks, desc="eval tasks", unit="task"):
        task_outputs[task] = run_task(
            task,
            llm,
            tokenizer,
            args.split,
            args.data_dir,
            args.limit,
            args.temperature,
            args.max_model_len,
        )

    zip_path = build_submission_files(output_dir, args.split, task_outputs)
    if args.skip_scoring:
        print(f"Skipping scoring. Submission at {zip_path}")
        if wandb_run is not None:
            wandb_run.summary["eval/skipped_scoring"] = 1
            wandb_run.summary["eval/submission_zip"] = str(zip_path)
            wandb_run.finish()
        return
    results_path = (output_dir / "submission_results.json").resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = run_scoring(zip_path, results_path)
    if wandb_run is not None:
        metrics = {}
        results_payload = results.get("results", {})
        total_score = results_payload.get("total_score")
        if isinstance(total_score, (int, float)):
            metrics["total_score"] = total_score
        for task_name, task_metrics in results_payload.items():
            if task_name == "total_score" or not isinstance(task_metrics, dict):
                continue
            for metric_name, value in task_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"{task_name}/{metric_name}"] = value
        wandb_run.log(metrics)
        wandb_run.finish()


if __name__ == "__main__":
    main()
