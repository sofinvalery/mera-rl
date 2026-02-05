from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

env_root = Path(__file__).resolve().parents[1] / "environments"
if str(env_root) not in sys.path:
    sys.path.insert(0, str(env_root))

import mera_common


TRAIN_TASKS = [
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


def build_rutie_context(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dialogs = {}
    for rec in records:
        dialog_id = int(rec["meta"]["dialog_id"])
        dialogs.setdefault(dialog_id, []).append(rec)
    output = []
    for dialog_id, items in sorted(dialogs.items(), key=lambda x: x[0]):
        items.sort(key=lambda x: int(x["meta"]["question_id"]))
        context = ""
        for rec in items:
            prompt = mera_common.format_prompt(rec["instruction"], rec["inputs"], context=context)
            output.append(
                {
                    "prompt": prompt,
                    "answer": rec["outputs"],
                    "inputs": rec["inputs"],
                    "meta": rec["meta"],
                }
            )
            choice1 = str(rec["inputs"]["choice1"])
            choice2 = str(rec["inputs"]["choice2"])
            answer_text = choice1 if str(rec["outputs"]).strip() == "1" else choice2
            context += (
                f"{rec['inputs']['question']}\n1. {choice1}\n2. {choice2}\nОтвет: {answer_text}\n\n"
            )
    return output


def build_grpo_dataset(
    data_dir: str | None, limit: int | None, tokenizer: AutoTokenizer
) -> Dataset:
    rows = []
    for task in tqdm(TRAIN_TASKS, desc="grpo tasks", unit="task"):
        records = mera_common.load_task_split(task, "train", data_dir=data_dir)
        if limit:
            records = records[:limit]
        if task == "rutie":
            for sample in tqdm(build_rutie_context(records), desc=f"grpo:{task}", unit="ex", leave=False):
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rows.append(
                    {
                        "task": task,
                        "prompt": formatted,
                        "answer": sample["answer"],
                        "inputs": sample["inputs"],
                        "meta": sample["meta"],
                    }
                )
            continue
        for rec in tqdm(records, desc=f"grpo:{task}", unit="ex", leave=False):
            prompt = mera_common.format_prompt(rec["instruction"], rec["inputs"], context="")
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            rows.append(
                {
                    "task": task,
                    "prompt": formatted,
                    "answer": rec["outputs"],
                    "inputs": rec["inputs"],
                    "meta": rec.get("meta", {}),
                }
            )
    return Dataset.from_list(rows)


def compute_reward(
    task: str,
    completion: str,
    answer: Any,
    inputs: Any,
    meta: Dict[str, Any],
    rudetox_scorer: mera_common.RuDetoxScorer,
) -> float:
    completion = completion.strip()
    if task in mera_common.TASK_SPECS:
        spec = mera_common.TASK_SPECS[task]
        parsed = spec.parser.parse(completion)
        if parsed is None:
            return 0.0
        return 1.0 if spec.normalize(parsed) == spec.normalize(answer) else 0.0
    if task in {"chegeka", "multiq"}:
        parser = mera_common.FreeTextParser()
        return mera_common.f1_em_reward(parser, completion, str(answer))
    if task == "use":
        parser = mera_common.FreeTextParser()
        parsed = parser.parse(completion)
        if parsed is None:
            return 0.0
        score = mera_common.use_example_score(
            str(answer), parsed, meta.get("type", ""), str(meta.get("id_task", ""))
        )
        max_score = int(meta.get("score", 1)) or 1
        return float(score) / float(max_score)
    if task == "rudetox":
        if not completion:
            return 0.0
        scores = rudetox_scorer.score(str(inputs), completion)
        return float(scores.get("j", 0.0))
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/grpo")
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="Use fast ruDetox reward (STA only)")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:  # pragma: no cover - depends on installed TRL version
        raise RuntimeError("GRPOTrainer not available. Install trl>=0.12.0.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
    )
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = build_grpo_dataset(args.data_dir, args.limit, tokenizer)
    rudetox_scorer = mera_common.RuDetoxScorer(fast=args.fast)

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        tasks = kwargs.get("task", [])
        answers = kwargs.get("answer", [])
        inputs = kwargs.get("inputs", [])
        metas = kwargs.get("meta", [])
        rewards = []
        for task, completion, answer, inp, meta in zip(tasks, completions, answers, inputs, metas):
            rewards.append(compute_reward(task, completion, answer, inp, meta, rudetox_scorer))
        return rewards

    config = GRPOConfig(
        output_dir=str(output_dir),
        max_prompt_length=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        report_to=["wandb"] if args.wandb else [],
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        peft_config=peft_config,
        args=config,
        prompt_field="prompt",
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
