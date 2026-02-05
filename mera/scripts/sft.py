from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
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


def build_text(prompt: str, answer: Any, tokenizer: AutoTokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted + str(answer)


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
            output.append({"prompt": prompt, "answer": rec["outputs"]})
            choice1 = str(rec["inputs"]["choice1"])
            choice2 = str(rec["inputs"]["choice2"])
            answer_text = choice1 if str(rec["outputs"]).strip() == "1" else choice2
            context += (
                f"{rec['inputs']['question']}\n1. {choice1}\n2. {choice2}\nОтвет: {answer_text}\n\n"
            )
    return output


def build_sft_dataset(data_dir: str | None, tokenizer: AutoTokenizer, limit: int | None) -> Dataset:
    rows = []
    for task in tqdm(TRAIN_TASKS, desc="sft tasks", unit="task"):
        records = mera_common.load_task_split(task, "train", data_dir=data_dir)
        if limit:
            records = records[:limit]
        if task == "rutie":
            samples = build_rutie_context(records)
            for sample in tqdm(samples, desc=f"sft:{task}", unit="ex", leave=False):
                rows.append(
                    {
                        "text": build_text(sample["prompt"], sample["answer"], tokenizer),
                        "task": task,
                    }
                )
            continue
        for rec in tqdm(records, desc=f"sft:{task}", unit="ex", leave=False):
            prompt = mera_common.format_prompt(rec["instruction"], rec["inputs"], context="")
            rows.append({"text": build_text(prompt, rec["outputs"], tokenizer), "task": task})
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/sft")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--limit", type=int, default=None)
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

    dataset = build_sft_dataset(args.data_dir, tokenizer, args.limit)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to=["wandb"] if args.wandb else [],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
