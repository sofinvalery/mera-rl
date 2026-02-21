from __future__ import annotations

import argparse
from pathlib import Path
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
from typing import Any, Dict, List

from collections import Counter

import sys

import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

env_root = Path(__file__).resolve().parents[1] / "environments"
if str(env_root) not in sys.path:
    sys.path.insert(0, str(env_root))

import mera_common


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


def resolve_sft_tasks(args: argparse.Namespace) -> list[str]:
    if args.tasks:
        tasks = args.tasks
    elif args.task_set == "all":
        tasks = ALL_SFT_TASKS
    else:
        tasks = FAIR_SFT_TASKS

    unknown = sorted(set(tasks) - set(SFT_SPLITS))
    if unknown:
        raise ValueError(f"Unknown SFT task(s): {', '.join(unknown)}")
    return tasks


def build_sft_dataset(
    data_dir: str | None,
    tokenizer: AutoTokenizer,
    limit: int | None,
    tasks: list[str],
) -> Dataset:
    rows = []
    for task in tqdm(tasks, desc="sft tasks", unit="task"):
        split = SFT_SPLITS[task]
        records = mera_common.load_task_split(task, split, data_dir=data_dir)
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
    parser.add_argument(
        "--task-set",
        choices=["fair", "all"],
        default="fair",
        help="Task preset: fair excludes diagnostic tasks and keeps fair training splits.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit task subset to train on (overrides --task-set).",
    )
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Save a merged full model checkpoint after LoRA SFT.",
    )
    parser.add_argument(
        "--merged-output-dir",
        default=None,
        help="Path for merged model output (default: <output-dir>/merged).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merged_output_dir and not args.save_merged:
        args.save_merged = True

    tasks = resolve_sft_tasks(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    dataset = build_sft_dataset(args.data_dir, tokenizer, args.limit, tasks)

    if wandb_run is not None:
        task_counts = Counter(dataset["task"])
        wandb_run.summary["dataset/num_examples"] = len(dataset)
        for task, count in task_counts.items():
            wandb_run.summary[f"dataset/tasks/{task}"] = count

    training_args = SFTConfig(
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
        dataset_text_field="text",
        max_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    if args.save_merged:
        merged_output_dir = (
            Path(args.merged_output_dir) if args.merged_output_dir else output_dir / "merged"
        )

        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        merge_model = AutoPeftModelForCausalLM.from_pretrained(
            str(output_dir),
            torch_dtype=dtype,
            device_map="auto",
        )
        merged_model = merge_model.merge_and_unload()
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_output_dir))

        if wandb_run is not None:
            wandb_run.summary["model/merged_path"] = str(merged_output_dir)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
