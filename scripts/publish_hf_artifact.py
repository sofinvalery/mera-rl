#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.hf_utils import upload_file, upload_folder
from prime_lab_rl.manifest import ensure_manifest, update_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a local SFT/RL artifact to Hugging Face.")
    parser.add_argument("--kind", choices=["sft", "rl"], required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--artifact-type", choices=["adapter", "merged", "checkpoint"], default="adapter")
    parser.add_argument("--path-in-repo", default=".")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--base-model", default=None, help="Base model name for LoRA merge.")
    parser.add_argument("--merge-lora", action="store_true", help="Merge a PEFT adapter into a full model before upload.")
    parser.add_argument("--task", default=None, help="Required for --kind rl.")
    parser.add_argument("--experiment", default="manual")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_false", dest="private")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_model_card(path: Path, *, repo_id: str, artifact_type: str, base_model: str | None) -> None:
    readme = path / "README.md"
    if readme.exists():
        return
    lines = [
        f"# {repo_id}",
        "",
        f"Artifact type: `{artifact_type}`.",
    ]
    if base_model:
        lines.append(f"Base model: `{base_model}`.")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_lora(source_path: Path, base_model: str) -> Path:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers and peft are required for --merge-lora") from exc

    merged_dir = Path(tempfile.mkdtemp(prefix="mera-merged-"))
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(model, str(source_path))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir


def main() -> None:
    args = parse_args()
    source_path = args.source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")
    if args.kind == "rl" and not args.task:
        raise ValueError("--task is required for --kind rl")

    upload_source = source_path
    temp_dir: Path | None = None
    if args.merge_lora:
        if not args.base_model:
            raise ValueError("--base-model is required with --merge-lora")
        temp_dir = _merge_lora(source_path, args.base_model)
        upload_source = temp_dir

    if upload_source.is_dir():
        _write_model_card(upload_source, repo_id=args.repo_id, artifact_type=args.artifact_type, base_model=args.base_model)

    commit_message = args.commit_message or f"Upload {args.kind} {args.artifact_type} artifact"
    print(f"repo_id={args.repo_id}")
    print(f"upload_source={upload_source}")
    print(f"path_in_repo={args.path_in_repo}")

    if args.dry_run:
        return

    if upload_source.is_dir():
        result = upload_folder(
            upload_source,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            private=args.private,
            commit_message=commit_message,
            revision=args.revision,
        )
    else:
        result = upload_file(
            upload_source,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            private=args.private,
            commit_message=commit_message,
            revision=args.revision,
        )

    print(f"commit_oid={result.get('commit_oid')}")
    print(f"commit_url={result.get('commit_url')}")

    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        ensure_manifest(manifest_path, experiment=args.experiment, base_model=args.base_model or "")
        patch = {
            "sft": {"hf": result | {"artifact_type": args.artifact_type}} if args.kind == "sft" else {},
            "rl_runs": {
                args.task: {"hf": result | {"artifact_type": args.artifact_type}}
            }
            if args.kind == "rl" and args.task
            else {},
        }
        update_manifest(manifest_path, patch)

    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
