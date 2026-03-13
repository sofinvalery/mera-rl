#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.manifest import ensure_manifest, update_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a READY hosted RL checkpoint from Prime.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--step", type=int, default=None, help="Pick a specific checkpoint step.")
    parser.add_argument("--task", default=None, help="Task key for manifest updates.")
    parser.add_argument("--experiment", default="manual")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--extract", action="store_true", help="Extract .zip/.tar(.gz) after download.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _list_checkpoints(run_id: str) -> list[dict]:
    cmd = ["prime", "rl", "checkpoints", run_id, "--output", "json"]
    result = shutil.which("prime")
    if result is None:
        raise FileNotFoundError("prime CLI not found on PATH")
    import subprocess

    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    proc = subprocess.run(cmd, check=True, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    payload = json.loads(proc.stdout)
    return payload.get("checkpoints", [])


def _select_checkpoint(checkpoints: list[dict], checkpoint_id: str | None, step: int | None) -> dict:
    ready = [cp for cp in checkpoints if cp.get("status") == "READY"]
    if checkpoint_id:
        for checkpoint in ready:
            if checkpoint.get("id") == checkpoint_id:
                return checkpoint
        raise ValueError(f"Checkpoint '{checkpoint_id}' not found among READY checkpoints.")
    if step is not None:
        for checkpoint in ready:
            if int(checkpoint.get("step", -1)) == step:
                return checkpoint
        raise ValueError(f"No READY checkpoint found at step {step}.")
    if not ready:
        raise ValueError("No READY checkpoints available for this run.")
    ready.sort(key=lambda checkpoint: int(checkpoint.get("step", 0)))
    return ready[-1]


def _filename_from_storage_url(storage_url: str, checkpoint_id: str) -> str:
    parsed = urllib.parse.urlparse(storage_url)
    name = Path(parsed.path).name
    return name or f"{checkpoint_id}.bin"


def _extract_archive(path: Path, output_dir: Path) -> Path:
    if path.suffix == ".zip":
        target_dir = output_dir / path.stem
        with zipfile.ZipFile(path, "r") as archive:
            archive.extractall(target_dir)
        return target_dir
    if path.suffix in {".tar", ".tgz", ".gz"} or path.name.endswith(".tar.gz"):
        stem = path.name[:-7] if path.name.endswith(".tar.gz") else path.stem
        target_dir = output_dir / stem
        with tarfile.open(path, "r:*") as archive:
            archive.extractall(target_dir)
        return target_dir
    raise ValueError(f"Unsupported archive format for extraction: {path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoints = _list_checkpoints(args.run_id)
    checkpoint = _select_checkpoint(checkpoints, args.checkpoint_id, args.step)

    storage_url = checkpoint["storageUrl"]
    filename = _filename_from_storage_url(storage_url, checkpoint["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    download_path = output_dir / filename
    print(f"checkpoint_id={checkpoint['id']}")
    print(f"step={checkpoint['step']}")
    print(f"storage_url={storage_url}")
    print(f"download_path={download_path}")

    if args.dry_run:
        return

    urllib.request.urlretrieve(storage_url, download_path)
    extracted_path = None
    if args.extract and (
        download_path.suffix == ".zip"
        or download_path.suffix in {".tar", ".tgz", ".gz"}
        or download_path.name.endswith(".tar.gz")
    ):
        extracted_path = _extract_archive(download_path, output_dir)
        print(f"extracted_path={extracted_path}")

    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        ensure_manifest(manifest_path, experiment=args.experiment, base_model="")
        patch = {
            "rl_runs": {
                args.task or args.run_id: {
                    "checkpoint": {
                        "run_id": args.run_id,
                        "checkpoint_id": checkpoint["id"],
                        "step": checkpoint["step"],
                        "storage_url": storage_url,
                        "download_path": str(download_path),
                        "extracted_path": str(extracted_path) if extracted_path else None,
                    }
                }
            }
        }
        update_manifest(manifest_path, patch)


if __name__ == "__main__":
    main()
