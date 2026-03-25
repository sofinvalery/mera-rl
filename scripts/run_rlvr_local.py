#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Literal, assert_never

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.manifest import ensure_manifest, update_manifest


Mode = Literal["smoke", "train"]
SFT_MODEL = "sofinvalery/mera-qwen3-4b-sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local MERA RLVR with prime-rl.")
    parser.add_argument("mode", choices=["smoke", "train"])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode: Mode = args.mode
    run_root = REPO_ROOT / "outputs" / "runs" / args.experiment
    output_dir = run_root / ("rlvr_smoke" if mode == "smoke" else "rlvr")
    manifest_path = run_root / "manifest.json"
    config_copy_path = run_root / "configs" / f"rlvr_{mode}.toml"
    config_copy_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "smoke":
        template_path = REPO_ROOT / "configs" / "rl" / "local" / "mera-rlvr-smoke.toml"
    elif mode == "train":
        template_path = REPO_ROOT / "configs" / "rl" / "local" / "mera-rlvr-train.toml"
    else:
        assert_never(mode)
    assert template_path.exists(), template_path

    rl_entry = os.getenv("PRIME_RL_ENTRY") or shutil.which("rl")
    if args.dry_run and rl_entry is None:
        rl_entry = "rl"
    assert rl_entry is not None, "PRIME_RL_ENTRY is not set and 'rl' is not on PATH."
    if not args.dry_run:
        assert Path(rl_entry).exists() or rl_entry == "rl", rl_entry

    config_copy_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    ensure_manifest(manifest_path, experiment=args.experiment, base_model=SFT_MODEL)

    cmd = [
        rl_entry,
        "@",
        str(config_copy_path),
        "--output-dir",
        str(output_dir),
    ]

    print(f"mode={mode}")
    print(f"experiment={args.experiment}")
    print(f"config={config_copy_path}")
    print(f"output_dir={output_dir}")
    print(f"command={shlex.join(cmd)}")

    if args.dry_run:
        return

    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    if "UV_PROJECT" not in env:
        resolved_rl_entry = rl_entry if rl_entry != "rl" else (shutil.which("rl") or rl_entry)
        rl_path = Path(resolved_rl_entry)
        if rl_path.is_absolute():
            # prime-rl launcher uses nested `uv run ...`; pin uv project to the rl installation root.
            uv_project = rl_path.resolve().parents[2]
            if (uv_project / "pyproject.toml").exists():
                env["UV_PROJECT"] = str(uv_project)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    if mode == "smoke":
        update_manifest(
            manifest_path,
            {
                "notes": {
                    "rlvr_smoke": {
                        "config_path": str(config_copy_path),
                        "output_dir": str(output_dir),
                        "status": "completed",
                    }
                }
            },
        )
        return

    weight_dir = output_dir / "weights"
    stable_steps = [p for p in weight_dir.glob("step_*") if (p / "STABLE").exists()]
    assert stable_steps, f"No stable checkpoint in {weight_dir}"
    stable_steps.sort(key=lambda p: int(p.name.split("_")[-1]))
    final_step_path = stable_steps[-1]

    hf_repo_id = os.getenv("HF_RL_REPO_ID")
    assert hf_repo_id, "HF_RL_REPO_ID is required for train mode."
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    assert hf_token, "Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN before train mode."
    env["HUGGINGFACE_HUB_TOKEN"] = hf_token

    publish_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "publish_hf_artifact.py"),
        "--kind",
        "rl",
        "--source-path",
        str(final_step_path),
        "--repo-id",
        hf_repo_id,
        "--artifact-type",
        "checkpoint",
        "--task",
        "fair",
        "--manifest",
        str(manifest_path),
        "--experiment",
        args.experiment,
        "--public",
    ]
    print(f"publish_command={shlex.join(publish_cmd)}")
    subprocess.run(publish_cmd, check=True, cwd=REPO_ROOT, env=env)

    update_manifest(
        manifest_path,
        {
            "rl_runs": {
                "fair_local": {
                    "config_path": str(config_copy_path),
                    "output_dir": str(output_dir),
                    "final_checkpoint_path": str(final_step_path),
                    "hf_repo_id": hf_repo_id,
                    "status": "completed",
                }
            }
        },
    )


if __name__ == "__main__":
    main()
