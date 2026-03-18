#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_SFT_BATCH_SIZE,
    DEFAULT_SFT_EPOCHS,
    DEFAULT_SFT_GRAD_ACCUM,
    DEFAULT_SFT_LORA_ALPHA,
    DEFAULT_SFT_LORA_DROPOUT,
    DEFAULT_SFT_LORA_RANK,
    DEFAULT_SFT_LR,
    DEFAULT_SFT_MAX_SEQ_LEN,
    DEFAULT_SFT_MICRO_BATCH_SIZE,
    DEFAULT_SFT_SAVE_STEPS,
    DEFAULT_SFT_WARMUP_RATIO,
    OUTPUTS_ROOT,
)
from prime_lab_rl.manifest import ensure_manifest, update_manifest
from prime_lab_rl.sft_dataset import (
    RUTIE_CONTEXT_MODES,
    prepare_sft_dataset_artifacts,
    resolve_sft_tasks,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run local Prime-RL SFT on MERA fair tasks.")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "sft")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "sft" / "mera-fair-no-lora.toml")
    parser.add_argument("--task-set", choices=["fair"], default="fair")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drop-overlength", action="store_true", default=True)
    parser.add_argument("--no-drop-overlength", action="store_false", dest="drop_overlength")
    parser.add_argument("--rutie-context-mode", choices=RUTIE_CONTEXT_MODES, default="single_turn")

    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_SFT_MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SFT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_SFT_GRAD_ACCUM)
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_SFT_MICRO_BATCH_SIZE)
    parser.add_argument("--epochs", type=float, default=DEFAULT_SFT_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=DEFAULT_SFT_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_SFT_WARMUP_RATIO)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SFT_SAVE_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pack-function", choices=["cat", "stack"], default="cat")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")

    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_SFT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=DEFAULT_SFT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_SFT_LORA_DROPOUT)

    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--torchrun-bin", default=None, help="Optional explicit torchrun binary.")
    parser.add_argument("--sft-entry", default="sft", help="Prime-RL SFT entrypoint on PATH.")
    parser.add_argument(
        "--min-free-gb-preflight",
        type=float,
        default=45.0,
        help="Abort before launch if free disk on output filesystem is below this threshold.",
    )
    parser.add_argument(
        "--min-free-gb-runtime",
        type=float,
        default=10.0,
        help="Disk watchdog threshold during training; terminate run if free disk goes below this value.",
    )
    parser.add_argument(
        "--disk-watch-interval-seconds",
        type=int,
        default=180,
        help="Disk watchdog polling interval during training.",
    )
    parser.add_argument("--disk-watchdog", action="store_true", default=True)
    parser.add_argument("--no-disk-watchdog", action="store_false", dest="disk_watchdog")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "mera"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME"))
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline"],
        default=os.getenv("MERA_WANDB_MODE", "auto"),
        help="W&B launch mode. auto probes online access and falls back to offline on failure.",
    )
    parser.add_argument(
        "--wandb-online-probe-timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for W&B online probe in auto/online mode.",
    )
    parser.add_argument("--wandb-offline-fallback", action="store_true", default=True)
    parser.add_argument("--no-wandb-offline-fallback", action="store_false", dest="wandb_offline_fallback")
    parser.add_argument("--hf-adapter-repo-id", default=None, help="Upload the final SFT adapter to this HF repo after training.")
    parser.add_argument("--hf-merged-repo-id", default=None, help="Upload a merged full-model handoff artifact to this HF repo after training.")
    parser.add_argument("--hf-private", action="store_true", default=True)
    parser.add_argument("--hf-public", action="store_false", dest="hf_private")
    parser.add_argument(
        "--gpu-peak-tflops",
        type=float,
        default=None,
        help="Sets PRIME_GPU_PEAK_FLOPS_TFLOPS for corrected MFU reporting (or use MERA_GPU_PEAK_TFLOPS env).",
    )

    parser.add_argument("--experiment", default="manual")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()

def _infer_nproc_per_node(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        num = len([token for token in visible.split(",") if token.strip()])
        if num > 0:
            return num
    return 1


def _build_launch_command(
    *,
    torchrun_bin: str | None,
    resolved_sft_entry: str,
    config_path: Path,
    nproc_per_node: int,
    master_port: int | None,
    override_args: list[str],
    extra_args: list[str],
) -> list[str]:
    # Prime-RL's `sft` entrypoint manages distributed launch internally via deployment.num-gpus.
    # Wrapping it in an outer torchrun causes nested launches and GPU contention.
    _ = (torchrun_bin, nproc_per_node, master_port)
    cmd = [resolved_sft_entry, "@", str(config_path)]
    cmd.extend(override_args)
    cmd.extend(extra_args)
    return cmd


def _infer_torchrun_bin(explicit_torchrun_bin: str | None, resolved_sft_entry: str) -> str:
    if explicit_torchrun_bin:
        return explicit_torchrun_bin

    entry_path = Path(resolved_sft_entry).expanduser()
    if entry_path.exists():
        sibling_torchrun = entry_path.resolve().with_name("torchrun")
        if sibling_torchrun.exists():
            return str(sibling_torchrun)

    return shutil.which("torchrun") or "torchrun"


def _infer_sft_workdir(resolved_sft_entry: str) -> Path:
    entry_path = Path(resolved_sft_entry).expanduser()
    if not entry_path.exists():
        return REPO_ROOT

    # Typical editable prime-rl install: <repo>/.venv/bin/sft
    candidate = entry_path.resolve().parents[2]
    if (candidate / "pyproject.toml").exists() and (
        (candidate / "src" / "prime_rl").exists() or (candidate / "prime_rl").exists()
    ):
        return candidate

    return REPO_ROOT


def _bytes_from_gb(gb: float) -> int:
    return int(gb * (1024**3))


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _ensure_min_free_space(path: Path, *, min_free_gb: float, stage: str) -> None:
    if min_free_gb <= 0:
        return
    min_free_bytes = _bytes_from_gb(min_free_gb)
    free_bytes = _free_bytes(path)
    if free_bytes < min_free_bytes:
        free_gb = free_bytes / (1024**3)
        raise RuntimeError(
            f"Insufficient free disk at {stage}. path={path} free_gb={free_gb:.2f} "
            f"required_gb={min_free_gb:.2f}"
        )


def _run_with_disk_watchdog(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    disk_path: Path,
    min_free_gb_runtime: float,
    poll_interval_seconds: int,
    enable_watchdog: bool,
) -> None:
    if not enable_watchdog or min_free_gb_runtime <= 0:
        subprocess.run(cmd, check=True, cwd=cwd, env=env)
        return

    threshold_bytes = _bytes_from_gb(min_free_gb_runtime)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)
    poll = max(5, poll_interval_seconds)

    while True:
        code = proc.poll()
        if code is not None:
            if code != 0:
                raise subprocess.CalledProcessError(code, cmd)
            return

        free_bytes = _free_bytes(disk_path)
        if free_bytes < threshold_bytes:
            free_gb = free_bytes / (1024**3)
            print(
                "disk_watchdog=terminate "
                f"free_gb={free_gb:.2f} threshold_gb={min_free_gb_runtime:.2f} path={disk_path}"
            )
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            raise RuntimeError(
                "Disk watchdog terminated training due to low free space. "
                f"free_gb={free_gb:.2f} threshold_gb={min_free_gb_runtime:.2f}"
            )

        time.sleep(poll)


def _resolve_latest_stable_weight_dir(output_dir: Path) -> Path:
    weights_dir = output_dir / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    stable_steps: list[tuple[int, Path]] = []
    for step_dir in weights_dir.glob("step_*"):
        if not step_dir.is_dir():
            continue
        try:
            step = int(step_dir.name.split("_")[-1])
        except ValueError:
            continue
        if (step_dir / "STABLE").exists():
            stable_steps.append((step, step_dir))

    if not stable_steps:
        raise FileNotFoundError(f"No stable Prime-RL weight checkpoints found in {weights_dir}.")

    stable_steps.sort(key=lambda item: item[0])
    return stable_steps[-1][1]


def _write_latest_pointers(output_dir: Path, latest_weight_dir: Path) -> None:
    step = latest_weight_dir.name.split("_")[-1]
    (output_dir / "LATEST_WEIGHT_STEP").write_text(f"{step}\n", encoding="utf-8")

    latest_link = output_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(latest_weight_dir.resolve())


def _run_hf_publish(
    *,
    repo_id: str,
    source_path: Path,
    artifact_type: str,
    private: bool,
    manifest_path: Path | None,
    experiment: str,
    base_model: str | None,
    merge_lora: bool,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "publish_hf_artifact.py"),
        "--kind",
        "sft",
        "--source-path",
        str(source_path),
        "--repo-id",
        repo_id,
        "--artifact-type",
        artifact_type,
        "--experiment",
        experiment,
    ]
    if manifest_path is not None:
        cmd.extend(["--manifest", str(manifest_path)])
    cmd.append("--private" if private else "--public")
    if merge_lora:
        cmd.extend(["--merge-lora", "--base-model", str(base_model)])

    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _print_sft_metrics_summary(*, output_dir: Path, gpu_peak_tflops: float | None) -> None:
    report_script = REPO_ROOT / "scripts" / "report_sft_metrics.py"
    if not report_script.exists():
        return
    cmd = [sys.executable, str(report_script), "--output-dir", str(output_dir)]
    if gpu_peak_tflops is not None:
        cmd.extend(["--gpu-peak-tflops", str(gpu_peak_tflops)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit_code={result.returncode}"
        print(f"metrics_summary_warning={detail}")
        return

    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        print(f"sft_metrics_{line}")


def _resolve_gpu_peak_tflops(explicit: float | None) -> float | None:
    if explicit is not None:
        return explicit
    env_value = os.getenv("MERA_GPU_PEAK_TFLOPS")
    if env_value is None or not env_value.strip():
        return None
    return float(env_value)


def _probe_wandb_online(timeout_seconds: float) -> tuple[bool, str]:
    timeout = max(1.0, float(timeout_seconds))
    req = urllib.request.Request("https://api.wandb.ai/graphql", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.getcode())
        return True, f"http_{status}"
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        # API is reachable for online runs in these cases; auth is handled by W&B SDK later.
        if status in (401, 404, 405):
            return True, f"http_{status}"
        return False, f"http_{status}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _resolve_wandb_offline(
    *,
    wandb_mode: str,
    allow_offline_fallback: bool,
    probe_timeout: float,
    dry_run: bool,
) -> tuple[bool, str]:
    if wandb_mode == "offline":
        return True, "forced_offline"

    if wandb_mode == "online":
        if dry_run:
            return False, "forced_online_dry_run"
        ok, detail = _probe_wandb_online(timeout_seconds=probe_timeout)
        if ok:
            return False, "forced_online_probe_ok"
        if allow_offline_fallback:
            return True, f"forced_online_fallback_offline: {detail}"
        raise RuntimeError(f"W&B online probe failed and fallback disabled: {detail}")

    env_wandb_mode = os.getenv("WANDB_MODE", "").strip().lower()
    if env_wandb_mode == "offline":
        return True, "auto_env_offline"
    if dry_run:
        return False, "auto_dry_run"
    ok, detail = _probe_wandb_online(timeout_seconds=probe_timeout)
    if ok:
        return False, "auto_probe_ok"
    if allow_offline_fallback:
        return True, f"auto_fallback_offline: {detail}"
    raise RuntimeError(f"W&B auto mode probe failed and fallback disabled: {detail}")


def main() -> None:
    args, extra_args = parse_args()

    # Allow shell patterns like --wandb-project "$WANDB_PROJECT" when the variable is unset.
    if not str(args.wandb_project or "").strip():
        args.wandb_project = "mera"
    if args.wandb_entity is not None and not str(args.wandb_entity).strip():
        args.wandb_entity = None
    if args.wandb_run_name is not None and not str(args.wandb_run_name).strip():
        args.wandb_run_name = None

    if args.batch_size < 1 or args.grad_accum < 1 or args.micro_batch_size < 1:
        raise ValueError("--batch-size, --grad-accum and --micro-batch-size must be >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.max_seq_len < 1:
        raise ValueError("--max-seq-len must be >= 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0.0, 1.0)")
    if args.save_steps < 1:
        raise ValueError("--save-steps must be >= 1")
    if args.min_free_gb_preflight < 0 or args.min_free_gb_runtime < 0:
        raise ValueError("--min-free-gb-preflight and --min-free-gb-runtime must be >= 0")
    if args.disk_watch_interval_seconds < 1:
        raise ValueError("--disk-watch-interval-seconds must be >= 1")
    if not args.config.exists():
        raise FileNotFoundError(f"SFT config template not found: {args.config}")
    if not args.dry_run and shutil.which(args.sft_entry) is None:
        raise FileNotFoundError(
            f"SFT entrypoint '{args.sft_entry}' not found on PATH. Install prime-rl or pass --sft-entry."
        )
    resolved_sft_entry = shutil.which(args.sft_entry) or args.sft_entry
    sft_workdir = _infer_sft_workdir(resolved_sft_entry)
    resolved_torchrun_bin = _infer_torchrun_bin(args.torchrun_bin, resolved_sft_entry)
    resolved_gpu_peak_tflops = _resolve_gpu_peak_tflops(args.gpu_peak_tflops)
    resolved_wandb_offline, wandb_resolution = _resolve_wandb_offline(
        wandb_mode=args.wandb_mode,
        allow_offline_fallback=args.wandb_offline_fallback,
        probe_timeout=args.wandb_online_probe_timeout,
        dry_run=args.dry_run,
    )
    if resolved_gpu_peak_tflops is not None and resolved_gpu_peak_tflops <= 0:
        raise ValueError("--gpu-peak-tflops (or MERA_GPU_PEAK_TFLOPS) must be > 0")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_min_free_space(
        output_dir,
        min_free_gb=args.min_free_gb_preflight,
        stage="preflight-before-dataset",
    )

    tasks = resolve_sft_tasks(args.task_set, args.tasks)
    dataset_missing_on_dry_run = False
    try:
        artifacts = prepare_sft_dataset_artifacts(
            output_dir=output_dir,
            data_dir=args.data_dir,
            tasks=tasks,
            limit=args.limit,
            base_model=args.model,
            max_seq_len=args.max_seq_len,
            drop_overlength=args.drop_overlength,
            rutie_context_mode=args.rutie_context_mode,
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        if not args.dry_run:
            raise
        dataset_missing_on_dry_run = True
        placeholder_dir = output_dir / "dataset"
        estimated_rows = max(1, (args.limit or 1) * max(1, len(tasks)))
        artifacts = SimpleNamespace(
            train_path=placeholder_dir / "train.jsonl",
            manifest_path=placeholder_dir / "manifest.json",
            num_rows=estimated_rows,
            dataset_stats={
                "raw_rows": estimated_rows,
                "kept_rows": estimated_rows,
                "dropped_empty": 0,
                "dropped_overlength": 0,
                "zero_trainable_before_filter": 0,
                "zero_trainable_after_filter": 0,
                "task_stats": {},
            },
        )

    nproc_per_node = _infer_nproc_per_node(args.nproc_per_node)
    if nproc_per_node > 1 and not args.dry_run:
        torchrun_exe = shutil.which(resolved_torchrun_bin)
        if torchrun_exe is None:
            raise FileNotFoundError(
                "Unable to resolve torchrun binary "
                f"'{resolved_torchrun_bin}'. Pass --torchrun-bin explicitly or ensure it is on PATH."
            )
        resolved_torchrun_bin = torchrun_exe
    global_batch_size = args.batch_size * args.grad_accum * nproc_per_node
    if global_batch_size % (nproc_per_node * args.micro_batch_size) != 0:
        raise ValueError(
            "Derived Prime-RL batch_size is incompatible with world size. Ensure "
            "(batch_size * grad_accum * nproc_per_node) is divisible by "
            "(nproc_per_node * micro_batch_size)."
        )

    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = max(1, math.ceil((artifacts.num_rows * args.epochs) / max(1, global_batch_size)))

    warmup_steps = 0
    if args.warmup_ratio > 0.0 and max_steps > 1:
        warmup_steps = max(1, int(round(max_steps * args.warmup_ratio)))

    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve()
    resolved_args_path = config_dir / "sft_overrides.txt"
    override_args = [
        "--model.name",
        args.model,
        "--model.seq-len",
        str(args.max_seq_len),
        "--model.optimization-dtype",
        "bfloat16" if args.precision == "bf16" else "float32",
        "--model.reduce-dtype",
        "bfloat16" if args.precision == "bf16" else "float32",
        "--data.name",
        str(artifacts.train_path.parent.resolve()),
        "--data.seq-len",
        str(args.max_seq_len),
        "--data.batch-size",
        str(global_batch_size),
        "--deployment.num-gpus",
        str(nproc_per_node),
        "--deployment.gpus-per-node",
        str(nproc_per_node),
        "--data.micro-batch-size",
        str(args.micro_batch_size),
        "--data.pack-function",
        args.pack_function,
        "--data.shuffle",
        "false" if args.no_shuffle else "true",
        "--data.seed",
        str(args.seed),
        "--optim.lr",
        str(args.lr),
        "--scheduler.warmup-steps",
        str(warmup_steps),
        "--ckpt.interval",
        str(args.save_steps),
        "--max-steps",
        str(max_steps),
        "--output-dir",
        str(output_dir),
        "--wandb.project",
        args.wandb_project,
        "--wandb.offline",
        "true" if resolved_wandb_offline else "false",
    ]
    if args.use_lora:
        override_args.extend(
            [
                "--model.lora.rank",
                str(args.lora_rank),
                "--model.lora.alpha",
                str(args.lora_alpha),
                "--model.lora.dropout",
                str(args.lora_dropout),
            ]
        )
    if args.wandb_run_name:
        override_args.extend(["--wandb.name", args.wandb_run_name])
    resolved_args_path.write_text(
        "\n".join(override_args) + "\n",
        encoding="utf-8",
    )

    cmd = _build_launch_command(
        torchrun_bin=resolved_torchrun_bin,
        resolved_sft_entry=resolved_sft_entry,
        config_path=config_path,
        nproc_per_node=nproc_per_node,
        master_port=args.master_port,
        override_args=override_args,
        extra_args=extra_args,
    )

    print(f"output_dir={output_dir}")
    print(f"dataset_rows={artifacts.num_rows}")
    print(f"dataset_raw_rows={artifacts.dataset_stats['raw_rows']}")
    print(f"dataset_dropped_overlength={artifacts.dataset_stats['dropped_overlength']}")
    print(f"dataset_zero_trainable_before_filter={artifacts.dataset_stats['zero_trainable_before_filter']}")
    print(f"dataset_zero_trainable_after_filter={artifacts.dataset_stats['zero_trainable_after_filter']}")
    print(f"tasks={','.join(tasks)}")
    print(f"drop_overlength={args.drop_overlength}")
    print(f"rutie_context_mode={args.rutie_context_mode}")
    if dataset_missing_on_dry_run:
        print("dataset_warning=MERA dataset not found; using placeholder dataset path for dry-run")
    print(f"nproc_per_node={nproc_per_node}")
    print(f"resolved_torchrun_bin={resolved_torchrun_bin}")
    print(f"global_batch_size={global_batch_size}")
    print(f"max_steps={max_steps}")
    print(f"config_template={config_path}")
    print(f"override_args_path={resolved_args_path}")
    print(f"sft_workdir={sft_workdir}")
    print(f"command={shlex.join(cmd)}")
    print(f"wandb_mode_requested={args.wandb_mode}")
    print(f"wandb_offline_effective={resolved_wandb_offline}")
    print(f"wandb_mode_resolution={wandb_resolution}")
    if args.min_free_gb_preflight > 0:
        print(f"min_free_gb_preflight={args.min_free_gb_preflight}")
    if args.disk_watchdog and args.min_free_gb_runtime > 0:
        print(
            f"disk_watchdog=enabled threshold_gb={args.min_free_gb_runtime} "
            f"interval_seconds={args.disk_watch_interval_seconds}"
        )
    if args.wandb_entity:
        print(f"wandb_entity_env={args.wandb_entity}")
    if args.hf_adapter_repo_id:
        print(f"hf_adapter_repo_id={args.hf_adapter_repo_id}")
    if args.hf_merged_repo_id:
        print(f"hf_merged_repo_id={args.hf_merged_repo_id}")
    if resolved_gpu_peak_tflops is not None:
        print(f"gpu_peak_tflops={resolved_gpu_peak_tflops}")
        print(f"prime_gpu_peak_flops_tflops_env={resolved_gpu_peak_tflops}")

    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        ensure_manifest(manifest_path, experiment=args.experiment, base_model=args.model)
        update_manifest(manifest_path, {"base_model": args.model})
        update_manifest(
            manifest_path,
            {
                "sft": {
                    "dataset": {
                        "output_dir": str((output_dir / "dataset").resolve()),
                        "train_path": str(artifacts.train_path.resolve()),
                        "manifest_path": str(artifacts.manifest_path.resolve()),
                        "num_rows": artifacts.num_rows,
                        "tasks": tasks,
                        "limit": args.limit,
                        "max_seq_len": args.max_seq_len,
                        "drop_overlength": args.drop_overlength,
                        "rutie_context_mode": args.rutie_context_mode,
                        "stats": artifacts.dataset_stats,
                    },
                    "run": {
                        "output_dir": str(output_dir),
                        "config_template": str(config_path),
                        "override_args_path": str(resolved_args_path),
                        "max_steps": max_steps,
                        "global_batch_size": global_batch_size,
                        "nproc_per_node": nproc_per_node,
                        "use_lora": args.use_lora,
                        "gpu_peak_tflops": resolved_gpu_peak_tflops,
                        "hf_adapter_repo_id": args.hf_adapter_repo_id,
                        "hf_merged_repo_id": args.hf_merged_repo_id,
                        "wandb": {
                            "project": args.wandb_project,
                            "entity": args.wandb_entity,
                            "name": args.wandb_run_name,
                            "mode_requested": args.wandb_mode,
                            "offline_effective": resolved_wandb_offline,
                            "resolution": wandb_resolution,
                        },
                    },
                }
            },
        )

    if args.dry_run:
        return

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")
    env["UV_HTTP_TIMEOUT"] = env.get("UV_HTTP_TIMEOUT", "120")
    env["UV_HTTP_RETRIES"] = env.get("UV_HTTP_RETRIES", "5")
    env["WANDB_MODE"] = "offline" if resolved_wandb_offline else "online"
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if resolved_gpu_peak_tflops is not None:
        env["PRIME_GPU_PEAK_FLOPS_TFLOPS"] = str(resolved_gpu_peak_tflops)
    _ensure_min_free_space(
        output_dir,
        min_free_gb=args.min_free_gb_preflight,
        stage="preflight-before-launch",
    )
    _run_with_disk_watchdog(
        cmd=cmd,
        cwd=sft_workdir,
        env=env,
        disk_path=output_dir,
        min_free_gb_runtime=args.min_free_gb_runtime,
        poll_interval_seconds=args.disk_watch_interval_seconds,
        enable_watchdog=args.disk_watchdog,
    )

    latest_weight_dir = _resolve_latest_stable_weight_dir(output_dir)
    _write_latest_pointers(output_dir, latest_weight_dir)
    print(f"latest_weight_dir={latest_weight_dir}")
    _print_sft_metrics_summary(output_dir=output_dir, gpu_peak_tflops=resolved_gpu_peak_tflops)

    manifest_path = args.manifest.expanduser().resolve() if args.manifest is not None else None
    if manifest_path is not None:
        update_manifest(
            manifest_path,
            {
                "sft": {
                    "run": {
                        "latest_weight_dir": str(latest_weight_dir.resolve()),
                        "latest_link": str((output_dir / "latest").resolve()),
                    }
                }
            },
        )

    if args.hf_adapter_repo_id:
        _run_hf_publish(
            repo_id=args.hf_adapter_repo_id,
            source_path=output_dir / "latest",
            artifact_type="adapter" if args.use_lora else "checkpoint",
            private=args.hf_private,
            manifest_path=manifest_path,
            experiment=args.experiment,
            base_model=args.model,
            merge_lora=False,
        )

    if args.hf_merged_repo_id:
        _run_hf_publish(
            repo_id=args.hf_merged_repo_id,
            source_path=output_dir / "latest",
            artifact_type="merged" if args.use_lora else "checkpoint",
            private=True,
            manifest_path=manifest_path,
            experiment=args.experiment,
            base_model=args.model,
            merge_lora=args.use_lora,
        )


if __name__ == "__main__":
    main()
