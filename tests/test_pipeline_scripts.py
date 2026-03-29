from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.sft_dataset import prepare_sft_dataset_artifacts


class _FakeTokenizer:
    eos_token_id = -1

    def apply_chat_template(self, messages, add_generation_prompt: bool = False, return_dict: bool = False):
        del add_generation_prompt, return_dict
        token_count = 0
        for message in messages:
            content = str(message.get("content", "")).strip()
            token_count += max(1, len(content.split())) if content else 1
        return list(range(token_count))


def test_prepare_sft_dataset_artifacts(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "mera-data"
    task_dir = data_root / "chegeka"
    task_dir.mkdir(parents=True)
    record = {
        "instruction": "Ответь на вопрос: {question}",
        "inputs": {"question": "Столица России?"},
        "outputs": "Москва",
        "meta": {"id": 1},
    }
    (task_dir / "train.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setenv("MERA_DATA_DIR", str(data_root))
    monkeypatch.setattr("prime_lab_rl.sft_dataset._load_tokenizer", lambda _: _FakeTokenizer())
    artifacts = prepare_sft_dataset_artifacts(
        output_dir=tmp_path / "out",
        data_dir=None,
        tasks=["chegeka"],
        limit=None,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )

    train_rows = artifacts.train_path.read_text(encoding="utf-8").strip().splitlines()
    assert artifacts.num_rows == 1
    assert len(train_rows) == 1
    payload = json.loads(train_rows[0])
    assert payload["task"] == "chegeka"
    assert payload["completion"][0]["content"] == "Москва"

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["tasks"] == ["chegeka"]
    assert manifest["task_splits"] == {"chegeka": "train"}
    assert manifest["dataset_stats"]["raw_rows"] == 1
    assert manifest["dataset_stats"]["kept_rows"] == 1


def test_prepare_sft_dataset_artifacts_rutie_defaults_to_single_turn(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "mera-data"
    task_dir = data_root / "rutie"
    task_dir.mkdir(parents=True)
    records = [
        {
            "instruction": "CTX:{context}\nQ:{question}",
            "inputs": {"question": "Сколько ног у человека?", "choice1": "Две", "choice2": "Четыре"},
            "outputs": "1",
            "meta": {"dialog_id": 0, "question_id": 1},
        },
        {
            "instruction": "CTX:{context}\nQ:{question}",
            "inputs": {"question": "Сколько ног у муравья?", "choice1": "Две", "choice2": "Шесть"},
            "outputs": "2",
            "meta": {"dialog_id": 0, "question_id": 2},
        },
    ]
    (task_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MERA_DATA_DIR", str(data_root))
    monkeypatch.setattr("prime_lab_rl.sft_dataset._load_tokenizer", lambda _: _FakeTokenizer())
    artifacts = prepare_sft_dataset_artifacts(
        output_dir=tmp_path / "out",
        data_dir=None,
        tasks=["rutie"],
        limit=None,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )

    rows = [json.loads(line) for line in artifacts.train_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2
    assert "Ответ:" not in rows[1]["prompt"][0]["content"]


def test_prepare_sft_dataset_artifacts_rutie_rolling_mode(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "mera-data"
    task_dir = data_root / "rutie"
    task_dir.mkdir(parents=True)
    records = [
        {
            "instruction": "CTX:{context}\nQ:{question}",
            "inputs": {"question": "Сколько ног у человека?", "choice1": "Две", "choice2": "Четыре"},
            "outputs": "1",
            "meta": {"dialog_id": 0, "question_id": 1},
        },
        {
            "instruction": "CTX:{context}\nQ:{question}",
            "inputs": {"question": "Сколько ног у муравья?", "choice1": "Две", "choice2": "Шесть"},
            "outputs": "2",
            "meta": {"dialog_id": 0, "question_id": 2},
        },
    ]
    (task_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MERA_DATA_DIR", str(data_root))
    monkeypatch.setattr("prime_lab_rl.sft_dataset._load_tokenizer", lambda _: _FakeTokenizer())
    artifacts = prepare_sft_dataset_artifacts(
        output_dir=tmp_path / "out",
        data_dir=None,
        tasks=["rutie"],
        limit=None,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        rutie_context_mode="rolling",
    )

    rows = [json.loads(line) for line in artifacts.train_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2
    assert "Ответ:" in rows[1]["prompt"][0]["content"]


def test_prepare_sft_dataset_artifacts_drop_overlength(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "mera-data"
    task_dir = data_root / "chegeka"
    task_dir.mkdir(parents=True)
    records = [
        {
            "instruction": "{text}",
            "inputs": {"text": "короткий текст"},
            "outputs": "ответ",
            "meta": {"id": 1},
        },
        {
            "instruction": "{text}",
            "inputs": {"text": "очень " * 64 + "длинный текст"},
            "outputs": "ответ",
            "meta": {"id": 2},
        },
    ]
    (task_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MERA_DATA_DIR", str(data_root))
    monkeypatch.setattr("prime_lab_rl.sft_dataset._load_tokenizer", lambda _: _FakeTokenizer())
    artifacts = prepare_sft_dataset_artifacts(
        output_dir=tmp_path / "out",
        data_dir=None,
        tasks=["chegeka"],
        limit=None,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        max_seq_len=8,
        drop_overlength=True,
    )

    rows = [json.loads(line) for line in artifacts.train_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert artifacts.dataset_stats["dropped_overlength"] == 1
    assert artifacts.dataset_stats["zero_trainable_before_filter"] == 1
    assert artifacts.dataset_stats["zero_trainable_after_filter"] == 0


def test_render_eval_config_script(tmp_path: Path) -> None:
    endpoints_path = tmp_path / "endpoints.toml"
    endpoints_path.write_text("", encoding="utf-8")
    output_path = tmp_path / "eval.toml"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_eval_config.py"),
        "--output",
        str(output_path),
        "--endpoint-id",
        "mera-test",
        "--endpoints-path",
        str(endpoints_path),
        "--task-set",
        "smoke",
        "--num-examples",
        "7",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    rendered = output_path.read_text(encoding="utf-8")
    assert 'endpoint_id = "mera-test"' in rendered
    assert 'env_id = "parus"' in rendered
    assert "num_examples = 7" in rendered


def test_render_hosted_config_supports_rl_extensions(tmp_path: Path) -> None:
    output_path = tmp_path / "rl.toml"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_hosted_config.py"),
        "--task",
        "mathlogicqa",
        "--owner",
        "demo-owner",
        "--output",
        str(output_path),
        "--model",
        "demo/sft-merged",
        "--learning-rate",
        "1e-6",
        "--lora-alpha",
        "32",
        "--checkpoint-interval",
        "50",
        "--checkpoint-keep-cloud",
        "3",
        "--eval-interval",
        "25",
        "--eval-num-examples",
        "8",
        "--eval-rollouts-per-example",
        "1",
        "--eval-base-model",
    ]
    env = os.environ.copy()
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    rendered = output_path.read_text(encoding="utf-8")
    assert 'model = "demo/sft-merged"' in rendered
    assert "learning_rate = 1e-06" in rendered
    assert "lora_alpha = 32" in rendered
    assert "[checkpoints]" in rendered
    assert "interval = 50" in rendered
    assert "keep_cloud = 3" in rendered
    assert "[eval]" in rendered
    assert "eval_base_model = true" in rendered


def test_report_sft_metrics_script(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "[default0]:12:00:00 SUCCESS Step 10 | Time: 40.00s | Loss: 1.2 | Grad. Norm: 5.0 | LR: 2.00e-05 | Throughput: 1200 tokens/s | MFU: 5.0% | Peak Mem.: 29/31 GiB",
                "[default0]:12:00:40 SUCCESS Step 11 | Time: 38.00s | Loss: 1.1 | Grad. Norm: 4.8 | LR: 2.00e-05 | Throughput: 1300 tokens/s | MFU: 5.5% | Peak Mem.: 29/31 GiB",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "report_sft_metrics.py"),
        "--log-file",
        str(log_path),
        "--gpu-peak-tflops",
        "165",
    ]
    result = subprocess.run(cmd, check=True, cwd=REPO_ROOT, capture_output=True, text=True)
    assert "mfu_raw_pct_median=5.250" in result.stdout
    assert "mfu_corrected_pct_median=" in result.stdout


def test_run_rlvr_local_smoke_dry_run() -> None:
    experiment = "test_rlvr_smoke"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_rlvr_local.py"),
        "smoke",
        "--experiment",
        experiment,
        "--dry-run",
    ]
    env = os.environ.copy()
    env["PRIME_RL_ENTRY"] = "/bin/echo"
    result = subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert "mode=smoke" in result.stdout
    assert "command=/bin/echo @" in result.stdout
    assert "--wandb." not in result.stdout
    assert "wandb_mode_effective=disabled" in result.stdout
    config_path = REPO_ROOT / "outputs" / "runs" / experiment / "configs" / "rlvr_smoke.toml"
    assert config_path.exists()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "wandb" not in config
    env_ids = [entry["id"] for entry in config["orchestrator"]["env"]]
    assert len(env_ids) == 14


def test_run_rlvr_local_train_dry_run_without_wandb_flags() -> None:
    experiment = "test_rlvr_train"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_rlvr_local.py"),
        "train",
        "--experiment",
        experiment,
        "--dry-run",
    ]
    env = os.environ.copy()
    env["PRIME_RL_ENTRY"] = "/bin/echo"
    result = subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert "mode=train" in result.stdout
    assert "command=/bin/echo @" in result.stdout
    assert "--wandb." not in result.stdout
    assert "wandb_mode_effective=offline" in result.stdout
    config_path = REPO_ROOT / "outputs" / "runs" / experiment / "configs" / "rlvr_train.toml"
    assert config_path.exists()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert config["wandb"]["offline"] is True
    assert config["wandb"]["shared"] is False


def test_run_rlvr_local_train_dry_run_with_online_mode_override() -> None:
    experiment = "test_rlvr_train_online_override"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_rlvr_local.py"),
        "train",
        "--experiment",
        experiment,
        "--dry-run",
    ]
    env = os.environ.copy()
    env["PRIME_RL_ENTRY"] = "/bin/echo"
    env["MERA_RLVR_WANDB_MODE"] = "online"
    result = subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert "mode=train" in result.stdout
    assert "wandb_mode_requested=online" in result.stdout
    assert "wandb_mode_effective=online" in result.stdout
    assert "--wandb." not in result.stdout
    config_path = REPO_ROOT / "outputs" / "runs" / experiment / "configs" / "rlvr_train.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert config["wandb"]["offline"] is False


def test_run_rlvr_local_smoke_dry_run_model_dir_injection(tmp_path: Path) -> None:
    experiment = "test_rlvr_smoke_model_dir"
    model_dir = tmp_path / "cached-model"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_rlvr_local.py"),
        "smoke",
        "--experiment",
        experiment,
        "--dry-run",
    ]
    env = os.environ.copy()
    env["PRIME_RL_ENTRY"] = "/bin/echo"
    env["MERA_RLVR_MODEL_DIR"] = str(model_dir)
    result = subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert "model_source=dry_run_not_downloaded" in result.stdout
    config_path = REPO_ROOT / "outputs" / "runs" / experiment / "configs" / "rlvr_smoke.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert config["model"]["name"] == str(model_dir.resolve())
