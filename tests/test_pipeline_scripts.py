from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prime_lab_rl.sft_dataset import prepare_sft_dataset_artifacts


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
