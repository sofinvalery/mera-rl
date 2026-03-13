from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
RUNS_ROOT = OUTPUTS_ROOT / "runs"
GENERATED_CONFIGS_ROOT = OUTPUTS_ROOT / "generated_configs"

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_SFT_MAX_SEQ_LEN = 1536
DEFAULT_SFT_BATCH_SIZE = 4
DEFAULT_SFT_GRAD_ACCUM = 2
DEFAULT_SFT_MICRO_BATCH_SIZE = 1
DEFAULT_SFT_EPOCHS = 1.0
DEFAULT_SFT_LR = 1e-5
DEFAULT_SFT_WARMUP_RATIO = 0.03
DEFAULT_SFT_SAVE_STEPS = 200
DEFAULT_SFT_LORA_RANK = 16
DEFAULT_SFT_LORA_ALPHA = 32.0
DEFAULT_SFT_LORA_DROPOUT = 0.05

DEFAULT_RL_BATCH_SIZE = 32
DEFAULT_RL_ROLLOUTS_PER_EXAMPLE = 8
DEFAULT_RL_MAX_STEPS = 400
DEFAULT_RL_MAX_TOKENS = 192
DEFAULT_RL_LEARNING_RATE = 1e-6
DEFAULT_RL_LORA_ALPHA = 32
DEFAULT_RL_CHECKPOINT_INTERVAL = 100
DEFAULT_RL_KEEP_CLOUD = 5

DEFAULT_EVAL_NUM_EXAMPLES = 100
DEFAULT_EVAL_ROLLOUTS_PER_EXAMPLE = 1
DEFAULT_SMOKE_NUM_EXAMPLES = 5

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

FAIR_SFT_SPLITS = {
    "chegeka": "train",
    "lcs": "public_test",
    "mamuramu": "train",
    "mathlogicqa": "train",
    "multiq": "train",
    "parus": "train",
    "rcb": "train",
    "rumodar": "public_test",
    "rumultiar": "train",
    "ruopenbookqa": "train",
    "rutie": "train",
    "ruworldtree": "train",
    "rwsd": "train",
    "use": "train",
}

FAIR_EVAL_TASKS = [
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

SMOKE_EVAL_TASKS = [
    "parus",
    "rcb",
    "rwsd",
    "use",
]

HF_TASKS = [
    "bps",
    "chegeka",
    "lcs",
    "mamuramu",
    "mathlogicqa",
    "multiq",
    "parus",
    "rcb",
    "rucodeeval",
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

SFT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
