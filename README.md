This repo provides:
- MERA task environments under `mera/environments/*`
- Prime-RL configs for GRPO under `mera/configs/prime_rl/*`
- Wrapper scripts under `mera/scripts/*` for SFT, GRPO, and evals

## Layout

- MERA evaluation repo (scoring + calibration artifacts): `_deps/MERA`
- Prime-RL repo (trainer/orchestrator/inference): `_deps/prime-rl`

## Setup (Ubuntu)

Clone with submodules:

```bash
git clone --recurse-submodules <this-repo-url>
cd mera-rl
```

Then run:

```bash
bash mera/tools/remote_setup.sh --repo-dir .
```

This creates two virtual environments:

- MERA venv (Python 3.11): `.venv`
- Prime-RL venv (Python 3.12): `_deps/prime-rl/.venv`

It also writes `env.sh` with local Hugging Face cache settings.

## Runtime recommendation (RTX 5090)

Use Prime-RL Python for all stages (eval, SFT, GRPO):

```bash
source env.sh
PY=_deps/prime-rl/.venv/bin/python
```

Default base model across scripts/configs: `Qwen/Qwen3-4B-Instruct-2507`.

## Fair split policy

Default GRPO behavior is strict fair mode:
- allowed splits: `train`, `public_test`
- blocked by default: `test`, `validation`

To run exploratory/non-fair training (for example, `rucodeeval` which only has `test`), pass:

```bash
--allow-test-split
```

SFT defaults to fair task set (`--task-set fair`) and uses per-task fair splits:
- `lcs`: `public_test`
- `rumodar`: `public_test`
- others in fair set: `train`

## Pipeline (recommended)

1) Base eval

```bash
$PY mera/scripts/eval.py --model <base_model> --tensor-parallel 2 --output-dir outputs/eval_base
```

2) SFT

```bash
$PY mera/scripts/sft.py --model <base_model> --task-set fair --epochs 1 --batch-size 4 --grad-accum 2 --nproc-per-node 2 --output-dir outputs/sft
```

Prime-RL SFT writes checkpoints to `outputs/sft/weights/step_<N>` and also creates `outputs/sft/latest`.
LoRA in SFT is opt-in via `--use-lora` (default is full-parameter SFT).

3) Intermediate eval after SFT (recommended)

```bash
$PY mera/scripts/eval.py --model outputs/sft/latest --task-set benchmark --tensor-parallel 2 --output-dir outputs/eval_sft
```

For a fast proxy eval on validation-only subset:

```bash
$PY mera/scripts/eval_base.py --model outputs/sft/latest --task-set validation --tasks parus rcb rwsd use --split validation --skip-scoring --output-dir outputs/eval_sft_val
```

4) GRPO

```bash
$PY mera/scripts/grpo.py mathlogicqa --trainer-gpu-ids "[1]" --inference-gpu-ids "[0]" --trainer.model.name outputs/sft/latest --orchestrator.model.name outputs/sft/latest --inference.model.name outputs/sft/latest
```

5) Post-GRPO eval

```bash
$PY mera/scripts/eval.py --model outputs/grpo_prime/mathlogicqa/weights/step_<N> --task-set benchmark --tensor-parallel 2 --output-dir outputs/eval_post
```

## Script notes

- `mera/scripts/sft.py`
  - defaults to fair task preset
  - uses Prime-RL native SFT trainer
  - exports checkpoints to `<output_dir>/weights/step_<N>` and updates `<output_dir>/latest`
- `mera/scripts/grpo.py`
  - enforces fair splits by default
  - supports explicit override via `--allow-test-split`
- `mera/scripts/eval.py` / `mera/scripts/eval_base.py`
  - support task presets via `--task-set {all,benchmark,validation}`
  - support custom subsets via `--tasks ...`
