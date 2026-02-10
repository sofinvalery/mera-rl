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

It also writes `env.sh` with local Hugging Face cache settings (defaults to `<repo>/.hf`).

## Running

Activate the MERA venv and load environment variables:

```bash
source .venv/bin/activate
source env.sh
```

### SFT

```bash
python mera/scripts/sft.py --limit 200 --epochs 1 --batch-size 1 --grad-accum 8
```

### GRPO (Prime-RL)

`grpo.py` is a thin wrapper that runs Prime-RL via `uv run` inside `_deps/prime-rl`.

```bash
python mera/scripts/grpo.py bps --dry-run
python mera/scripts/grpo.py bps
```

### Evals

Generate a submission zip without scoring:

```bash
python mera/scripts/eval_base.py --limit 50 --tensor-parallel 1 --skip-scoring
```

If you have the MERA dataset cached (or set `MERA_DATA_DIR`), you can run full scoring via:

```bash
python mera/scripts/eval.py --limit 50 --tensor-parallel 1
```
