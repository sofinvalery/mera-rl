This repo provides:
- MERA `verifiers` environments (one package per MERA task) under `mera/environments/*`
- Prime-RL configs for running GRPO on those environments under `mera/configs/prime_rl/*`
- Small wrapper scripts under `mera/scripts/*`

The intended runtime environment is the Prime-RL venv (Python 3.12, Linux + NVIDIA).

## Layout

- Prime-RL repo: `_deps/prime-rl`
- MERA repo (scoring + calibration artifacts): `_deps/MERA`

## Quick Setup (Linux)

Clone with submodules (recommended):

```bash
git clone --recurse-submodules <this-repo-url>
cd mera-rl
```

Or if already cloned:

```bash
git submodule update --init --recursive
```

Then run the setup script (Debian/Ubuntu):

```bash
bash mera/tools/remote_setup.sh --repo-dir .
```

That will:
- ensure `uv` is installed
- ensure `_deps/MERA` and `_deps/prime-rl` are present (clone/pull if needed)
- create the Prime-RL venv at `_deps/prime-rl/.venv` and `uv sync`
- install MERA task environments into the Prime-RL venv (editable)

## Running GRPO

```bash
python mera/scripts/grpo.py bps --dry-run
python mera/scripts/grpo.py bps
```

## Running Eval

Use the Prime-RL venv Python:

```bash
_deps/prime-rl/.venv/bin/python mera/scripts/eval.py --limit 50 --tensor-parallel 1
```

If you only want submission files (no MERA scoring), add `--skip-scoring`.
