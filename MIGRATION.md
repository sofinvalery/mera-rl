# Migration Notes: `mera-rl` -> `prime-lab-rl`

## Scope

This migration switches from local Prime-RL execution in:

- `/Users/valeriysofin/Developer/mera-rl/mera/scripts/grpo.py`
- `/Users/valeriysofin/Developer/mera-rl/run_grpo_task.sh`
- `/Users/valeriysofin/Developer/mera-rl/run_all_grpo.sh`
- `/Users/valeriysofin/Developer/mera-rl/mera/configs/prime_rl/*`

to hosted execution with:

- `prime rl run <config.toml>`
- generated configs from `scripts/render_hosted_config.py`

## Key behavioral mapping

1. Environment IDs
- Old local flow used local package ids like `mathlogicqa`.
- Hosted flow uses hub ids: `<owner>/<task>` (for example `my-team/mathlogicqa`).
- Publish environments first with `scripts/push_envs.sh`.

2. Split policy
- Old GRPO fair guard allowed only `train`/`public_test`, unless `--allow-test-split`.
- New hosted wrapper enforces the same rule in config generation.
- `rucodeeval` remains non-fair (`test`) and requires `ALLOW_NONFAIR=1`.

3. Hyperparameters
- Old wrapper defaults:
  - `orchestrator.batch-size=32`
  - `orchestrator.rollouts-per-example=8`
  - `orchestrator.max-steps=400`
  - `orchestrator.sampling.max-tokens=192`
- New hosted defaults in `configs/rl/mera/task_map.toml` match these values.

4. Run lifecycle
- Old flow launched local processes in `_deps/prime-rl`.
- New flow submits managed hosted runs and uses:
  - `prime rl get <run-id>`
  - `prime rl logs <run-id> -f`
  - dashboard URL returned by `prime rl run`.

## Not carried over

- Local SFT wrapper execution (`mera/scripts/sft.py`, `run_sft_fast.sh`).
- Local vLLM eval wrappers (`mera/scripts/eval.py`, `run_eval*.sh`).

This repository is intentionally focused on hosted RL orchestration.
