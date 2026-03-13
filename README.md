# prime-lab-rl

MERA workspace for:

- local Prime-RL SFT on 2 GPUs
- hosted Prime RL submission per MERA task
- Prime-eval-driven baseline/intermediate/final evaluation stages
- Hugging Face publication for SFT and RL artifacts

The repo keeps the hosted-RL migration work that already existed, and adds the missing SFT/eval/artifact pipeline around it.

## Design

The SFT path now follows the upstream `prime-rl` example style from `reverse_text` and `wordle`:

- a checked-in SFT config template lives in `configs/sft/mera-fair.toml`
- MERA-specific preprocessing is isolated in a dataset builder
- the launcher is thin and calls the official `sft` entrypoint with CLI overrides

That means the custom logic in this repo is mostly:

- building a MERA SFT dataset from fair splits
- keeping a pipeline manifest under `outputs/runs/<experiment>/manifest.json`
- rendering Prime eval configs and capturing the resulting run directories
- rendering hosted RL configs with SFT warm-start, LoRA, and checkpoint retention
- downloading hosted checkpoints and publishing them to HF

## Upstream References

- Prime RL docs: https://docs.primeintellect.ai/prime-rl
- Hosted training: https://docs.primeintellect.ai/hosted-training/getting-started
- Recipes: https://docs.primeintellect.ai/guides/recipes
- Examples:
  - https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/reverse_text
  - https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/wordle
  - https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/alphabet_sort/README.md
  - https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/wiki_search
  - https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/hendrycks_sanity/

## One-Time Setup

1. Install the Prime CLI and authenticate:

```bash
uv tool install -U prime
prime login
```

2. Install the MERA environments locally for evals:

```bash
prime env install chegeka lcs mamuramu mathlogicqa multiq parus rcb rumodar rumultiar ruopenbookqa rutie ruworldtree rwsd use -p ./environments
```

3. Make sure the MERA dataset is locally available. The dataset builder looks in:

- `MERA_DATA_DIR`
- `HF_HOME`
- `~/.cache/huggingface`

4. Install Prime-RL locally on the 2x5090 machine so the `sft` entrypoint is on `PATH`.

## Manual Setup

Local secrets/defaults can be staged from:

```bash
configs/local.env.example
```

Typical local setup:

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=mera
export WANDB_ENTITY=<wandb-entity>

export HF_TOKEN=...
# or:
export HUGGINGFACE_HUB_TOKEN=...

export PRIME_ENV_OWNER=<owner>
```

Notes:

- Local SFT reads W&B and HF credentials from the shell environment.
- Hosted RL submission can forward secrets with `prime rl run --env-var ...`; the pipeline wrapper exposes this as `--rl-env-var`.
- Final HF uploads after SFT or after hosted RL use your local HF token, not a hosted secret.

## SFT

The checked-in SFT template is:

```bash
configs/sft/mera-fair.toml
```

The local launcher builds a MERA JSONL dataset and then runs the official Prime-RL `sft @ <config>` flow with overrides:

```bash
python3 scripts/run_sft_local.py \
  --output-dir outputs/runs/mera_01/sft \
  --manifest outputs/runs/mera_01/manifest.json \
  --experiment mera_01 \
  --wandb-project mera \
  --wandb-entity <wandb-entity> \
  --hf-adapter-repo-id <user-or-org>/mera-qwen3-4b-sft \
  --hf-merged-repo-id <user-or-org>/mera-qwen3-4b-sft-merged
```

Key defaults:

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- fair SFT tasks: `chegeka lcs mamuramu mathlogicqa multiq parus rcb rumodar rumultiar ruopenbookqa rutie ruworldtree rwsd use`
- LoRA enabled by default
- sequence length: `1536`
- micro batch size: `1`
- effective batch semantics equivalent to the previous `4 x 2 GPUs x grad_accum 2`

Dataset-only preprocessing is also exposed directly:

```bash
python3 scripts/build_sft_dataset.py \
  --output-dir outputs/runs/mera_01/sft \
  --manifest outputs/runs/mera_01/manifest.json \
  --experiment mera_01
```

After training, the launcher resolves the latest stable checkpoint and writes:

- `outputs/.../latest`
- `outputs/.../LATEST_WEIGHT_STEP`

If `--hf-adapter-repo-id` is set, it automatically uploads the final adapter after SFT.

If `--hf-merged-repo-id` is set, it also produces and uploads a merged full-model handoff artifact for hosted RL warm-start.

### Tiny SFT Smoke Test

For a cheap end-to-end local SFT check on a very small subset, use:

```bash
scripts/run_sft_test.sh
```

Default smoke settings:

- tasks: `chegeka mathlogicqa rcb`
- per-task limit: `16`
- max steps: `8`
- sequence length: `1024`

Useful overrides:

```bash
LIMIT=8 MAX_STEPS=4 scripts/run_sft_test.sh
TASKS="chegeka rcb" scripts/run_sft_test.sh --dry-run
HF_ADAPTER_REPO_ID=<user-or-org>/mera-qwen3-4b-sft-test scripts/run_sft_test.sh
```

## Evaluation

The repo includes reusable eval suites:

- `configs/eval/mera-fair-suite.toml`
- `configs/eval/mera-fair-smoke.toml`

The stage wrapper renders a concrete config with endpoint information, installs the local envs if needed, runs `prime eval`, and writes a stage summary back into the pipeline manifest:

```bash
python3 scripts/run_eval_stage.py \
  --stage baseline \
  --experiment mera_01 \
  --manifest outputs/runs/mera_01/manifest.json \
  --run-root outputs/runs/mera_01 \
  --endpoint-id my-endpoint
```

For a custom OpenAI-compatible endpoint instead of an alias:

```bash
python3 scripts/run_eval_stage.py \
  --stage intermediate \
  --experiment mera_01 \
  --manifest outputs/runs/mera_01/manifest.json \
  --run-root outputs/runs/mera_01 \
  --endpoint-url http://localhost:8000/v1 \
  --model-name my-served-model \
  --api-key-env OPENAI_API_KEY
```

Each stage writes:

- `outputs/runs/<experiment>/evaluations/<stage>/summary.json`
- `outputs/runs/<experiment>/evaluations/<stage>/comparison.json`

## Hosted RL

The hosted RL generator now supports:

- SFT-derived `model` override
- `learning_rate`
- `lora_alpha`
- checkpoint interval / keep-cloud retention
- optional online eval section
- W&B metadata

Render a single task:

```bash
python3 scripts/render_hosted_config.py \
  --task mathlogicqa \
  --owner <owner> \
  --output outputs/generated_configs/mathlogicqa.toml \
  --model <hf-model-or-handoff-model>
```

Submit a single task with the shell wrapper:

```bash
MODEL=<hf-model-or-handoff-model> \
WANDB_PROJECT=mera \
WANDB_ENTITY=<wandb-entity> \
scripts/run_task_hosted.sh mathlogicqa <owner>
```

Or launch the whole fair set:

```bash
MODEL=<hf-model-or-handoff-model> \
WANDB_PROJECT=mera \
WANDB_ENTITY=<wandb-entity> \
scripts/run_all_hosted.sh fair <owner>
```

## Hugging Face Publication

Upload an SFT adapter:

```bash
python3 scripts/publish_hf_artifact.py \
  --kind sft \
  --source-path outputs/runs/mera_01/sft/latest \
  --repo-id <user-or-org>/mera-qwen3-4b-sft \
  --artifact-type adapter \
  --manifest outputs/runs/mera_01/manifest.json \
  --experiment mera_01
```

Merge a LoRA adapter into a private handoff model for hosted RL:

```bash
python3 scripts/publish_hf_artifact.py \
  --kind sft \
  --source-path outputs/runs/mera_01/sft/latest \
  --repo-id <user-or-org>/mera-qwen3-4b-sft-merged \
  --artifact-type merged \
  --merge-lora \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --manifest outputs/runs/mera_01/manifest.json \
  --experiment mera_01
```

Download the latest READY hosted checkpoint:

```bash
python3 scripts/download_hosted_checkpoint.py \
  --run-id <run-id> \
  --output-dir outputs/runs/mera_01/rl_checkpoints/mathlogicqa \
  --task mathlogicqa \
  --manifest outputs/runs/mera_01/manifest.json \
  --experiment mera_01 \
  --extract
```

## Pipeline Runner

The stage orchestrator is:

```bash
python3 scripts/run_mera_pipeline.py
```

Main stages:

- `pre-rl`: baseline eval -> local SFT -> SFT publish -> intermediate eval
- `submit-rl`: submit hosted RL runs for the selected MERA tasks
- `wait-rl`: poll hosted RL runs until terminal status
- `finalize`: download hosted checkpoints -> publish to HF -> final eval
- `all`: run every stage in order

Example:

```bash
python3 scripts/run_mera_pipeline.py \
  --stage pre-rl \
  --experiment mera_01 \
  --owner <owner> \
  --baseline-endpoint-id <base-endpoint> \
  --intermediate-endpoint-id <sft-endpoint> \
  --sft-adapter-repo-id <user-or-org>/mera-qwen3-4b-sft \
  --sft-handoff-repo-id <user-or-org>/mera-qwen3-4b-sft-merged \
  --wandb-project mera \
  --wandb-entity <wandb-entity>
```

Then:

```bash
python3 scripts/run_mera_pipeline.py \
  --stage submit-rl \
  --experiment mera_01 \
  --owner <owner> \
  --sft-handoff-repo-id <user-or-org>/mera-qwen3-4b-sft-merged \
  --wandb-project mera \
  --wandb-entity <wandb-entity> \
  --rl-env-var WANDB_API_KEY
```

And after hosted runs finish:

```bash
python3 scripts/run_mera_pipeline.py \
  --stage finalize \
  --experiment mera_01 \
  --final-endpoint-id <final-endpoint> \
  --final-rl-repo-prefix <user-or-org>/mera-qwen3-4b-rl
```

## Validation

Run:

```bash
pytest -q
```

Current test coverage verifies:

- environment metadata sanity
- MERA SFT dataset preprocessing
- eval config rendering
- hosted RL config rendering with LoRA/checkpoint extensions
