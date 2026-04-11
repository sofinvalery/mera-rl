# MERA Benchmark Sync

This directory captures the MERA benchmark-side changes used for local and remote
evaluation runs alongside this `mera-rl` workspace.

It is not a full vendored copy of the upstream
[`MERA-Evaluation/MERA`](https://github.com/MERA-Evaluation/MERA) repository.
Instead, it stores:

- patched benchmark launch scripts
- the patched `lm_eval.models.vllm_causallms` implementation
- the operational benchmark workflow that worked on the 2x5090 machines

## Synced Files

- `scripts/run_benchmark.sh`
- `scripts/run_benchmark_all.sh`
- `scripts/run_benchmark_gen.sh`
- `scripts/run_benchmark_all_localscore.sh`
- `scripts/run_benchmark_fixed.sh`
- `lm-eval-patches/lm_eval/models/vllm_causallms.py`

## What Changed

### Benchmark scripts

- persistent dataset cache via `MERA_DATASETS_CACHE`
- no per-run dataset cache deletion by default
- `run_benchmark_fixed.sh` bootstrap for uv-managed installs and cache warmup
- `all_local` mode for `*_localscore` tasks
- vLLM defaults tuned for 2x RTX 5090:
  - `tensor_parallel_size=2`
  - `data_parallel_size=1`
  - `gpu_memory_utilization=0.50`
  - `max_model_len=32768`
  - `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`

### Harness patch

`vllm_causallms.py` carries compatibility fixes for the newer vLLM runtime:

- remove dependency on missing `vllm.utils.get_open_port`
- add local free-port selection
- keep tokenizer import path working
- filter unsupported `EngineArgs` kwargs against the installed vLLM signature

## Official MERA vs Local Metrics

There are two distinct workflows:

1. Submission workflow:
   - use `run_benchmark_fixed.sh --mode all`
   - MERA tasks run with `--predict_only`
   - `bypass=999` in harness JSON is expected
   - the useful artifacts are the submission ZIPs

2. Local metric workflow:
   - use `run_benchmark_fixed.sh --mode all_local`
   - runs the repo's `*_localscore` tasks
   - useful for comparing models locally

## Apply To A MERA Checkout

Use `benchmarks/mera/apply_to_checkout.sh` to copy these synced files into a
fresh upstream MERA clone.

Example:

```bash
cd /Users/valeriysofin/Developer/mera-rl
benchmarks/mera/apply_to_checkout.sh /path/to/MERA
```

## Dependency Notes

The working bootstrap path is encoded in `scripts/run_benchmark_fixed.sh` and
uses uv with the MERA clone's own `.venv`:

- `torch`, `torchvision` from the CUDA 12.8 index
- `lm-evaluation-harness.[vllm]`
- `transformers<5`
- `ray`
- `accelerate`
- `datasets`
- `sentencepiece`
- `protobuf`
- `evaluate`
- `sacrebleu`
- `huggingface_hub`
- `omegaconf`
- `boto3`
- `scikit-learn`
