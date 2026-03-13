# ruHumanEval Environment

Overview
- Environment ID: `ruhumaneval`
- Description: MERA code generation task evaluated by unit tests.
- Task type: single-turn code completion

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets
- multiprocess

Reward
- Pass/fail based on executing tests.

Quickstart
```bash
uv run vf-eval -s ruhumaneval
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `max_eval_examples` | int | `None` | Limit evaluation examples. |
