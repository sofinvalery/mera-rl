# ruModAr Environment

Overview
- Environment ID: `rumodar`
- Description: MERA modified arithmetic task.
- Task type: single-turn numeric response

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets

Reward
- Exact match on numeric output.

Quickstart
```bash
uv run vf-eval -s rumodar
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `max_eval_examples` | int | `None` | Limit evaluation examples. |
