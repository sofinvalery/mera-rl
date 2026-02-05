# ruOpenBookQA Environment

Overview
- Environment ID: `ruopenbookqa`
- Description: MERA elementary science multiple-choice task.
- Task type: single-turn classification

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets

Reward
- Exact match on A/B/C/D.

Quickstart
```bash
uv run vf-eval -s ruopenbookqa
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `max_eval_examples` | int | `None` | Limit evaluation examples. |
