# ruDetox Environment

Overview
- Environment ID: `rudetox`
- Description: MERA detoxification task with style/meaning/fluency scoring.
- Task type: single-turn generation

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets
- transformers
- torch
- scikit-learn

Reward
- Default: joint score (STA * SIM * FL).
- Fast mode: STA only (use `fast=True`).

Quickstart
```bash
uv run vf-eval -s rudetox
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `fast` | bool | `False` | Use STA-only reward. |
| `max_eval_examples` | int | `None` | Limit evaluation examples. |
