# ruHateSpeech Environment

Overview
- Environment ID: `ruhatespeech`
- Description: MERA hate speech diagnostic task.
- Task type: single-turn classification

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets

Reward
- Exact match on 1/2 choice.

Quickstart
```bash
uv run vf-eval -s ruhatespeech
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `max_eval_examples` | int | `None` | Limit evaluation examples. |
