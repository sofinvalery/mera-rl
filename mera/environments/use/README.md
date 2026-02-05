# USE Environment

Overview
- Environment ID: `use`
- Description: MERA Unified State Exam (Russian language) tasks.
- Task type: single-turn mixed response

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets

Reward
- Per-example score normalized by task max score.

Quickstart
```bash
uv run vf-eval -s use
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `max_eval_examples` | int | `None` | Limit evaluation examples. |
