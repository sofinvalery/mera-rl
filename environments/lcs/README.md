# LCS Environment

Overview
- Environment ID: `lcs`
- Description: MERA Longest Common Subsequence length task.
- Task type: single-turn numeric response

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets

Reward
- Exact match on the numeric length.

Quickstart
```bash
uv run vf-eval -s lcs
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `max_eval_examples` | int | `None` | Limit evaluation examples. |
