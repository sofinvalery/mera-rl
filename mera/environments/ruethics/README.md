# ruEthics Environment

Overview
- Environment ID: `ruethics`
- Description: MERA ethics diagnostic task.
- Task type: single-turn classification

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets

Reward
- Proxy reward based on majority of ethical criteria.

Quickstart
```bash
uv run vf-eval -s ruethics
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `max_eval_examples` | int | `None` | Limit evaluation examples. |
