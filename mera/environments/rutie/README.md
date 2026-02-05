# ruTiE Environment

Overview
- Environment ID: `rutie`
- Description: MERA dialogue context task.
- Task type: multi-turn dialogue

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets

Reward
- Average accuracy over turns in a dialogue.

Quickstart
```bash
uv run vf-eval -s rutie
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `max_eval_examples` | int | `None` | Limit evaluation dialogs. |
