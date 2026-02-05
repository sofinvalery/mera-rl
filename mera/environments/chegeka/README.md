# CheGeKa Environment

Overview
- Environment ID: `chegeka`
- Description: MERA open-domain QA task with free-text answers.
- Task type: single-turn generation

Datasets
- Source: MERA-evaluation/MERA (local cache or HF)
- Default split: `test`

Dependencies
- verifiers
- datasets

Reward
- Average of SQuAD-style F1 and exact match.

Quickstart
```bash
uv run vf-eval -s chegeka
```

Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `split` | str | `"test"` | Which split to load (`train`, `dev`, `test`). |
| `data_dir` | str | `None` | Override MERA data root. |
| `max_eval_examples` | int | `None` | Limit evaluation examples. |
