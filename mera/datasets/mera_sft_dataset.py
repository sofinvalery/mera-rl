from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import datasets

_DESCRIPTION = "MERA local SFT dataset for Prime-RL"
_MANIFEST_ENV = "MERA_SFT_MANIFEST"


def _message_feature() -> datasets.Sequence:
    return datasets.Sequence(
        {
            "role": datasets.Value("string"),
            "content": datasets.Value("string"),
        }
    )


def _load_manifest() -> dict[str, Any]:
    manifest_path = os.getenv(_MANIFEST_ENV)
    if not manifest_path:
        raise ValueError(
            f"Environment variable {_MANIFEST_ENV} is required and must point to a dataset manifest JSON file."
        )
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class MeraSFTDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "prompt": _message_feature(),
                    "completion": _message_feature(),
                    "task": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        del dl_manager
        manifest = _load_manifest()
        train_file = manifest.get("train_file")
        if not train_file:
            raise ValueError("Manifest must contain 'train_file'.")

        split_gens = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": str(Path(train_file).expanduser().resolve())},
            )
        ]

        eval_file = manifest.get("eval_file")
        if eval_file:
            split_gens.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"path": str(Path(eval_file).expanduser().resolve())},
                )
            )
        return split_gens

    def _generate_examples(self, path: str) -> Iterable[tuple[int, dict[str, Any]]]:
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset split file does not exist: {data_path}")

        with data_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                prompt = row.get("prompt")
                completion = row.get("completion")
                if not isinstance(prompt, list) or not isinstance(completion, list):
                    raise ValueError(
                        f"Invalid row at index {idx}: expected 'prompt' and 'completion' as message lists."
                    )

                normalized_prompt = [
                    {
                        "role": str(message.get("role", "")),
                        "content": str(message.get("content", "")),
                    }
                    for message in prompt
                ]
                normalized_completion = [
                    {
                        "role": str(message.get("role", "")),
                        "content": str(message.get("content", "")),
                    }
                    for message in completion
                ]

                yield idx, {
                    "prompt": normalized_prompt,
                    "completion": normalized_completion,
                    "task": str(row.get("task", "")),
                }
