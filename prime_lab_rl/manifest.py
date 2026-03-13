from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def default_manifest(experiment: str, base_model: str) -> dict[str, Any]:
    return {
        "version": 1,
        "experiment": experiment,
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "base_model": base_model,
        "sft": {},
        "evaluations": {},
        "rl_runs": {},
        "hf": {},
        "notes": {},
    }


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(payload)
    payload["updated_at_utc"] = utc_now_iso()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def ensure_manifest(path: Path, experiment: str, base_model: str) -> dict[str, Any]:
    if path.exists():
        manifest = load_manifest(path)
        if "experiment" not in manifest:
            manifest["experiment"] = experiment
        if "base_model" not in manifest:
            manifest["base_model"] = base_model
        save_manifest(path, manifest)
        return manifest

    manifest = default_manifest(experiment=experiment, base_model=base_model)
    save_manifest(path, manifest)
    return manifest


def update_manifest(path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    manifest = load_manifest(path) if path.exists() else {}
    manifest = _deep_merge(manifest, patch)
    save_manifest(path, manifest)
    return manifest
