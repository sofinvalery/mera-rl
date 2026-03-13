from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


def ensure_repo(
    repo_id: str,
    *,
    private: bool = True,
    repo_type: str = "model",
    api: HfApi | None = None,
) -> None:
    client = api or HfApi()
    client.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)


def upload_folder(
    source_path: Path,
    *,
    repo_id: str,
    path_in_repo: str = ".",
    private: bool = True,
    commit_message: str,
    revision: str | None = None,
    ignore_patterns: list[str] | None = None,
    api: HfApi | None = None,
) -> dict[str, Any]:
    client = api or HfApi()
    ensure_repo(repo_id, private=private, api=client)
    commit_info = client.upload_folder(
        repo_id=repo_id,
        folder_path=str(source_path),
        path_in_repo=path_in_repo,
        revision=revision,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )
    return {
        "repo_id": repo_id,
        "commit_oid": getattr(commit_info, "oid", None),
        "commit_url": getattr(commit_info, "commit_url", None),
        "path_in_repo": path_in_repo,
        "revision": revision,
    }


def upload_file(
    source_path: Path,
    *,
    repo_id: str,
    path_in_repo: str,
    private: bool = True,
    commit_message: str,
    revision: str | None = None,
    api: HfApi | None = None,
) -> dict[str, Any]:
    client = api or HfApi()
    ensure_repo(repo_id, private=private, api=client)
    commit_info = client.upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(source_path),
        path_in_repo=path_in_repo,
        revision=revision,
        repo_type="model",
        commit_message=commit_message,
    )
    return {
        "repo_id": repo_id,
        "commit_oid": getattr(commit_info, "oid", None),
        "commit_url": getattr(commit_info, "commit_url", None),
        "path_in_repo": path_in_repo,
        "revision": revision,
    }
