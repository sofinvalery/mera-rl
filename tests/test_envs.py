from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ENV_ROOT = Path(__file__).resolve().parents[1] / "environments"


def get_environments() -> list[Path]:
    return sorted(
        env_dir
        for env_dir in ENV_ROOT.iterdir()
        if env_dir.is_dir() and (env_dir / "pyproject.toml").exists()
    )


def test_environment_count() -> None:
    envs = get_environments()
    assert envs, "No environments found under environments/"


def test_pyproject_has_metadata() -> None:
    for env_dir in get_environments():
        with (env_dir / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)

        project = pyproject.get("project", {})
        assert project.get("name"), f"{env_dir.name}: missing project.name"
        assert project.get("version"), f"{env_dir.name}: missing project.version"
        assert project.get("description"), f"{env_dir.name}: missing project.description"
        assert project.get("tags"), f"{env_dir.name}: missing project.tags"
        assert project["description"] != "Your environment description here", (
            f"{env_dir.name}: placeholder description"
        )
        assert project["tags"] != ["placeholder-tag", "train", "eval"], (
            f"{env_dir.name}: placeholder tags"
        )


def test_readme_exists() -> None:
    for env_dir in get_environments():
        assert (env_dir / "README.md").exists(), f"{env_dir.name}: missing README.md"


def test_environment_module_defines_load_environment() -> None:
    for env_dir in get_environments():
        build_include = [f"{env_dir.name}.py"]
        with (env_dir / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)
            build_include = (
                pyproject.get("tool", {})
                .get("hatch", {})
                .get("build", {})
                .get("include", build_include)
            )

        module_candidates = [Path(item) for item in build_include if str(item).endswith(".py")]
        assert module_candidates, f"{env_dir.name}: no Python module in [tool.hatch.build].include"

        module_path = env_dir / module_candidates[0]
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
        has_loader = any(
            isinstance(node, ast.FunctionDef) and node.name == "load_environment"
            for node in tree.body
        )
        assert has_loader, f"{env_dir.name}: missing load_environment(...) in {module_path.name}"
