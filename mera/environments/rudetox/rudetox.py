from __future__ import annotations

import sys
from pathlib import Path

env_root = Path(__file__).resolve().parents[1]
if str(env_root) not in sys.path:
    sys.path.insert(0, str(env_root))

from mera_common import load_task_environment


def load_environment(split: str = "test", fast: bool = False, **kwargs):
    return load_task_environment("rudetox", split=split, fast=fast, **kwargs)
